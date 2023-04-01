import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
import numpy as np
import tqdm
import glob, json, os

def set_feature(x, features):
    assert(x.F.shape[0] == features.shape[0])
    if isinstance(x, ME.TensorField):
        return ME.TensorField(
            features=features.to(x.device),
            coordinate_field_map_key=x.coordinate_field_map_key,
            coordinate_manager=x.coordinate_manager,
            quantization_mode=x.quantization_mode,
            device=x.device,
        )
    elif isinstance(x, ME.SparseTensor):
        return ME.SparseTensor(
            features=features.to(x.device),
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager,
            device=x.device,
        )
    else:
        pass
    raise ValueError("Input tensor is not ME.TensorField nor ME.SparseTensor.")
    return None


def inverse_permutation(perm):
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.size(0), device=perm.device)
    return inv


class Identity(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class UnbatchedMinkowski(nn.Module):

    def __init__(self, module):
        super().__init__()
        # a PyTorch 1D module/layer
        self.module = module

    def forward(self, x):
        list_permutations = x.decomposition_permutations
        list_feats = list(map(
            lambda perm: self.module(
                x.F[perm].transpose(0, 1).unsqueeze(0) # shape of (1 (batch), C, N)
            )[0].transpose(0, 1), # shape of (N, C)
            list_permutations,
        ))
        ft = torch.cat(list_feats, dim=0)
        perm = torch.cat(list_permutations, dim=0)
        inv_perm = inverse_permutation(perm)
        return set_feature(x, ft[inv_perm])


class BatchedMinkowski(nn.Module):

    def __init__(self, module):
        super().__init__()
        # a PyTorch 1D module/layer
        self.module = module

    def forward(self, x):
        return set_feature(x, self.module(x.F))


class ConvBlock(nn.Module):

    def __init__(self, ch_in, ch_out, ks=3, groups=8, act="SiLU"):
        super().__init__()
        self.conv = ME.MinkowskiConvolution(
            ch_in, ch_out, kernel_size=ks, stride=1, dilation=1, dimension=3,
        )
        self.norm = UnbatchedMinkowski(
            nn.GroupNorm(num_groups=groups, num_channels=ch_out)
        )
        if act is None or act == "":
            self.act = Identity()
        else:
            assert(type(act) == str)
            self.act = getattr(torch.nn.functional, act.lower())

    def forward(self, x, scale_shift=None):
        x = self.conv(x)
        x = self.norm(x)
        if scale_shift is not None:
            scale, shift = scale_shift
            idx = x.C[:, 0].long() # batch index
            x = set_feature(x, x.F * (scale[idx] + 1) + shift[idx])
        return set_feature(x, self.act(x.F))


class ResNetBlock(nn.Module):

    def __init__(self, ch_in, ch_out, ch_time=None, act="SiLU"):
        super().__init__()
        self.mlp = nn.Sequential(
            getattr(nn, act)(),
            nn.Linear(ch_time, ch_out * 2)
        ) if ch_time is not None else None
        self.block1 = ConvBlock(ch_in, ch_out, act=act)
        self.block2 = ConvBlock(ch_out, ch_out, act=act)
        self.res_conv = ME.MinkowskiConvolution(
            ch_in, ch_out, kernel_size=1, stride=1, dilation=1, dimension=3, bias=True
        ) if ch_in != ch_out else Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb) # B, ch_out * 2
            scale_shift = time_emb.chunk(2, dim=1) # (B, ch_out) and (B, ch_out)
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)

# https://github.com/lucidrains/linear-attention-transformer
class LinearAttention(nn.Module):
    def __init__(self, ch, num_head=4, ch_head=32, norm=True, residual=True):
        super().__init__()
        self.num_head = num_head
        self.ch_head = ch_head
        self.scale = ch_head ** -0.5
        self.norm = UnbatchedMinkowski(
            nn.GroupNorm(num_groups=1, num_channels=ch) # LayerNorm
        ) if norm else Identity()
        self.to_qkv = ME.MinkowskiConvolution(
            ch, ch_head * num_head * 3,
            kernel_size=1, stride=1, dilation=1, dimension=3,
        )
        self.to_out = nn.Sequential(
            ME.MinkowskiConvolution(
                ch_head * num_head, ch,
                kernel_size=1, stride=1, dilation=1, dimension=3,
            ),
            UnbatchedMinkowski(
                nn.GroupNorm(num_groups=1, num_channels=ch) # LayerNorm
            )
        )
        self.residual = residual

    def forward(self, x):
        norm_x = self.norm(x)
        qkv = self.to_qkv(norm_x)
        list_permutations = qkv.decomposition_permutations
        
        q, k, v = map(
            lambda t: t.reshape(-1, self.num_head, self.ch_head),
            qkv.F.chunk(3, dim=1)
        ) # (spatial, num_head, ch_head)

        list_context = list(map(
            lambda perm: torch.einsum('ihk, ihv -> hkv', (k[perm] / np.sqrt(perm.size(0))).softmax(dim=0), v[perm]),
            list_permutations
        )) # list of (num_head, ch_head_key, ch_head_value)

        q = q * self.scale
        list_out = list(map(
            lambda context, perm: torch.einsum('ihk, hkv -> ihv', q[perm].softmax(dim=-1), context)
            .reshape(-1, self.num_head * self.ch_head),
            list_context, list_permutations
        )) # list of (spatial, num_head * ch_head)

        ft = torch.cat(list_out, dim=0)
        perm = torch.cat(list_permutations, dim=0)
        inv_perm = inverse_permutation(perm)
        
        out = self.to_out(set_feature(qkv, ft[inv_perm]))
        if self.residual:
            out = out + x
        
        return out


class Attention(nn.Module):

    def __init__(self, ch, num_head=4, ch_head=32, norm=True, residual=True):
        super().__init__()
        self.num_head = num_head
        self.ch_head = ch_head
        self.scale = ch_head ** -0.5
        self.norm = UnbatchedMinkowski(
            nn.GroupNorm(num_groups=1, num_channels=ch) # LayerNorm
        ) if norm else Identity()
        self.to_qkv = ME.MinkowskiConvolution(
            ch, num_head * ch_head * 3,
            kernel_size=1, stride=1, dilation=1, dimension=3,
        )
        self.to_out = ME.MinkowskiConvolution(
            num_head * ch_head, ch,
            kernel_size=1, stride=1, dilation=1, dimension=3, bias=True
        )
        self.residual = residual

    def forward(self, x):
        norm_x = self.norm(x)
        qkv = self.to_qkv(norm_x)
        list_permutations = qkv.decomposition_permutations
        
        q, k, v = map(
            lambda t: t.reshape(-1, self.num_head, self.ch_head),
            qkv.F.chunk(3, dim=1)
        ) # (spatial, num_head, ch_head)
        q = q * self.scale

        list_attn = list(map(
            lambda perm: torch.einsum('ihd, jhd -> hij', q[perm], k[perm]).softmax(dim=-1),
            list_permutations
        )) # list of (num_head, spatial, spatial)

        list_out = list(map(
            lambda attn, perm: torch.einsum('hij, jhd -> ihd', attn, v[perm]).reshape(-1, self.num_head * self.ch_head),
            list_attn, list_permutations
        )) # list of (spatial, num_head * ch_head)

        ft = torch.cat(list_out, dim=0)
        perm = torch.cat(list_permutations, dim=0)
        inv_perm = inverse_permutation(perm)

        out = self.to_out(set_feature(qkv, ft[inv_perm]))
        if self.residual:
            out = out + x

        return out


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, times):
        half_dim = self.dim // 2
        emb = np.log(1000) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=times.device) * -emb)
        emb = times[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


ATTN_LAYER = {
    "L": LinearAttention,
    "A": Attention,
    "N": Identity,
}


class MinkUNet(nn.Module):
    BLOCK = ResNetBlock
    PLANES = (32, 64, 128, 256, 512, 256, 128, 64, 32)
    REPEATS = (2, 2, 2, 2, 2, 2, 2, 2, 2, 2)
    # ATTNS = "L L L L L L L L L L".split()
    # ATTNS = "L L L L N N L L L L".split()
    # ATTNS = "N N N N A A N N N N".split()
    ATTNS = "N N N N N N N N N N".split()
    # "L", "A", "N" for linear/general/no attention

    def __init__(self, in_channels, out_channels, time_channels=None, attns=None):
        super().__init__()
        assert len(self.PLANES) % 2 == 1
        assert len(self.REPEATS) == (len(self.PLANES) + 1)
        self.levels = len(self.PLANES) // 2
        if attns is not None:
            self.ATTNS = attns.split()
            print(self.ATTNS)
            input()
        self.init_network(in_channels, out_channels, time_channels)
        self.init_weight()

    def init_network(self, in_channels, out_channels, time_channels=None):
        self.time_mlp = None
        if time_channels is not None:
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_channels // 4),
                nn.Linear(time_channels // 4, time_channels),
                nn.GELU(),
                nn.Linear(time_channels, time_channels)
            )

        self.init_conv = ConvBlock(in_channels, self.PLANES[0], ks=5)

        # Encoder
        self.enc_blocks = nn.ModuleList([])
        self.downs = nn.ModuleList([])
        for i in range(self.levels + 1):
            ch = self.PLANES[i]
            ch_next = self.PLANES[i + 1] if i < self.levels else None
            blocks = nn.ModuleList([])
            for j in range(self.REPEATS[i]):
                blocks.append(nn.ModuleList([
                    ResNetBlock(ch, ch, ch_time=time_channels),
                    ATTN_LAYER[self.ATTNS[i]](ch) if j == self.REPEATS[i] - 1 else Identity(),
                ]))
            self.enc_blocks.append(blocks)
            self.downs.append(
                ME.MinkowskiConvolution(ch, ch_next, kernel_size=2, stride=2, dimension=3)
                if i < self.levels else Identity()
            )
        
        # Mid
        mid_dim = self.PLANES[self.levels]
        self.mid_block1 = ResNetBlock(mid_dim, mid_dim, ch_time=time_channels)
        self.mid_attn = Attention(mid_dim)
        self.mid_block2 = ResNetBlock(mid_dim, mid_dim, ch_time=time_channels)

        # Decoder
        self.dec_blocks = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        for i in range(self.levels, len(self.PLANES)):
            ch_skip = self.PLANES[len(self.PLANES) - 1 - i]
            ch = self.PLANES[i]
            ch_next = self.PLANES[i + 1] if i < len(self.PLANES) - 1 else None
            blocks = nn.ModuleList([])
            for j in range(self.REPEATS[i]):
                blocks.append(nn.ModuleList([
                    ResNetBlock(ch + ch_skip, ch, ch_time=time_channels),
                    ATTN_LAYER[self.ATTNS[i]](ch) if j == self.REPEATS[i] - 1 else Identity(),
                ]))
            self.dec_blocks.append(blocks)
            self.ups.append(
                ME.MinkowskiConvolutionTranspose(ch, ch_next, kernel_size=2, stride=2, dimension=3)
                if i < len(self.PLANES) - 1 else Identity()
            )

        self.final_block = ResNetBlock(self.PLANES[-1] + self.PLANES[0], self.PLANES[-1], ch_time=time_channels)
        self.final_conv = ME.MinkowskiConvolution(
            self.PLANES[-1], out_channels, kernel_size=1, stride=1, dimension=3
        )

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

    def forward(self, x, times=None):
        time_emb = None
        if self.time_mlp is not None and times is not None:
            time_emb = self.time_mlp(times)

        x = self.init_conv(x)
        stack = [x]
        for blocks, down in zip(self.enc_blocks, self.downs):
            for block, attn in blocks:
                x = block(x, time_emb=time_emb)
                x = attn(x)
                stack.append(x)
            x = down(x)
        
        x = self.mid_block1(x, time_emb=time_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, time_emb=time_emb)

        for blocks, up in zip(self.dec_blocks, self.ups):
            for block, attn in blocks:
                x = ME.cat(x, stack.pop())
                x = block(x, time_emb=time_emb)
                x = attn(x)
            
            x = up(x)
        
        x = ME.cat(x, stack.pop())
        assert(len(stack) == 0)
        x = self.final_block(x, time_emb=time_emb)
        return self.final_conv(x)


class MinkFieldUNet(MinkUNet):

    def init_network(self, in_channels, out_channels, time_channels=None):
        field_ch1 = 32
        field_ch2 = 32
        self.field_network1 = nn.Sequential(
            ME.MinkowskiSinusoidal(in_channels, field_ch1),
            UnbatchedMinkowski(nn.GroupNorm(num_groups=8, num_channels=field_ch1)),
            BatchedMinkowski(nn.GELU()),
            ME.MinkowskiLinear(field_ch1, field_ch1),
            UnbatchedMinkowski(nn.GroupNorm(num_groups=8, num_channels=field_ch1)),
            BatchedMinkowski(nn.GELU()),
        )
        self.field_network2 = nn.Sequential(
            ME.MinkowskiSinusoidal(field_ch1 + in_channels, field_ch2),
            UnbatchedMinkowski(nn.GroupNorm(num_groups=8, num_channels=field_ch1)),
            BatchedMinkowski(nn.GELU()),
            ME.MinkowskiLinear(field_ch2, field_ch2),
            UnbatchedMinkowski(nn.GroupNorm(num_groups=8, num_channels=field_ch1)),
            BatchedMinkowski(nn.GELU()),
        )
        MinkUNet.init_network(self, field_ch2, out_channels, time_channels)

    def forward(self, x: ME.TensorField, times=None):
        otensor1 = self.field_network1(x)
        otensor1 = ME.cat(otensor1, x)
        otensor2 = self.field_network2(otensor1)
        otensor2 = otensor2.sparse()
        out = MinkUNetBase.forward(self, otensor2, times)
        return out.slice(x)


def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def extract(param, times, x):
    # times in a shape of (B,)
    # x is a minkowski sparse tensor
    assert(len(times.shape) == 1)
    return set_feature(x, param[times.long()][x.C[:, 0].long()][:, None])


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        timesteps=1000,
        sampling_timesteps=1000,
        loss_type="smooth_l1",
        objective="pred_noise",
        beta_schedule="sigmoid",
        schedule_fn_kwargs=dict(),
        p2_loss_weight_gamma=0.5, # p2 loss weight, 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k=1,
        ddim_sampling_eta=0.,
    ):
        super().__init__()

        self.model = model
        self.self_condition = False

        self.objective = objective

        assert objective in {"pred_noise", "pred_x0", "pred_v"}, "objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])"

        if beta_schedule == "linear":
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == "cosine":
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == "sigmoid":
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f"unknown beta schedule {beta_schedule}")

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        # self.sampling_timesteps = timesteps # default num sampling timesteps to number of timesteps at training
        self.sampling_timesteps = sampling_timesteps

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate p2 reweighting
        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1. - alphas_cumprod)) ** -p2_loss_weight_gamma)
    
    def predict_start_from_noise(self, x_t, t, noise):
        assert(x_t.coordinate_map_key == noise.coordinate_map_key)
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, noise) * noise
        )

    def predict_noise_from_start(self, x_t, t, x_0):
        assert(x_t.coordinate_map_key == x_0.coordinate_map_key)
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t) * x_t - x_0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t)
        )

    def predict_v(self, x_start, t, noise):
        assert(x_start.coordinate_map_key == noise.coordinate_map_key)
        return (
            extract(self.sqrt_alphas_cumprod, t, noise) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        assert(x_t.coordinate_map_key == v.coordinate_map_key)
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, v) * v
        )

    def q_posterior(self, x_start, x_t, t):
        assert(x_start.coordinate_map_key == x_t.coordinate_map_key)
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_start) * x_start +
            extract(self.posterior_mean_coef2, t, x_t) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = set_feature(x_start, torch.randn_like(x_start.F))
        else:
            assert(x_start.coordinate_map_key == noise.coordinate_map_key)
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start) * noise
        )

    def model_predictions(self, x, t, x_self_cond=None, clip_x_start=False):
        model_output = self.model(x, t)#, x_self_cond)
        clip = BatchedMinkowski(lambda y: torch.clamp(y, -1., 1.)) if clip_x_start else Identity()

        if self.objective == "pred_noise":
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = clip(x_start)

        elif self.objective == "pred_x0":
            x_start = model_output
            x_start = clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == "pred_v":
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)
        
        else:
            assert(False)

        return pred_noise, x_start

    def p_mean_variance(self, x, t, x_self_cond=None, clip_denoised=True):
        pred_noise, x_start = self.model_predictions(x, t, x_self_cond, clip_denoised)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, x_self_cond=None):
        batch_size = x.C[:, 0].int().max() + 1
        batched_times = torch.full((batch_size,), t, device=x.F.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x, t=batched_times, x_self_cond=x_self_cond, clip_denoised=True
        )
        noise = set_feature(x, torch.randn_like(x.F) if t > 0 else torch.zeros_like(x.F)) # no noise if t == 0
        pred_back = model_mean + set_feature(model_log_variance, (0.5 * model_log_variance.F).exp()) * noise
        return pred_back, x_start

    @torch.no_grad()
    def p_sample_loop(self, x):
        x = set_feature(x, torch.randn_like(x.F))
        x_start = None
        for t in tqdm.tqdm(reversed(range(0, self.num_timesteps)), desc="sampling loop time step", total=self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            x, x_start = self.p_sample(x, t, self_cond)
            # if t % 50 == 0:
            #     with open(f'testoverfit80/{t}.txt', 'w') as f:
            #         for (_, xx, yy, zz), (r, g, b) in zip(
            #             x.C.cpu().numpy(),
            #             (torch.clamp(x.F, -1, 1) * 127.5 + 127.5).cpu().numpy().astype(np.uint8),
            #         ):
            #             f.write('%.3lf %.3lf %.3lf %d %d %d\n' % (xx, yy, zz, r, g, b))
        return x

    @torch.no_grad()
    def ddim_sample(self, x): # for faster sampling
        batch_size = x.C[:, 0].int().max() + 1
        times = torch.linspace(-1, self.num_timesteps-1, steps=self.sampling_timesteps+1)
        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == num_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x = set_feature(x, torch.randn_like(x.F))
        x_start = None
        for time, time_next in tqdm.tqdm(time_pairs, desc="sampling loop time step"):
            batched_times = torch.full((batch_size,), time, device=x.F.device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start = self.model_predictions(x, batched_times, self_cond, clip_x_start=True)

            if time_next < 0:
                break

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = self.ddim_sampling_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            x = set_feature(x_start, x_start.F * alpha_next.sqrt()) + \
                set_feature(pred_noise, c * pred_noise.F) + \
                set_feature(x, sigma * torch.randn_like(x.F))

        return x_start

    @torch.no_grad()
    def sample(self, x):
        sample_fn = self.ddim_sample if self.is_ddim_sampling else self.p_sample_loop
        return sample_fn(x)

    @property
    def loss_fn(self):
        if self.loss_type == "l1":
            return F.l1_loss
        elif self.loss_type == "l2":
            return F.mse_loss
        elif self.loss_type == "smooth_l1":
            return F.smooth_l1_loss
        else:
            raise ValueError(f"invalid loss type {self.loss_type}")

    def p_losses(self, x_start, t):
        # noise sample
        noise = set_feature(x_start, torch.randn_like(x_start.F))
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly
        # x_self_cond = None
        # if self.self_condition and np.random() < 0.5:
        #     with torch.no_grad():
        #         _, x_self_cond = self.model_predictions(x, t)
        #         x_self_cond.detach_()

        # predict and take gradient step
        model_out = self.model(x, t)#, x_self_cond)

        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x_start
        elif self.objective == "pred_v":
            target = self.predict_v(x_start, t, noise)
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = self.loss_fn(model_out.F, target.F, reduction="none").mean(dim=1)
        loss *= self.p2_loss_weight[t.long()][model_out.C[:, 0].long()]
        return loss.mean()

    def forward(self, x, fix_t=None):
        batch_size = x.C[:, 0].int().max() + 1
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=x.F.device).long()
        if fix_t:
            t *= 0
            t += fix_t
        return self.p_losses(x, t)





if __name__ == "__main__":

    import argparse
    import sys

    sys.path.append("/home/lzq/lzy/torch-ngp")
    # sys.path.append("/home/lzq/lzy/HoliCity/")
    from main_nerf import get_args

    opt = get_args()

    if opt.O:
        opt.fp16 = True
        opt.cuda_ray = True
        opt.preload = True
    
    if opt.patch_size > 1:
        opt.error_map = False # do not use error_map if use patch-based training
        # assert opt.patch_size > 16, "patch_size should > 16 to run LPIPS loss."
        assert opt.num_rays % (opt.patch_size ** 2) == 0, "patch_size ** 2 should be dividable by num_rays."


    if opt.ff:
        opt.fp16 = True
        assert opt.bg_radius <= 0, "background model is not implemented for --ff"
    elif opt.tcnn:
        opt.fp16 = True
        assert opt.bg_radius <= 0, "background model is not implemented for --tcnn"
    else:
        pass
    
    from nerf.provider import NeRFDataset

    scale = 10
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    func = lambda x: torch.cat([x, x[:1]], dim=0)
    train_loader = NeRFDataset(opt, device=device, type=opt.diff_arg_dataset).dataloader_my(batch_size=1)

    # vals = [[], [], []]
    # for data_idx, data in enumerate(train_loader):
    #     pts = data['pts_coords'][data['pts_masks']].cpu()
    #     for i in range(3):
    #         val, bins = torch.histogram(pts[:, i], bins=400, range=(-200, 200))
    #         vals[i].append(val)
    #     print(data_idx)
    # import matplotlib; matplotlib.use("agg")
    # import matplotlib.pyplot as plt
    # rs = [(-64, 64), (-64, 64), (-5, 27)]
    # for i in range(3):
    #     s = torch.stack(vals[i]).sum(dim=0).numpy()
    #     plt.plot(bins.numpy()[:-1], s, "-")
    #     plt.savefig(f"{i}.png")
    #     plt.clf()
    #     a, b = rs[i]
    #     print(s[a+200: b+200].sum() / s.sum())
    
    # quit()

    # for data in train_loader:
    #     pts = data['pts_coords'][data['pts_masks']].to(device)
    #     pts_batch = data['pts_batch'][data['pts_masks']].unsqueeze(-1).to(device)
    #     feats = func(data['pts_rgbs'][data['pts_masks']]).to(device)
    #     coords = func(torch.cat([pts_batch, pts * scale], dim=-1))
    #     pts_field = ME.TensorField(
    #         features=feats * 2 - 1, # in a range of -1 to 1
    #         coordinates=coords,
    #         quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
    #         minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
    #         device=pts.device,
    #     )
    #     pts_sparse = pts_field.sparse()
    #     torch.save({
    #         "features": pts_sparse.F,
    #         "coordinates": pts_sparse.C,
    #     }, f"test_point_cloud_scale_20.pt")
    # quit()

    # count = 0
    # for data in train_loader:
    #     with open(f'test_newsurface/datavis/{count:04d}.txt', 'w') as f:
    #         for (x, y, z), (r, g, b) in zip(
    #             data['pts_coords'][data['pts_masks']].cpu().numpy(),
    #             (255 * data['pts_rgbs'][data['pts_masks']].cpu().numpy()).astype(np.uint8),
    #         ):
    #             f.write('%.3lf %.3lf %.3lf %d %d %d\n' % (x, y, z, r, g, b))
    #         count += 1
    #         # if count % 8 == 0:
    #         input('check')

    net = MinkUNet(3, 3, time_channels=32, attns=opt.diff_arg_net_attention).to(device)
    diffusion_model = GaussianDiffusion(net, sampling_timesteps=opt.diff_arg_sampling_steps).to(device)

    optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=1e-3, betas=(0.9, 0.99))

    # folder = "test190furtherperturb"
    # folder = "test1520newsurface"
    # folder = "test1520constdepth"
    # folder = "test190dataaug"
    # folder = "test1520dataaugattn2"
    folder = opt.diff_arg_folder
    if opt.diff_arg_ckpt is not None:
        diffusion_model.load_state_dict(torch.load(f"{folder}/{opt.diff_arg_ckpt}"))
        import os
        os.system(f"mkdir {folder}/datavis")

    # pcfiles = sorted(glob.glob("/home/lzq/lzy/torch-ngp/data/holicity_single_view/point_clouds/*.txt"))#[:640]
    # pcfiles = pcfiles[632:648] + pcfiles[1528:1536]

    # from make_ngp_singleview_dataset import HoliCityDataset
    # dataset = HoliCityDataset('/home/lzq/lzy/HoliCity', 'train', since_month='2018-01')

    # pcfiles = sorted(
    #     [f"/home/lzq/lzy/torch-ngp/data/holicity_single_view/point_clouds/{file.split('/')[-1]}.txt" for file in dataset.filelist[:680]]
    # )[:24]

    # import os
    # with open("/home/lzq/lzy/torch-ngp/data/holicity_single_view/transforms.json", 'r') as f:
    #     names = [f['file_path'].split("/")[-1].replace(".jpg", "") for f in json.load(f)["frames"]]
    # pcfiles = [f"/home/lzq/lzy/torch-ngp/data/holicity_single_view/point_clouds/{name}.txt" for name in names][:1520]
    # pcfiles = pcfiles[:1520]
    # pcfiles.sort(key=lambda x: os.path.getmtime(x))
    # import os
    # for pcfile in pcfiles:
    #     if not os.path.exists(pcfile):
    #         # print(os.path.getmtime(pcfile), pcfile)
    #         print(pcfile)
    # quit()

    logger_file = "log.txt" if opt.diff_arg_dataset == "train" else "log_no.txt"
    num_epochs = 99999 if opt.diff_arg_dataset == "train" else 1

    with open(f"{folder}/{logger_file}", "w") as f:
        for epoch in range(0, num_epochs):
            data_idx = 0
            # diffusion_model.load_state_dict(torch.load(f"{folder}/epoch{epoch}.pt"))
            for data in train_loader:
            # for pcfile in pcfiles:

                # if not os.path.exists(pcfile):
                #     continue

                optimizer.zero_grad(set_to_none=True)

                pts = data['pts_coords'][data['pts_masks']].to(device)
                pts_batch = data['pts_batch'][data['pts_masks']].unsqueeze(-1).to(device) * 0
                feats = data['pts_rgbs'][data['pts_masks']].to(device)
                # data = np.loadtxt(pcfile)
                # pts = torch.from_numpy(data[:,:3]).to(device)
                # pts_batch = pts[:,0:1] * 0
                # feats = torch.from_numpy(data[:,3:]).float().to(device) / 255.0

                if opt.random_x_flip and np.random.rand() < 0.5:
                    pts[:, 0] *= -1

                if opt.random_y_flip and np.random.rand() < 0.5:
                    pts[:, 1] *= -1
                
                if opt.random_z_rotate:
                    rand_theta = np.random.rand() * 2 * np.pi
                    pts = torch.Tensor([
                        [np.cos(rand_theta), -np.sin(rand_theta), 0],
                        [np.sin(rand_theta), np.cos(rand_theta), 0],
                        [0, 0, 1],
                    ]).float().to(pts.device) @ pts.transpose(0, 1)
                    pts = pts.transpose(0, 1)
                
                if opt.random_gamma_correction:
                    assert(feats.min() >= 0)
                    assert(feats.max() <= 1)
                    max_gamma = 1.6
                    gamma = np.random.rand() * (max_gamma - 1) + 1
                    if np.random.rand() < 0.5:
                        gamma = 1.0 / gamma
                    feats = feats ** gamma

                if False: # crop point cloud, only closer points are kept
                    close_pts_mask = torch.sqrt((pts ** 2).sum(dim=1)) < 50
                    pts = pts[close_pts_mask]
                    pts_batch = pts_batch[close_pts_mask]
                    feats = feats[close_pts_mask]
                
                if False: # randomly disturb the location
                    pts = pts + torch.rand_like(pts) / scale

                coords = func(torch.cat([pts_batch, pts * scale], dim=-1))
                feats = func(feats)
                pts_field = ME.TensorField(
                    features=feats * 2 - 1, # in a range of -1 to 1
                    coordinates=coords,
                    quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                    minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
                    device=pts.device,
                )
                pts_sparse = pts_field.sparse()

                if False: # select xxx% of the points
                    perm = torch.randperm(pts_sparse.F.shape[0])
                    bound = int(pts_sparse.F.shape[0] * 0.25)
                    pts_sparse = ME.SparseTensor(
                        features=pts_sparse.F[perm[:bound]],
                        coordinates=pts_sparse.C[perm[:bound]],
                        device=pts.device,
                    )
                
                # for gamma in [1/5, 1/4, 1/3, 1/2, 1, 2, 3, 4, 5]:
                #     with open(f"{folder}/datavis/data_gamma_{gamma:02f}.txt", "w") as f:
                #         for (_, x, y, z), (r, g, b) in zip(
                #             pts_sparse.C.cpu().numpy(),
                #             ((((pts_sparse.F + 1) / 2) ** gamma) * 255).cpu().numpy().astype(np.uint8),
                #         ):
                #             f.write('%.3lf %.3lf %.3lf %d %d %d\n' % (x, y, z, r, g, b))
                # quit()

                if opt.diff_arg_dataset != "train":
                    with torch.no_grad():
                        pred = diffusion_model.sample(pts_sparse)
                        with open(f"{folder}/datavis/data_{data_idx}.txt", "w") as f:
                            for (_, x, y, z), (r, g, b) in zip(
                                pred.C.cpu().numpy(),
                                (torch.clamp(pred.F, -1, 1) * 127.5 + 127.5).cpu().numpy().astype(np.uint8),
                            ):
                                f.write('%.3lf %.3lf %.3lf %d %d %d\n' % (x, y, z, r, g, b))
                    data_idx += 1
                    continue

                # for ttt in [100, 300, 500, 700, 900]:
                optimizer.zero_grad(set_to_none=True)
                loss = diffusion_model(pts_sparse)#, fix_t=ttt)
                print(epoch, loss.item())
                f.write(f"{epoch}\t{loss}\n")
                f.flush()
                loss.backward()

                optimizer.step()

            if epoch % 10 == 0:
                state_dict = diffusion_model.state_dict()
                torch.save(state_dict, f"{folder}/epoch{epoch}.pt")

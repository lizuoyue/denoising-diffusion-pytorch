import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
import numpy as np
import tqdm
import glob, json, os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib; matplotlib.use("agg")
import matplotlib.pyplot as plt

def set_feature(x, features, same_manager=True):
    assert(x.F.shape[0] == features.shape[0])
    if isinstance(x, ME.TensorField):
        if same_manager:
            return ME.TensorField(
                features=features.to(x.device),
                coordinate_field_map_key=x.coordinate_field_map_key,
                coordinate_manager=x.coordinate_manager,
                quantization_mode=x.quantization_mode,
                device=x.device,
            )
        else:
            return ME.TensorField(
                features=features.to(x.device),
                coordinates=x.C,
                quantization_mode=x.quantization_mode,
                device=x.device,
            )
    elif isinstance(x, ME.SparseTensor):
        if same_manager:
            return ME.SparseTensor(
                features=features.to(x.device),
                coordinate_map_key=x.coordinate_map_key,
                coordinate_manager=x.coordinate_manager,
                device=x.device,
            )
        else:
            return ME.SparseTensor(
                features=features.to(x.device),
                coordinates=x.C,
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


class LinearBlock(nn.Module):

    def __init__(self, ch_in, ch_out, groups=8, act="SiLU"):
        super().__init__()
        self.conv = ME.MinkowskiLinear(ch_in, ch_out)
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
        )) # list of (num_head, ch_key, ch_value)

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


class CrossAttention(nn.Module):
    def __init__(self, ch, context_ch, num_head=8, ch_head=64, residual=True):
        super().__init__()

        self.scale = ch_head ** -0.5
        self.num_head = num_head
        self.ch_head = ch_head

        self.to_q = ME.MinkowskiConvolution(
            ch, num_head * ch_head,
            kernel_size=1, stride=1, dilation=1, dimension=3, bias=False,
        )
        self.to_k = nn.Linear(context_ch, num_head * ch_head, bias=False)
        self.to_v = nn.Linear(context_ch, num_head * ch_head, bias=False)

        self.to_out = ME.MinkowskiConvolution(
            num_head * ch_head, ch,
            kernel_size=1, stride=1, dilation=1, dimension=3, bias=True,
        )
        # TODO: dropout

    def forward(self, x, context):
        # x: minkowski tensor
        # context: (B, H*W, ch)
        h = self.heads
        list_permutations = x.decomposition_permutations
        batch_range = list(range(len(list_permutations)))

        q = self.to_q(x)
        qF = q.F.reshape(-1, self.num_head, self.ch_head) # (spatial, n, h)
        k = self.to_k(context).reshape(context.shape[0], context.shape[1], self.num_head, self.ch_head) #(B, H*W, n, h)
        v = self.to_v(context).reshape(context.shape[0], context.shape[1], self.num_head, self.ch_head) #(B, H*W, n, h)

        list_attn = list(map(
            lambda idx, perm: (self.scale * torch.einsum('qnh, knh -> nqk', qF[perm], k[idx])).softmax(dim=-1),
            batch_range, list_permutations
        )) # list of (num_head, q_spatial, H*W)

        list_out = list(map(
            lambda idx, attn: torch.einsum('nqv, vnh -> qnh', attn, v[idx]).reshape(-1, self.num_head * self.ch_head),
            batch_range, list_attn
        )) # list of (q_spatial, num_head * ch_head)

        ft = torch.cat(list_out, dim=0)
        perm = torch.cat(list_permutations, dim=0)
        inv_perm = inverse_permutation(perm)

        out = self.to_out(set_feature(q, ft[inv_perm]))
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

    def forward(self, x, times=None, x_self_cond=None):
        time_emb = None
        if self.time_mlp is not None and times is not None:
            time_emb = self.time_mlp(times)
        
        if x_self_cond is not None:
            x = ME.cat(x, x_self_cond)

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
        field_ch2 = 64
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
            UnbatchedMinkowski(nn.GroupNorm(num_groups=8, num_channels=field_ch2)),
            BatchedMinkowski(nn.GELU()),
            ME.MinkowskiLinear(field_ch2, field_ch2),
            UnbatchedMinkowski(nn.GroupNorm(num_groups=8, num_channels=field_ch2)),
            BatchedMinkowski(nn.GELU()),
        )
        self.field_final_block = LinearBlock(field_ch2 * 2, field_ch1)
        self.field_final = ME.MinkowskiLinear(field_ch1, out_channels)
        MinkUNet.init_network(self, field_ch2, field_ch2, time_channels)

    def forward(self, x: ME.TensorField, times=None, x_self_cond=None):
        if x_self_cond is not None:
            x = ME.cat(x, x_self_cond)
        otensor1 = self.field_network1(x)
        otensor1 = ME.cat(otensor1, x)
        otensor2 = self.field_network2(otensor1)
        out = MinkUNet.forward(self, otensor2.sparse(), times)
        out_field = out.slice(x)
        out2 = self.field_final_block(ME.cat(out_field, otensor2))
        return self.field_final(out2)


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

def has_same_map_key(a, b):
    if isinstance(a, ME.TensorField):
        return a.coordinate_field_map_key == b.coordinate_field_map_key
    elif isinstance(a, ME.SparseTensor):
        return a.coordinate_map_key == b.coordinate_map_key
    else:
        assert(False)
    return None

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        timesteps=1000,
        sampling_timesteps=1000,
        self_condition=False,
        loss_type="smooth_l1",
        objective="pred_noise",
        beta_schedule="sigmoid",
        schedule_fn_kwargs=dict(),
        p2_loss_weight_gamma=0.0, # p2 loss weight, 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k=1,
        ddim_sampling_eta=0.,
    ):
        super().__init__()

        self.model = model
        self.self_condition = self_condition

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
        assert(has_same_map_key(x_t, noise))
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, noise) * noise
        )

    def predict_noise_from_start(self, x_t, t, x_0):
        assert(has_same_map_key(x_t, x_0))
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t) * x_t - x_0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t)
        )

    def predict_v(self, x_start, t, noise):
        assert(has_same_map_key(x_start, noise))
        return (
            extract(self.sqrt_alphas_cumprod, t, noise) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        assert(has_same_map_key(x_t, v))
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, v) * v
        )

    def q_posterior(self, x_start, x_t, t):
        assert(has_same_map_key(x_start, x_t))
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
            assert(has_same_map_key(x_start, noise))
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start) * noise
        )

    def model_predictions(self, x, t, x_self_cond=None, mask_info=None, clip_x_start=False):

        if mask_info is not None:
            # mask_info == 0 means known info
            # mask_info == 1 means to be generated
            x_num_channel = x.F.shape[1]
            masked_start = x.F * (1 - mask_info)
            x = set_feature(x, torch.cat([x.F, masked_start, mask_info], dim=-1))

        model_output = self.model(x, t, x_self_cond)
        clip = BatchedMinkowski(lambda y: torch.clamp(y, -1., 1.)) if clip_x_start else Identity()

        if mask_info is not None:
            x = set_feature(x, x.F[:, :x_num_channel])

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

    def p_mean_variance(self, x, t, x_self_cond=None, mask_info=None, clip_denoised=True):
        pred_noise, x_start = self.model_predictions(x, t, x_self_cond, mask_info, clip_denoised)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, x_self_cond=None, mask_info=None):
        batch_size = x.C[:, 0].int().max() + 1
        batched_times = torch.full((batch_size,), t, device=x.F.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x, t=batched_times, x_self_cond=x_self_cond, mask_info=mask_info, clip_denoised=True
        )
        noise = set_feature(x, torch.randn_like(x.F) if t > 0 else torch.zeros_like(x.F)) # no noise if t == 0
        pred_back = model_mean + set_feature(model_log_variance, (0.5 * model_log_variance.F).exp()) * noise
        return pred_back, x_start

    @torch.no_grad()
    def p_sample_loop(self, x, mask_info=None):
        x = set_feature(x, torch.randn_like(x.F))
        x_start = set_feature(x, torch.zeros_like(x.F)) if self.self_condition else None
        for t in tqdm.tqdm(reversed(range(0, self.num_timesteps)), desc="sampling loop time step", total=self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            x, x_start = self.p_sample(x, t, self_cond, mask_info)

            x = set_feature(x, x.F, same_manager=False)
            x_start = set_feature(x, x_start.F)
            # if t % 50 == 0:
            #     with open(f'testoverfit80/{t}.txt', 'w') as f:
            #         for (_, xx, yy, zz), (r, g, b) in zip(
            #             x.C.cpu().numpy(),
            #             (torch.clamp(x.F, -1, 1) * 127.5 + 127.5).cpu().numpy().astype(np.uint8),
            #         ):
            #             f.write('%.3lf %.3lf %.3lf %d %d %d\n' % (xx, yy, zz, r, g, b))
        return x

    @torch.no_grad()
    def ddim_sample(self, x, mask_info=None): # for faster sampling
        batch_size = x.C[:, 0].int().max() + 1
        times = torch.linspace(-1, self.num_timesteps-1, steps=self.sampling_timesteps+1)
        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == num_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x = set_feature(x, torch.randn_like(x.F))
        x_start = set_feature(x, torch.zeros_like(x.F)) if self.self_condition else None
        for time, time_next in tqdm.tqdm(time_pairs, desc="sampling loop time step"):
            batched_times = torch.full((batch_size,), time, device=x.F.device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start = self.model_predictions(x, batched_times, self_cond, mask_info, clip_x_start=True)

            if time_next < 0:
                break

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = self.ddim_sampling_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            x = set_feature(x_start, x_start.F * alpha_next.sqrt()) + \
                set_feature(pred_noise, c * pred_noise.F) + \
                set_feature(x, sigma * torch.randn_like(x.F))
            
            x = set_feature(x, x.F, same_manager=False)
            x_start = set_feature(x, x_start.F)

        return x_start

    @torch.no_grad()
    def sample(self, x, mask_info=None):
        sample_fn = self.ddim_sample if self.is_ddim_sampling else self.p_sample_loop
        return sample_fn(x, mask_info)

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

    def p_losses(self, x_start, t, spatial_weight=None, mask_info=None):
        # noise sample
        noise = set_feature(x_start, torch.randn_like(x_start.F))
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly
        x_self_cond = None
        if self.self_condition:
            if np.random.rand() < 0.5:
                with torch.no_grad():
                    x_copy = set_feature(x, x.F, same_manager=False)
                    _, x_self_cond = self.model_predictions(x_copy, t, set_feature(x_copy, torch.zeros_like(x.F)))
                    x_self_cond = set_feature(x, x_self_cond.F.detach())
            else:
                x_self_cond = set_feature(x, torch.zeros_like(x.F))

        # predict and take gradient step
        if mask_info is not None:
            # mask_info == 0 means known info
            # mask_info == 1 means to be generated
            masked_start = x_start.F * (1 - mask_info)
            x = set_feature(x, torch.cat([x.F, masked_start, mask_info], dim=-1))

        model_out = self.model(x, t, x_self_cond)

        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x_start
        elif self.objective == "pred_v":
            target = self.predict_v(x_start, t, noise)
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = self.loss_fn(model_out.F, target.F, reduction="none")

        if mask_info is not None:
            loss = loss * (mask_info + 0.1) # 0.1 vs 1.1

        if spatial_weight is None:
            loss = loss.mean(dim=1)
        else:
            spatial_weight /= spatial_weight.mean()
            loss = (loss * spatial_weight).mean(dim=1)
        loss *= self.p2_loss_weight[t.long()][model_out.C[:, 0].long()]
        return loss.mean()

    def forward(self, x, fix_t=None, spatial_weight=None, mask_info=None):
        batch_size = x.C[:, 0].int().max() + 1
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=x.F.device).long()
        if fix_t:
            t *= 0
            t += fix_t
        return self.p_losses(x, t, spatial_weight, mask_info)




class HoliCityPointCloudDataset(Dataset):
    def __init__(self, rootdir):
        self.pc_files = sorted(glob.glob(f"{rootdir}/*.npz"))

    def __len__(self):
        return len(self.pc_files)

    def __getitem__(self, idx):
        pc = np.load(self.pc_files[idx])
        coord = pc["coord"]
        color = pc["color"]
        dist = pc["dist"]
        sem = pc["sem"]
        is_not_ground = pc["geo_is_not_ground"]
        return coord, color, dist, sem, is_not_ground


def get_param_generator(model, model_dict):
    for name, param in model.named_parameters():
        if name not in model_dict or param.shape != model_dict[name].shape:
            yield param

def update_model_dict(model, model_dict):
    for name, param in model.named_parameters():
        if name in model_dict and param.shape != model_dict[name].shape:
            model_dict.pop(name)


def xyz2lonlatdist(coord):
    # coord: N, 3
    dist = torch.norm(coord, dim=-1, keepdim=True)
    normed_coord = coord / dist
    lat = torch.arcsin(normed_coord[:, 2]) # -pi/2 to pi/2
    lon = torch.arctan2(normed_coord[:, 0], normed_coord[:, 1]) # -pi to pi
    return lon, lat, dist[:, 0]

def xyz2coord(coord):
    # coord: N, 3
    lon, lat, dist = xyz2lonlatdist(coord)
    y = lat / (torch.pi / 2.0)
    x = lon / torch.pi
    dist /= dist.max()
    return torch.stack([-x * 2, y, dist], dim=-1)

import pytorch3d
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import compositing
from pytorch3d.renderer.points import rasterize_points

class RasterizePointsXYsBlending(nn.Module):
    """
    Rasterizes a set of points using a differentiable renderer. Points are
    accumulated in a z-buffer using an accumulation function
    defined in opts.accumulation and are normalised with a value M=opts.M.
    Inputs:
    - pts3D: the 3D points to be projected
    - src: the corresponding features
    - C: size of feature
    - learn_feature: whether to learn the default feature filled in when
                     none project
    - radius: where pixels project to (in pixels)
    - size: size of the image being created
    - points_per_pixel: number of values stored in z-buffer per pixel
    - opts: additional options
    Outputs:
    - transformed_src_alphas: features projected and accumulated
        in the new view
    """

    def __init__(
        self,
        radius=3.0,
        size=(256, 512),
        points_per_pixel=8,
        rad_pow=2,
        tau=1.0,
        accumulation="alphacomposite",
    ):
        super().__init__()
        self.radius = radius
        self.size = size
        self.points_per_pixel = points_per_pixel
        self.rad_pow = rad_pow
        self.tau = tau
        self.accumulation = accumulation

    def forward(self, mink_field):
        # mink_field: batch size 1
        pts3D = pytorch3d.structures.Pointclouds(
            points=xyz2coord(mink_field.C[:, 1:])[None, ...],
            features=mink_field.F[None, ...]
        )

        radius = float(self.radius) / float(self.size[0]) * 2.0

        points_idx, _, dist = rasterize_points(
            pts3D, self.size, radius, self.points_per_pixel
        )
        # dist = dist / pow(radius, self.rad_pow)
        alphas = (
            (1 - dist.clamp(max=1, min=1e-3).pow(0.5))
            .pow(self.tau)
            .permute(0, 3, 1, 2)
        )

        if self.accumulation == 'alphacomposite':
            transformed_src_alphas = compositing.alpha_composite(
                points_idx.permute(0, 3, 1, 2).long(),
                alphas,
                pts3D.features_packed().permute(1,0),
            )
        elif self.accumulation == 'wsum':
            transformed_src_alphas = compositing.weighted_sum(
                points_idx.permute(0, 3, 1, 2).long(),
                alphas,
                pts3D.features_packed().permute(1,0),
            )
        elif self.accumulation == 'wsumnorm':
            transformed_src_alphas = compositing.weighted_sum_norm(
                points_idx.permute(0, 3, 1, 2).long(),
                alphas,
                pts3D.features_packed().permute(1,0),
            )

        return transformed_src_alphas



if __name__ == "__main__":

    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_folder', type=str)
    parser.add_argument('--dataset_mode', type=str, default='train')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_epoch', type=int, default=20)
    parser.add_argument('--net_attention', type=str, default=None)
    parser.add_argument('--work_folder', type=str)
    parser.add_argument('--point_scale', type=int)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--ckpt_not_strict', action='store_true')
    parser.add_argument('--num_sample', type=int, default=3)
    parser.add_argument('--sampling_steps', type=int, default=1000)
    parser.add_argument('--random_x_flip', action='store_true')
    parser.add_argument('--random_y_flip', action='store_true')
    parser.add_argument('--random_z_rotate', action='store_true')
    parser.add_argument('--random_gamma_correction', action='store_true')
    parser.add_argument('--self_condition', action='store_true')
    parser.add_argument('--field_network', action='store_true')
    parser.add_argument('--masked_generation', type=float, default=0.0) # ratio of known infomation
    parser.add_argument('--save_gt', action='store_true')
    parser.add_argument('--use_ema', action='store_true')
    parser.add_argument('--save_every_epoch', type=int, default=1)
    
    opt = parser.parse_args()

    holicity_pc_dataset = HoliCityPointCloudDataset(opt.dataset_folder)
    is_train = opt.dataset_mode == "train"
    data_loader = DataLoader(holicity_pc_dataset, batch_size=1, shuffle=is_train)

    scale = opt.point_scale
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    func = lambda x: torch.cat([x, x[:1]], dim=0)

    in_channels = 3 * (opt.self_condition + 1) + (opt.masked_generation > 0) * 4
    SelectedNetwork = MinkUNet
    if opt.field_network:
        SelectedNetwork = MinkFieldUNet
    
    net = SelectedNetwork(in_channels, 3, time_channels=32, attns=opt.net_attention).to(device)
    diffusion_model = GaussianDiffusion(
        net, sampling_timesteps=opt.sampling_steps, self_condition=opt.self_condition
    ).to(device)

    if opt.use_ema:
        from ema_pytorch import EMA
        ema_net = SelectedNetwork(in_channels, 3, time_channels=32, attns=opt.net_attention).to(device)
        ema_diffusion_model = GaussianDiffusion(
            ema_net, sampling_timesteps=opt.sampling_steps, self_condition=opt.self_condition
        ).to(device)
        ema_model = EMA(
            model=diffusion_model,
            ema_model=ema_diffusion_model,
            beta=0.9999,              # exponential moving average factor
            update_after_step=2000,      # only after this number of .update() calls will it start updating
            update_every=20,          # how often to actually update, to save on compute (updates every 10th .update() call)
        )

    folder = opt.work_folder
    start_epoch = 0
    trainable_params = diffusion_model.parameters()
    if is_train and opt.ckpt is not None:
        model_dict = torch.load(f"{folder}/{opt.ckpt}")
        if opt.ckpt_not_strict:
            trainable_params = get_param_generator(diffusion_model, model_dict["model"])
            update_model_dict(diffusion_model, model_dict["model"])
            update_model_dict(ema_model, model_dict["ema_model"])
        diffusion_model.load_state_dict(model_dict["model"], strict=not opt.ckpt_not_strict)
        if opt.use_ema:
            ema_model.load_state_dict(model_dict["ema_model"], strict=not opt.ckpt_not_strict)
        if "epoch" in model_dict:
            start_epoch = model_dict["epoch"]

    optimizer = torch.optim.Adam(trainable_params, lr=opt.lr, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.2, total_iters=opt.num_epoch)

    import os
    os.system(f"mkdir -p {folder}/datavis")

    logger_file = f"log_{opt.dataset_mode}.txt"
    num_epochs = opt.num_epoch if is_train else 1

    spatial_weight, mask_info = None, None

    with open(f"{folder}/{logger_file}", "a") as f:
        for epoch in range(0, num_epochs):

            if is_train and epoch <= start_epoch:
                optimizer.zero_grad(set_to_none=True)
                optimizer.step()
                scheduler.step()
                continue

            data_idx = 0
            for pts, feats, dists, sem, geo_nonground in data_loader:

                pts = pts.to(device)[0].float()
                feats = feats.to(device)[0].float() / 255.0
                dists = dists.to(device)[0].float()
                sem = sem.to(device)[0].long()
                sem = torch.nn.functional.one_hot(sem, num_classes=19).float()
                geo_nonground = geo_nonground.to(device)[0].float()

                if opt.random_x_flip and np.random.rand() < 0.5:
                    pts[:, 0] *= -1

                if opt.random_y_flip and np.random.rand() < 0.5:
                    pts[:, 1] *= -1
                
                if opt.random_z_rotate:
                    rand_theta = np.random.rand() * 2 * np.pi
                    rand_rotmat = torch.Tensor([
                        [np.cos(rand_theta), -np.sin(rand_theta), 0],
                        [np.sin(rand_theta), np.cos(rand_theta), 0],
                        [0, 0, 1],
                    ]).float().to(pts.device)
                    pts = pts @ rand_rotmat.transpose(0, 1)
                
                if opt.random_gamma_correction:
                    assert(feats.min() >= 0)
                    assert(feats.max() <= 1)
                    max_gamma = 1.25
                    gamma = np.random.rand() * (max_gamma - 1) + 1
                    if np.random.rand() < 0.5:
                        gamma = 1.0 / gamma
                    feats = feats ** gamma

                coords = func(torch.cat([pts[:, :1] * 0, pts * scale], dim=-1))
                feats = torch.cat([
                    feats * 2 - 1, # in a range of -1 to 1
                    sem,
                    geo_nonground[..., None],
                    dists[..., None],
                ], dim=-1)

                if opt.masked_generation > 0:
                    assert(opt.random_z_rotate)
                    pano_coord = xyz2coord(pts)
                    # pano_coord[:, :1] from -2 to 2
                    known_th = -2.0 + 4.0 * opt.masked_generation
                    mask_info = (pano_coord[:, :1] > known_th).float() # right half to be generated
                    feats = torch.cat([feats, mask_info], dim=-1)

                feats = func(feats)

                pts_field = ME.TensorField(
                    features=feats,
                    coordinates=coords,
                    quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                    minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
                    device=pts.device,
                )

                if opt.field_network:
                    pts_sparse = pts_field
                else:
                    pts_sparse = pts_field.sparse()

                spatial_weight = torch.exp(-pts_sparse.F[:, -1:])
                spatial_sem = torch.argmax(pts_sparse.F[:, 3: 22], dim=-1)
                spatial_geo_nonground = pts_sparse.F[:, 22] > 0.5
                spatial_geo_ground = ~spatial_geo_nonground

                # [
                #     '0 road', '1 sidewalk', '2 building', '3 wall', 'fence', 'pole',
                #     'traffic light', 'traffic sign', 'vegetation', '9 terrain', 'sky',
                #     'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
                #     'bicycle'
                # ]

                to_preserve = (spatial_geo_ground & (spatial_sem == 0)) | \
                    (spatial_geo_ground & (spatial_sem == 1)) | \
                    (spatial_geo_ground & (spatial_sem == 9)) | \
                    (spatial_geo_nonground & (spatial_sem == 2)) | \
                    (spatial_geo_nonground & (spatial_sem == 3))
                to_remove = ~to_preserve
                spatial_weight[to_remove] = 0

                if opt.masked_generation:
                    mask_info = pts_sparse.F[:, -1:]
                
                pts_sparse = set_feature(pts_sparse, pts_sparse.F[:, :3])

                if opt.save_gt:
                    # pts_vis = pts_sparse
                    # pool = ME.MinkowskiAvgPooling(kernel_size=2, stride=2, dimension=3)
                    # for k in range(4):
                    #     with open(f"{folder}/datavis/gt_data_{data_idx}_{k}.txt", "w") as f:
                    #         for (_, x, y, z), (r, g, b) in zip(
                    #             pts_vis.C.cpu().numpy(),
                    #             (torch.clamp(pts_vis.F, -1, 1) * 127.5 + 127.5).cpu().numpy().astype(np.uint8),
                    #         ):
                    #             f.write('%.3lf %.3lf %.3lf %d %d %d\n' % (x, y, z, r, g, b))
                    #     pts_vis = pool(pts_vis)
                    spatial_weight_colored = plt.get_cmap("viridis")(
                        torch.clamp(spatial_weight[..., 0], 0.0, 1.0).cpu().numpy()
                    )
                    spatial_weight_colored = torch.from_numpy(spatial_weight_colored).cuda()
                    spatial_weight_colored = set_feature(pts_sparse, spatial_weight_colored)
                    with open(f"{folder}/datavis/data_weight_{data_idx}.txt", "w") as f:
                        for (_, x, y, z), (r, g, b, _) in zip(
                            spatial_weight_colored.C.cpu().numpy(),
                            (spatial_weight_colored.F * 255.0).cpu().numpy().astype(np.uint8),
                        ):
                            f.write('%.3lf %.3lf %.3lf %d %d %d\n' % (x, y, z, r, g, b))
                    with open(f"{folder}/datavis/data_{data_idx}.txt", "w") as f:
                        for (_, x, y, z), (r, g, b) in zip(
                            pts_sparse.C.cpu().numpy(),
                            (torch.clamp(pts_sparse.F, -1, 1) * 127.5 + 127.5).cpu().numpy().astype(np.uint8),
                        ):
                            f.write('%.3lf %.3lf %.3lf %d %d %d\n' % (x, y, z, r, g, b))
                    data_idx += 1
                    continue

                if not is_train:
                    accum = RasterizePointsXYsBlending()
                    with torch.no_grad():
                        # for fix_t in range(100, 1000, 100):
                        #     loss = diffusion_model(pts_sparse, fix_t=fix_t)
                        #     print(epoch, fix_t, loss.item())
                        #     f.write(f"{epoch}\t{fix_t}\t{loss}\n")
                        #     f.flush()
                        for ckpt in list(range(*eval(opt.ckpt))):
                            model_dict = torch.load(f"{folder}/epoch{ckpt}.pt")
                            diffusion_model.load_state_dict(model_dict["model"])
                            if opt.use_ema:
                                ema_model.load_state_dict(model_dict["ema_model"])
                            
                            if opt.field_network:
                                gt_field = pts_sparse
                            else:
                                gt_field = pts_sparse.slice(pts_field)

                            gt_im = accum(gt_field)
                            gt_im = (torch.clamp(gt_im, -1, 1) * 127.5 + 127.5).cpu().numpy().astype(np.uint8)[0].transpose([1, 2, 0])
                            Image.fromarray(gt_im).save(f"{folder}/datavis/data_ep{ckpt}_{data_idx}_gt.png")

                            with open(f"{folder}/datavis/data_ep{ckpt}_{data_idx}_gt.txt", "w") as f:
                                for (_, x, y, z), (r, g, b) in zip(
                                    pts_field.C.cpu().numpy() / float(scale),
                                    (torch.clamp(pts_field.F[:, :3], -1, 1) * 127.5 + 127.5).cpu().numpy().astype(np.uint8),
                                ):
                                    f.write('%.3lf %.3lf %.3lf %d %d %d\n' % (x, y, z, r, g, b))

                            for sp_time in range(opt.num_sample):
                                pts_sparse = set_feature(pts_sparse, pts_sparse.F, same_manager=False)
                                pred = ema_model.ema_model.sample(pts_sparse, mask_info)

                                if mask_info is not None:
                                    comb_ft = mask_info * pred.F + (1 - mask_info) * pts_sparse.F
                                    pred = set_feature(pred, comb_ft)
                
                                if opt.field_network:
                                    pred_field = pred
                                else:
                                    pred_field = pred.slice(pts_field)
                                
                                pred_im = accum(pred_field)
                                with open(f"{folder}/datavis/data_ep{ckpt}_{data_idx}_sp{sp_time}.txt", "w") as f:
                                    for (_, x, y, z), (r, g, b) in zip(
                                        pred_field.C.cpu().numpy() / float(scale),
                                        (torch.clamp(pred_field.F, -1, 1) * 127.5 + 127.5).cpu().numpy().astype(np.uint8),
                                    ):
                                        f.write('%.3lf %.3lf %.3lf %d %d %d\n' % (x, y, z, r, g, b))
                                # pad_batch = lambda ppp: torch.cat([ppp[:, :1] * 0, ppp], dim=-1)
                                # pts_org_field = ME.TensorField(
                                #     features=torch.from_numpy(holicity_pc_dataset.org_color).to(device).float(),
                                #     coordinates=pad_batch(torch.from_numpy(holicity_pc_dataset.org_coord).to(device).float() * scale),
                                #     quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                                #     minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
                                #     device=device,
                                #     coordinate_manager=pts_field.coordinate_manager,
                                # )
                                # pred_field = pred.features_at_coordinates(pad_batch(torch.from_numpy(holicity_pc_dataset.org_coord).to(device).float() * scale))
                                pred_im = (torch.clamp(pred_im, -1, 1) * 127.5 + 127.5).cpu().numpy().astype(np.uint8)[0].transpose([1, 2, 0])
                                Image.fromarray(pred_im).save(f"{folder}/datavis/data_ep{ckpt}_{data_idx}_sp{sp_time}.png")

                                
                    data_idx += 1
                    continue

                optimizer.zero_grad(set_to_none=True)
                loss = diffusion_model(pts_sparse, spatial_weight=spatial_weight, mask_info=mask_info)
                print(epoch, loss.item())
                f.write(f"{epoch}\t{loss}\n")
                f.flush()
                loss.backward()
                optimizer.step()
                ema_model.update()
            
            if is_train:
                scheduler.step()

            if epoch % opt.save_every_epoch == 0:
                state_dict = {
                    "epoch": epoch,
                    "model": diffusion_model.state_dict(),
                    "ema_model": ema_model.state_dict(),
                }
                torch.save(state_dict, f"{folder}/epoch{epoch}.pt")

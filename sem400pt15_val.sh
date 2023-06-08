CUDA_VISIBLE_DEVICES=0 python3 denoising_diffusion_pytorch/denoising_diffusion_minkowski_resample_sem.py \
    --dataset_folder "/cluster/project/cvg/zuoyue/holicity_point_cloud/4096x2048_resample_400_index_0_220/val" \
    --net_attention "N N N N N N N N N N" \
    --dataset_mode val \
    --work_folder resample200augema_highres_sem400pt15 \
    --sampling_steps 1000 \
    --use_ema \
    --point_scale 15 \
    --ckpt "(135,136,1)"

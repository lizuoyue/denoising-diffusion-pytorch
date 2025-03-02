CUDA_VISIBLE_DEVICES=0 python3 denoising_diffusion_pytorch/denoising_diffusion_minkowski_resample_sem.py \
    --dataset_folder "/cluster/project/cvg/zuoyue/holicity_point_cloud/4096x2048_resample_400_index_0_220/train" \
    --net_attention "N N N N N N N N N N" \
    --dataset_mode train \
    --work_folder resample200augema_highres_sem400pt10field_selfcond \
    --sampling_steps 1000 \
    --random_x_flip \
    --random_y_flip \
    --random_z_rotate \
    --use_ema \
    --num_epoch 200 \
    --point_scale 10 \
    --self_condition \
    --field_network \
    --ckpt epoch_start.pt \
    --ckpt_not_strict

# CUDA_VISIBLE_DEVICES=0 python train.py --model_name ab2_cuda0/c3vd_monodepth2_b8e30_me --data_path rect_c3vd_data \
#   --dataset c3vd --split c3vd --batch_size 8 --height 256 --width 320 --png --use_cudnn \
#   --num_epochs 30 --scales 0 1 2 3 --num_workers=4 --scheduler_step_size 25 --half_epoch_matcher --matcher_loss_alpha 0.20


CUDA_VISIBLE_DEVICES=0 python train_v2.py --model_name ab2_cuda0/c3vd_litemono_b8e35_me --data_path rect_c3vd_data \
  --dataset c3vd --split c3vd --batch_size 8 --height 256 --width 320 --png  --use_cudnn \
  --num_epochs 35 --scales 0 1 2 --num_workers=4 --lr 0.0001 5e-6 36 0.0001 1e-5 36 --half_epoch_matcher --matcher_loss_alpha 0.20 
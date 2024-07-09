## BASELINE


# c3vd # monodepth2
CUDA_VISIBLE_DEVICES=0 python train.py --model_name c3vd_monodepth2 --data_path rect_c3vd_data \
  --dataset c3vd --split c3vd --batch_size 16 --height 256 --width 320 --png --disable_matcher --use_cudnn \
  --num_epochs 20 --scales 0 1 2 3 --num_workers=4

CUDA_VISIBLE_DEVICES=0 python train.py --model_name c3vd_monodepth2_epoch50 --data_path rect_c3vd_data \
  --dataset c3vd --split c3vd --batch_size 16 --height 256 --width 320 --png --disable_matcher --use_cudnn \
  --num_epochs 50 --scales 0 1 2 3 --num_workers=4 --scheduler_step_size 36 

CUDA_VISIBLE_DEVICES=0 python train.py --model_name c3vd_monodepth2_b12e20 --data_path rect_c3vd_data \
  --dataset c3vd --split c3vd --batch_size 12 --height 256 --width 320 --png --disable_matcher --use_cudnn \
  --num_epochs 20 --scales 0 1 2 3 --num_workers=4

CUDA_VISIBLE_DEVICES=0 python train.py --model_name c3vd_monodepth2_b8e20 --data_path rect_c3vd_data \
  --dataset c3vd --split c3vd --batch_size 8 --height 256 --width 320 --png --disable_matcher --use_cudnn \
  --num_epochs 20 --scales 0 1 2 3 --num_workers=4

CUDA_VISIBLE_DEVICES=1 python train.py --model_name c3vd_monodepth2_b12e50 --data_path rect_c3vd_data \
  --dataset c3vd --split c3vd --batch_size 12 --height 256 --width 320 --png --disable_matcher --use_cudnn \
  --num_epochs 50 --scales 0 1 2 3 --num_workers=4 --scheduler_step_size 36

CUDA_VISIBLE_DEVICES=1 python train.py --model_name c3vd_monodepth2_b8e50 --data_path rect_c3vd_data \
  --dataset c3vd --split c3vd --batch_size 16 --height 256 --width 320 --png --disable_matcher --use_cudnn \
  --num_epochs 50 --scales 0 1 2 3 --num_workers=4 --scheduler_step_size 36


# c3vd # lite-mono
CUDA_VISIBLE_DEVICES=0 python train_v2.py --model_name c3vd_litemono --data_path rect_c3vd_data \
  --dataset c3vd --split c3vd --batch_size 16 --height 256 --width 320 --png --disable_matcher --use_cudnn \
    --num_epochs 50 --scales 0 1 2 --num_workers=4

CUDA_VISIBLE_DEVICES=0 python train_v2.py --model_name c3vd_litemono --data_path rect_c3vd_data \
  --dataset c3vd --split c3vd --batch_size 16 --height 256 --width 320 --png --disable_matcher --use_cudnn \
    --num_epochs 50 --scales 0 1 2 --num_workers=4

CUDA_VISIBLE_DEVICES=0 python train_v2.py --model_name c3vd_litemono_b12e50 --data_path rect_c3vd_data \
  --dataset c3vd --split c3vd --batch_size 12 --height 256 --width 320 --png --disable_matcher --use_cudnn \
    --num_epochs 50 --scales 0 1 2 --num_workers=4

CUDA_VISIBLE_DEVICES=1 python train_v2.py --model_name c3vd_litemono_b8e50 --data_path rect_c3vd_data \
  --dataset c3vd --split c3vd --batch_size 8 --height 256 --width 320 --png --disable_matcher --use_cudnn \
    --num_epochs 50 --scales 0 1 2 --num_workers=4

## Matching Lossw


# c3vd # monodepth2 # Matching Loss, early stop
CUDA_VISIBLE_DEVICES=0 python train.py --model_name c3vd_monodepth2_M_E --data_path rect_c3vd_data \
  --dataset c3vd --split c3vd --batch_size 12 --height 256 --width 320 --png --use_cudnn \
  --num_epochs 20 --scales 0 1 2 3 --num_workers=4 --half_epoch_matcher --matcher_loss_alpha 0.20 

CUDA_VISIBLE_DEVICES=1 python train.py --model_name c3vd_monodepth2_M_E_b8e50 --data_path rect_c3vd_data \
  --dataset c3vd --split c3vd --batch_size 8 --height 256 --width 320 --png  --use_cudnn \
  --num_epochs 50 --scales 0 1 2 3 --num_workers=4 --scheduler_step_size 36 --half_epoch_matcher --matcher_loss_alpha 0.20 


# 发现epoch太大作用不大以后，调小epoch

CUDA_VISIBLE_DEVICES=0 python train.py --model_name c3vd_monodepth2_b16e25 --data_path rect_c3vd_data \
  --dataset c3vd --split c3vd --batch_size 16 --height 256 --width 320 --png --disable_matcher --use_cudnn \
  --num_epochs 25 --scales 0 1 2 3 --num_workers=4 --scheduler_step_size 20

CUDA_VISIBLE_DEVICES=0 python train.py --model_name c3vd_monodepth2_b12e25 --data_path rect_c3vd_data \
  --dataset c3vd --split c3vd --batch_size 12 --height 256 --width 320 --png --disable_matcher --use_cudnn \
  --num_epochs 25 --scales 0 1 2 3 --num_workers=4 --scheduler_step_size 20

CUDA_VISIBLE_DEVICES=0 python train.py --model_name c3vd_monodepth2_b8e25 --data_path rect_c3vd_data \
  --dataset c3vd --split c3vd --batch_size 8 --height 256 --width 320 --png --disable_matcher --use_cudnn \
  --num_epochs 25 --scales 0 1 2 3 --num_workers=4 --scheduler_step_size 20

CUDA_VISIBLE_DEVICES=0 python train.py --model_name c3vd_monodepth2_b16e30 --data_path rect_c3vd_data \
  --dataset c3vd --split c3vd --batch_size 16 --height 256 --width 320 --png --disable_matcher --use_cudnn \
  --num_epochs 30 --scales 0 1 2 3 --num_workers=4 --scheduler_step_size 25

CUDA_VISIBLE_DEVICES=0 python train.py --model_name c3vd_monodepth2_b12e30 --data_path rect_c3vd_data \
  --dataset c3vd --split c3vd --batch_size 12 --height 256 --width 320 --png --disable_matcher --use_cudnn \
  --num_epochs 30 --scales 0 1 2 3 --num_workers=4 --scheduler_step_size 25

CUDA_VISIBLE_DEVICES=0 python train.py --model_name c3vd_monodepth2_b8e30 --data_path rect_c3vd_data \
  --dataset c3vd --split c3vd --batch_size 8 --height 256 --width 320 --png --disable_matcher --use_cudnn \
  --num_epochs 30 --scales 0 1 2 3 --num_workers=4 --scheduler_step_size 25
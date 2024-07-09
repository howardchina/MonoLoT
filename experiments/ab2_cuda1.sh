
CUDA_VISIBLE_DEVICES=1 python train.py --model_name ab2_cuda1/c3vd_monodepth2_b16e20 --data_path rect_c3vd_data \
  --dataset c3vd --split c3vd --batch_size 16 --height 256 --width 320 --png --disable_matcher --use_cudnn \
  --num_epochs 20 --scales 0 1 2 3 --num_workers=4 --scheduler_step_size 15

CUDA_VISIBLE_DEVICES=1 python train.py --model_name ab2_cuda1/c3vd_monodepth2_b12e20 --data_path rect_c3vd_data \
  --dataset c3vd --split c3vd --batch_size 12 --height 256 --width 320 --png --disable_matcher --use_cudnn \
  --num_epochs 20 --scales 0 1 2 3 --num_workers=4 --scheduler_step_size 15

CUDA_VISIBLE_DEVICES=1 python train.py --model_name ab2_cuda1/c3vd_monodepth2_b8e20 --data_path rect_c3vd_data \
  --dataset c3vd --split c3vd --batch_size 8 --height 256 --width 320 --png --disable_matcher --use_cudnn \
  --num_epochs 20 --scales 0 1 2 3 --num_workers=4 --scheduler_step_size 15
  
CUDA_VISIBLE_DEVICES=1 python train.py --model_name ab2_cuda1/c3vd_monodepth2_b16e25 --data_path rect_c3vd_data \
  --dataset c3vd --split c3vd --batch_size 16 --height 256 --width 320 --png --disable_matcher --use_cudnn \
  --num_epochs 25 --scales 0 1 2 3 --num_workers=4 --scheduler_step_size 20

CUDA_VISIBLE_DEVICES=1 python train.py --model_name ab2_cuda1/c3vd_monodepth2_b12e25 --data_path rect_c3vd_data \
  --dataset c3vd --split c3vd --batch_size 12 --height 256 --width 320 --png --disable_matcher --use_cudnn \
  --num_epochs 25 --scales 0 1 2 3 --num_workers=4 --scheduler_step_size 20

CUDA_VISIBLE_DEVICES=1 python train.py --model_name ab2_cuda1/c3vd_monodepth2_b8e25 --data_path rect_c3vd_data \
  --dataset c3vd --split c3vd --batch_size 8 --height 256 --width 320 --png --disable_matcher --use_cudnn \
  --num_epochs 25 --scales 0 1 2 3 --num_workers=4 --scheduler_step_size 20

CUDA_VISIBLE_DEVICES=1 python train.py --model_name ab2_cuda1/c3vd_monodepth2_b16e30 --data_path rect_c3vd_data \
  --dataset c3vd --split c3vd --batch_size 16 --height 256 --width 320 --png --disable_matcher --use_cudnn \
  --num_epochs 30 --scales 0 1 2 3 --num_workers=4 --scheduler_step_size 25

CUDA_VISIBLE_DEVICES=1 python train.py --model_name ab2_cuda1/c3vd_monodepth2_b12e30 --data_path rect_c3vd_data \
  --dataset c3vd --split c3vd --batch_size 12 --height 256 --width 320 --png --disable_matcher --use_cudnn \
  --num_epochs 30 --scales 0 1 2 3 --num_workers=4 --scheduler_step_size 25

CUDA_VISIBLE_DEVICES=1 python train.py --model_name ab2_cuda1/c3vd_monodepth2_b8e30 --data_path rect_c3vd_data \
  --dataset c3vd --split c3vd --batch_size 8 --height 256 --width 320 --png --disable_matcher --use_cudnn \
  --num_epochs 30 --scales 0 1 2 3 --num_workers=4 --scheduler_step_size 25
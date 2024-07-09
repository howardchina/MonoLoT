
CUDA_VISIBLE_DEVICES=1 python train_v2.py --model_name ab2_cuda1/c3vd_litemono_b16e30 --data_path rect_c3vd_data \
  --dataset c3vd --split c3vd --batch_size 16 --height 256 --width 320 --png --disable_matcher --use_cudnn \
  --num_epochs 30 --scales 0 1 2 --num_workers=4 --lr 0.0001 5e-6 31 0.0001 1e-5 31

CUDA_VISIBLE_DEVICES=1 python train_v2.py --model_name ab2_cuda1/c3vd_litemono_b12e30 --data_path rect_c3vd_data \
  --dataset c3vd --split c3vd --batch_size 12 --height 256 --width 320 --png --disable_matcher --use_cudnn \
  --num_epochs 30 --scales 0 1 2 --num_workers=4 --lr 0.0001 5e-6 31 0.0001 1e-5 31

CUDA_VISIBLE_DEVICES=1 python train_v2.py --model_name ab2_cuda1/c3vd_litemono_b8e30 --data_path rect_c3vd_data \
  --dataset c3vd --split c3vd --batch_size 8 --height 256 --width 320 --png --disable_matcher --use_cudnn \
  --num_epochs 30 --scales 0 1 2 --num_workers=4 --lr 0.0001 5e-6 31 0.0001 1e-5 31
  
CUDA_VISIBLE_DEVICES=1 python train_v2.py --model_name ab2_cuda1/c3vd_litemono_b16e35 --data_path rect_c3vd_data \
  --dataset c3vd --split c3vd --batch_size 16 --height 256 --width 320 --png --disable_matcher --use_cudnn \
  --num_epochs 35 --scales 0 1 2 --num_workers=4 --lr 0.0001 5e-6 36 0.0001 1e-5 36

CUDA_VISIBLE_DEVICES=1 python train_v2.py --model_name ab2_cuda1/c3vd_litemono_b12e35 --data_path rect_c3vd_data \
  --dataset c3vd --split c3vd --batch_size 12 --height 256 --width 320 --png --disable_matcher --use_cudnn \
  --num_epochs 35 --scales 0 1 2 --num_workers=4 --lr 0.0001 5e-6 36 0.0001 1e-5 36

CUDA_VISIBLE_DEVICES=1 python train_v2.py --model_name ab2_cuda1/c3vd_litemono_b8e35 --data_path rect_c3vd_data \
  --dataset c3vd --split c3vd --batch_size 8 --height 256 --width 320 --png --disable_matcher --use_cudnn \
  --num_epochs 35 --scales 0 1 2 --num_workers=4 --lr 0.0001 5e-6 36 0.0001 1e-5 36

CUDA_VISIBLE_DEVICES=1 python train_v2.py --model_name ab2_cuda1/c3vd_litemono_b16e40 --data_path rect_c3vd_data \
  --dataset c3vd --split c3vd --batch_size 16 --height 256 --width 320 --png --disable_matcher --use_cudnn \
  --num_epochs 40 --scales 0 1 2 --num_workers=4 --lr 0.0001 5e-6 36 0.0001 1e-5 41

CUDA_VISIBLE_DEVICES=1 python train_v2.py --model_name ab2_cuda1/c3vd_litemono_b12e40 --data_path rect_c3vd_data \
  --dataset c3vd --split c3vd --batch_size 12 --height 256 --width 320 --png --disable_matcher --use_cudnn \
  --num_epochs 40 --scales 0 1 2 --num_workers=4 --lr 0.0001 5e-6 36 0.0001 1e-5 41

CUDA_VISIBLE_DEVICES=1 python train_v2.py --model_name ab2_cuda1/c3vd_litemono_b8e40 --data_path rect_c3vd_data \
  --dataset c3vd --split c3vd --batch_size 8 --height 256 --width 320 --png --disable_matcher --use_cudnn \
  --num_epochs 40 --scales 0 1 2 --num_workers=4 --lr 0.0001 5e-6 36 0.0001 1e-5 41
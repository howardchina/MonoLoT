
CUDA_VISIBLE_DEVICES=1 python train.py --model_name tab2_cuda1/simcol_monodepth2_b8e30 --data_path simcol_data \
  --dataset simcol --split simcol --batch_size 8 --height 448 --width 448 --png --disable_matcher --use_cudnn \
  --num_epochs 30 --scales 0 1 2 3 --num_workers=4 --scheduler_step_size 25


CUDA_VISIBLE_DEVICES=1 python train.py --model_name tab2_cuda1/simcol_monodepth2_b8e30_me --data_path simcol_data \
  --dataset simcol --split simcol --batch_size 8 --height 448 --width 448 --png --use_cudnn \
  --num_epochs 30 --scales 0 1 2 3 --num_workers=4 --scheduler_step_size 25 --half_epoch_matcher --matcher_loss_alpha 0.20 

CUDA_VISIBLE_DEVICES=1 python train_v2.py --model_name tab2_cuda1/simcol_litemono_b8e35 --data_path simcol_data \
  --dataset simcol --split simcol --batch_size 8 --height 448 --width 448 --png --disable_matcher --use_cudnn \
  --num_epochs 35 --scales 0 1 2 --num_workers=4 --lr 0.0001 5e-6 36 0.0001 1e-5 36

CUDA_VISIBLE_DEVICES=1 python train_v2.py --model_name tab2_cuda1/simcol_litemono_b8e35_me --data_path simcol_data \
  --dataset simcol --split simcol --batch_size 8 --height 448 --width 448 --png  --use_cudnn \
  --num_epochs 35 --scales 0 1 2 --num_workers=4 --lr 0.0001 5e-6 36 0.0001 1e-5 36 --half_epoch_matcher --matcher_loss_alpha 0.20 
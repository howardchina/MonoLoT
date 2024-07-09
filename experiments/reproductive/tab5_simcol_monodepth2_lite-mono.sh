CUDA_VISIBLE_DEVICES=1 python train_corres.py --model monodepth2 --model_name reproductive/tab4_simcol/simcol_monodepth2_b12e40_me --data_path simcol_data \
  --dataset simcolcorres --split simcol --batch_size 12 --height 448 --width 448 --png \
  --num_epochs 40 --scales 0 1 2 3 --num_workers=4 --lr 0.0001 1e-5 41 0.0001 1e-5 41 \
  --matcher_result "playground/heqi/Simcol/matcher_result.npy" \
  --half_epoch_matcher --matcher_loss_alpha 0.20

CUDA_VISIBLE_DEVICES=1 python train_corres.py --model lite-mono --model_name reproductive/tab4_simcol/simcol_lite-mono_b16e40_me --data_path simcol_data \
  --dataset simcolcorres --split simcol --batch_size 16 --height 448 --width 448 --png \
  --num_epochs 40 --scales 0 1 2 --num_workers=4 --lr 0.0001 1e-5 41 0.0001 1e-5 41 \
  --matcher_result "playground/heqi/Simcol/matcher_result.npy" \
  --half_epoch_matcher --matcher_loss_alpha 0.20 \
  --mypretrain models/lite-mono-pretrain.pth

CUDA_VISIBLE_DEVICES=1 python train_corres.py --model monodepth2 --model_name reproductive/tab4_simcol/simcol_monodepth2_b12e40_raw --data_path simcol_data \
  --dataset simcolcorres --split simcol --batch_size 12 --height 448 --width 448 --png \
  --num_epochs 40 --scales 0 1 2 3 --num_workers=4 --lr 0.0001 1e-5 41 0.0001 1e-5 41 \
  --matcher_result "playground/heqi/Simcol/matcher_result.npy" \
  --disable_matcher

CUDA_VISIBLE_DEVICES=1 python train_corres.py --model lite-mono --model_name reproductive/tab4_simcol/simcol_lite-mono_b16e40_raw --data_path simcol_data \
  --dataset simcolcorres --split simcol --batch_size 16 --height 448 --width 448 --png \
  --num_epochs 40 --scales 0 1 2 --num_workers=4 --lr 0.0001 1e-5 41 0.0001 1e-5 41 \
  --matcher_result "playground/heqi/Simcol/matcher_result.npy" \
  --disable_matcher \
  --mypretrain models/lite-mono-pretrain.pth 
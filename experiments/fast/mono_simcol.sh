CUDA_VISIBLE_DEVICES=0 python train_corres.py --model_name fast/tab2_cuda0/simcol_monodepth2_b8e30_me_h_mla02 --data_path simcol_data \
  --dataset simcolcorres --split simcol --batch_size 8 --height 448 --width 448 --png  --matcher_result playground/heqi/Simcol/matcher_result.npy \
  --num_epochs 30 --scales 0 1 2 3 --num_workers=4 --scheduler_step_size 25 --half_epoch_matcher --matcher_loss_alpha 0.20

CUDA_VISIBLE_DEVICES=0 python train_corres.py --model_name fast/tab2_cuda0/simcol_monodepth2_b8e30 --data_path simcol_data \
  --dataset simcolcorres --split simcol --batch_size 8 --height 448 --width 448 --png  --matcher_result playground/heqi/Simcol/matcher_result.npy --disable_matcher \
  --num_epochs 30 --scales 0 1 2 3 --num_workers=4 --scheduler_step_size 25 --half_epoch_matcher --matcher_loss_alpha 0.20


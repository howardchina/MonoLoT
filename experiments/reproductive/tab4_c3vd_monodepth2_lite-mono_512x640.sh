# CUDA_VISIBLE_DEVICES=0 python train_corres.py --model monodepth2 --model_name reproductive/tab4_c3vd/c3vd_monodepth2_b12e40_me_HD --data_path rect_c3vd_data \
#   --dataset c3vdcorres --split c3vd --batch_size 12 --height 512 --width 640 --png \
#   --num_epochs 40 --scales 0 1 2 3 --num_workers=4 --lr 0.0001 1e-5 41 0.0001 1e-5 41 \
#   --matcher_result "playground/heqi/C3VD/matcher_result_512x640.npy" \
#   --half_epoch_matcher --matcher_loss_alpha 0.20 \
#   --confidence 0.9

CUDA_VISIBLE_DEVICES=0 python train_corres.py --model monodepth2 --model_name reproductive/tab4_c3vd/c3vd_monodepth2_b8e40_me_HD --data_path rect_c3vd_data \
  --dataset c3vdcorres --split c3vd --batch_size 8 --height 512 --width 640 --png \
  --num_epochs 40 --scales 0 1 2 3 --num_workers=8 --lr 0.0001 1e-5 41 0.0001 1e-5 41 \
  --matcher_result "playground/heqi/C3VD/matcher_result_512x640.npy" \
  --half_epoch_matcher --matcher_loss_alpha 0.20 \
  --confidence 0.9

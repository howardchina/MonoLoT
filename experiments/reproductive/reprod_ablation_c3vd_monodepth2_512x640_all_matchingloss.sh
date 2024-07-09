# CUDA_VISIBLE_DEVICES=0 python train_corres.py --model monodepth2 --model_name reproductive/ablation_c3vd_512x640/c3vd_monodepth2_b8e30_me --data_path rect_c3vd_data \
#   --dataset c3vdcorres --split c3vd --batch_size 8 --height 512 --width 640 --png \
#   --num_epochs 30 --scales 0 1 2 3 --num_workers=6 --lr 0.0001 1e-5 31 0.0001 1e-5 31 \
#   --matcher_result "playground/heqi/C3VD/matcher_result_512x640.npy" \
#   --half_epoch_matcher --matcher_loss_alpha 0.20 \
#   --confidence 0.9

# CUDA_VISIBLE_DEVICES=0 python train_corres.py --model monodepth2 --model_name reproductive/ablation_c3vd_512x640/c3vd_monodepth2_b8e35_me --data_path rect_c3vd_data \
#   --dataset c3vdcorres --split c3vd --batch_size 8 --height 512 --width 640 --png \
#   --num_epochs 35 --scales 0 1 2 3 --num_workers=6 --lr 0.0001 1e-5 36 0.0001 1e-5 36 \
#   --matcher_result "playground/heqi/C3VD/matcher_result_512x640.npy" \
#   --half_epoch_matcher --matcher_loss_alpha 0.20 \t
#   --confidence 0.9

# CUDA_VISIBLE_DEVICES=0 python train_corres.py --model monodepth2 --model_name reproductive/ablation_c3vd_512x640/c3vd_monodepth2_b8e40_me --data_path rect_c3vd_data \
#   --dataset c3vdcorres --split c3vd --batch_size 8 --height 512 --width 640 --png \
#   --num_epochs 40 --scales 0 1 2 3 --num_workers=6 --lr 0.0001 1e-5 41 0.0001 1e-5 41 \
#   --matcher_result "playground/heqi/C3VD/matcher_result_512x640.npy" \
#   --half_epoch_matcher --matcher_loss_alpha 0.20 \
#   --confidence 0.9

# CUDA_VISIBLE_DEVICES=0 python train_corres.py --model monodepth2 --model_name reproductive/ablation_c3vd_512x640/c3vd_monodepth2_b8e50_me --data_path rect_c3vd_data \
#   --dataset c3vdcorres --split c3vd --batch_size 8 --height 512 --width 640 --png \
#   --num_epochs 50 --scales 0 1 2 3 --num_workers=6 --lr 0.0001 1e-5 51 0.0001 1e-5 51 \
#   --matcher_result "playground/heqi/C3VD/matcher_result_512x640.npy" \
#   --half_epoch_matcher --matcher_loss_alpha 0.20 \
#   --confidence 0.9

CUDA_VISIBLE_DEVICES=1 python train_corres.py --model monodepth2 --model_name reproductive/ablation_c3vd_512x640/c3vd_monodepth2_b8e60_me --data_path rect_c3vd_data \
  --dataset c3vdcorres --split c3vd --batch_size 8 --height 512 --width 640 --png \
  --num_epochs 60 --scales 0 1 2 3 --num_workers=6 --lr 0.0001 1e-5 61 0.0001 1e-5 61 \
  --matcher_result "playground/heqi/C3VD/matcher_result_512x640.npy" \
  --half_epoch_matcher --matcher_loss_alpha 0.20 \
  --confidence 0.9

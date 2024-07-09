CUDA_VISIBLE_DEVICES=1 python train_corres.py --model monodepth2 --model_name reproductive/ablation_c3vd/resnet50/c3vd_monodepth2_b8e30_me_conf95 --data_path rect_c3vd_data \
  --dataset c3vdcorres --split c3vd --batch_size 8 --height 256 --width 320 --png \
  --num_epochs 30 --scales 0 1 2 3 --num_workers=4 --lr 0.0001 1e-5 31 0.0001 1e-5 31 \
  --num_layers 50 \
  --matcher_result "playground/heqi/C3VD/matcher_result.npy" \
  --half_epoch_matcher --matcher_loss_alpha 0.20 \
  --confidence 0.95

CUDA_VISIBLE_DEVICES=1 python train_corres.py --model monodepth2 --model_name reproductive/ablation_c3vd/resnet50/c3vd_monodepth2_b8e35_me_conf95 --data_path rect_c3vd_data \
  --dataset c3vdcorres --split c3vd --batch_size 8 --height 256 --width 320 --png \
  --num_epochs 35 --scales 0 1 2 3 --num_workers=4 --lr 0.0001 1e-5 36 0.0001 1e-5 36 \
  --num_layers 50 \
  --matcher_result "playground/heqi/C3VD/matcher_result.npy" \
  --half_epoch_matcher --matcher_loss_alpha 0.20 \
  --confidence 0.95

CUDA_VISIBLE_DEVICES=1 python train_corres.py --model monodepth2 --model_name reproductive/ablation_c3vd/resnet50/c3vd_monodepth2_b8e40_me_conf95 --data_path rect_c3vd_data \
  --dataset c3vdcorres --split c3vd --batch_size 8 --height 256 --width 320 --png \
  --num_epochs 40 --scales 0 1 2 3 --num_workers=4 --lr 0.0001 1e-5 41 0.0001 1e-5 41 \
  --num_layers 50 \
  --matcher_result "playground/heqi/C3VD/matcher_result.npy" \
  --half_epoch_matcher --matcher_loss_alpha 0.20 \
  --confidence 0.95
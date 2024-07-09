CUDA_VISIBLE_DEVICES=1 python train_corres.py --model monodepth2 --model_name reproductive/std_c3vd/resnet50/c3vd_monodepth2_b12e30_me_3407 --data_path rect_c3vd_data \
  --dataset c3vdcorres --split c3vd --batch_size 12 --height 256 --width 320 --png \
  --num_epochs 30 --scales 0 1 2 3 --num_workers=4 --lr 0.0001 1e-5 31 0.0001 1e-5 31 \
  --num_layers 50 \
  --matcher_result "playground/heqi/C3VD/new_matcher_result.npy" \
  --half_epoch_matcher --matcher_loss_alpha 0.20 \
  --confidence 0.9 --seed 3407

CUDA_VISIBLE_DEVICES=1 python train_corres.py --model monodepth2 --model_name reproductive/std_c3vd/resnet50/c3vd_monodepth2_b12e30_me_1243 --data_path rect_c3vd_data \
  --dataset c3vdcorres --split c3vd --batch_size 12 --height 256 --width 320 --png \
  --num_epochs 30 --scales 0 1 2 3 --num_workers=4 --lr 0.0001 1e-5 31 0.0001 1e-5 31 \
  --num_layers 50 \
  --matcher_result "playground/heqi/C3VD/new_matcher_result.npy" \
  --half_epoch_matcher --matcher_loss_alpha 0.20 \
  --confidence 0.9 --seed 1243

CUDA_VISIBLE_DEVICES=1 python train_corres.py --model monodepth2 --model_name reproductive/std_c3vd/resnet50/c3vd_monodepth2_b12e30_me_4312 --data_path rect_c3vd_data \
  --dataset c3vdcorres --split c3vd --batch_size 12 --height 256 --width 320 --png \
  --num_epochs 30 --scales 0 1 2 3 --num_workers=4 --lr 0.0001 1e-5 31 0.0001 1e-5 31 \
  --num_layers 50 \
  --matcher_result "playground/heqi/C3VD/new_matcher_result.npy" \
  --half_epoch_matcher --matcher_loss_alpha 0.20 \
  --confidence 0.9 --seed 4312


CUDA_VISIBLE_DEVICES=1 python train_corres.py --model monodepth2 --model_name reproductive/std_c3vd/resnet50/c3vd_monodepth2_b12e30_me_1024 --data_path rect_c3vd_data \
  --dataset c3vdcorres --split c3vd --batch_size 12 --height 256 --width 320 --png \
  --num_epochs 30 --scales 0 1 2 3 --num_workers=4 --lr 0.0001 1e-5 31 0.0001 1e-5 31 \
  --num_layers 50 \
  --matcher_result "playground/heqi/C3VD/new_matcher_result.npy" \
  --half_epoch_matcher --matcher_loss_alpha 0.20 \
  --confidence 0.9 --seed 1024
  
CUDA_VISIBLE_DEVICES=1 python train_corres.py --model monodepth2 --model_name reproductive/std_c3vd/resnet50/c3vd_monodepth2_b12e30_me_2048 --data_path rect_c3vd_data \
  --dataset c3vdcorres --split c3vd --batch_size 12 --height 256 --width 320 --png \
  --num_epochs 30 --scales 0 1 2 3 --num_workers=4 --lr 0.0001 1e-5 31 0.0001 1e-5 31 \
  --num_layers 50 \
  --matcher_result "playground/heqi/C3VD/new_matcher_result.npy" \
  --half_epoch_matcher --matcher_loss_alpha 0.20 \
  --confidence 0.9 --seed 2048
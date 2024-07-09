CUDA_VISIBLE_DEVICES=0 python train_corres.py --model monodepth2 --model_name reproductive/std_c3vd/resnet18/c3vd_monodepth2_b8e30_raw_3407 --data_path rect_c3vd_data \
  --dataset c3vdcorres --split c3vd --batch_size 8 --height 256 --width 320 --png \
  --num_epochs 30 --scales 0 1 2 3 --num_workers=4 --lr 0.0001 1e-5 31 0.0001 1e-5 31 \
  --num_layers 18 \
  --matcher_result "playground/heqi/C3VD/new_matcher_result.npy" \
  --disable_matcher  --seed 3407

CUDA_VISIBLE_DEVICES=0 python train_corres.py --model monodepth2 --model_name reproductive/std_c3vd/resnet18/c3vd_monodepth2_b8e30_raw_1243 --data_path rect_c3vd_data \
  --dataset c3vdcorres --split c3vd --batch_size 8 --height 256 --width 320 --png \
  --num_epochs 30 --scales 0 1 2 3 --num_workers=4 --lr 0.0001 1e-5 31 0.0001 1e-5 31 \
  --num_layers 18 \
  --matcher_result "playground/heqi/C3VD/new_matcher_result.npy" \
  --disable_matcher  --seed 1243

CUDA_VISIBLE_DEVICES=0 python train_corres.py --model monodepth2 --model_name reproductive/std_c3vd/resnet18/c3vd_monodepth2_b8e30_raw_4312 --data_path rect_c3vd_data \
  --dataset c3vdcorres --split c3vd --batch_size 8 --height 256 --width 320 --png \
  --num_epochs 30 --scales 0 1 2 3 --num_workers=4 --lr 0.0001 1e-5 31 0.0001 1e-5 31 \
  --num_layers 18 \
  --matcher_result "playground/heqi/C3VD/new_matcher_result.npy" \
  --disable_matcher  --seed 4312


CUDA_VISIBLE_DEVICES=0 python train_corres.py --model monodepth2 --model_name reproductive/std_c3vd/resnet18/c3vd_monodepth2_b8e30_raw_1024 --data_path rect_c3vd_data \
  --dataset c3vdcorres --split c3vd --batch_size 8 --height 256 --width 320 --png \
  --num_epochs 30 --scales 0 1 2 3 --num_workers=4 --lr 0.0001 1e-5 31 0.0001 1e-5 31 \
  --num_layers 18 \
  --matcher_result "playground/heqi/C3VD/new_matcher_result.npy" \
  --disable_matcher  --seed 1024
  
CUDA_VISIBLE_DEVICES=0 python train_corres.py --model monodepth2 --model_name reproductive/std_c3vd/resnet18/c3vd_monodepth2_b8e30_raw_2048 --data_path rect_c3vd_data \
  --dataset c3vdcorres --split c3vd --batch_size 8 --height 256 --width 320 --png \
  --num_epochs 30 --scales 0 1 2 3 --num_workers=4 --lr 0.0001 1e-5 31 0.0001 1e-5 31 \
  --num_layers 18 \
  --matcher_result "playground/heqi/C3VD/new_matcher_result.npy" \
  --disable_matcher  --seed 2048
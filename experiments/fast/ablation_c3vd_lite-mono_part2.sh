CUDA_VISIBLE_DEVICES=1 python train_corres.py --model lite-mono --model_name fast/ablation_c3vd/c3vd_lite-mono_b8e30_raw --data_path rect_c3vd_data \
  --dataset c3vdcorres --split c3vd --batch_size 8 --height 256 --width 320 --png \
  --num_epochs 30 --scales 0 1 2 --num_workers=4 --lr 0.0001 5e-6 31 0.0001 1e-5 31 \
  --matcher_result "playground/heqi/C3VD/matcher_result.npy" \
  --disable_matcher \
  --mypretrain models/lite-mono-pretrain.pth 

CUDA_VISIBLE_DEVICES=1 python train_corres.py --model lite-mono --model_name fast/ablation_c3vd/c3vd_lite-mono_b16e35_raw --data_path rect_c3vd_data \
  --dataset c3vdcorres --split c3vd --batch_size 16 --height 256 --width 320 --png \
  --num_epochs 35 --scales 0 1 2 --num_workers=4 --lr 0.0001 5e-6 36 0.0001 1e-5 36 \
  --matcher_result "playground/heqi/C3VD/matcher_result.npy" \
  --disable_matcher \
  --mypretrain models/lite-mono-pretrain.pth 

CUDA_VISIBLE_DEVICES=1 python train_corres.py --model lite-mono --model_name fast/ablation_c3vd/c3vd_lite-mono_b12e35_raw --data_path rect_c3vd_data \
  --dataset c3vdcorres --split c3vd --batch_size 12 --height 256 --width 320 --png \
  --num_epochs 35 --scales 0 1 2 --num_workers=4 --lr 0.0001 5e-6 36 0.0001 1e-5 36 \
  --matcher_result "playground/heqi/C3VD/matcher_result.npy" \
  --disable_matcher \
  --mypretrain models/lite-mono-pretrain.pth 

CUDA_VISIBLE_DEVICES=1 python train_corres.py --model lite-mono --model_name fast/ablation_c3vd/c3vd_lite-mono_b8e35_raw --data_path rect_c3vd_data \
  --dataset c3vdcorres --split c3vd --batch_size 8 --height 256 --width 320 --png \
  --num_epochs 35 --scales 0 1 2 --num_workers=4 --lr 0.0001 5e-6 36 0.0001 1e-5 36 \
  --matcher_result "playground/heqi/C3VD/matcher_result.npy" \
  --disable_matcher \
  --mypretrain models/lite-mono-pretrain.pth 

CUDA_VISIBLE_DEVICES=1 python train_corres.py --model lite-mono --model_name fast/ablation_c3vd/c3vd_lite-mono_b16e40_raw --data_path rect_c3vd_data \
  --dataset c3vdcorres --split c3vd --batch_size 16 --height 256 --width 320 --png \
  --num_epochs 40 --scales 0 1 2 --num_workers=4 --lr 0.0001 5e-6 41 0.0001 1e-5 41 \
  --matcher_result "playground/heqi/C3VD/matcher_result.npy" \
  --disable_matcher \
  --mypretrain models/lite-mono-pretrain.pth 

CUDA_VISIBLE_DEVICES=1 python train_corres.py --model lite-mono --model_name fast/ablation_c3vd/c3vd_lite-mono_b12e40_raw --data_path rect_c3vd_data \
  --dataset c3vdcorres --split c3vd --batch_size 12 --height 256 --width 320 --png \
  --num_epochs 40 --scales 0 1 2 --num_workers=4 --lr 0.0001 5e-6 41 0.0001 1e-5 41 \
  --matcher_result "playground/heqi/C3VD/matcher_result.npy" \
  --disable_matcher \
  --mypretrain models/lite-mono-pretrain.pth 

CUDA_VISIBLE_DEVICES=1 python train_corres.py --model lite-mono --model_name fast/ablation_c3vd/c3vd_lite-mono_b8e40_raw --data_path rect_c3vd_data \
  --dataset c3vdcorres --split c3vd --batch_size 8 --height 256 --width 320 --png \
  --num_epochs 40 --scales 0 1 2 --num_workers=4 --lr 0.0001 5e-6 41 0.0001 1e-5 41 \
  --matcher_result "playground/heqi/C3VD/matcher_result.npy" \
  --disable_matcher \
  --mypretrain models/lite-mono-pretrain.pth 
  
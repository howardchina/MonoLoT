CUDA_VISIBLE_DEVICES=0 python train_corres.py --model monodepth2 --model_name fast/ab2_cuda0/c3vd_monodepth2_b16e20_raw --data_path rect_c3vd_data \
  --dataset c3vdcorres --split c3vd --batch_size 12 --height 256 --width 320 --png --disable_matcher  \
  --num_epochs 20 --scales 0 1 2 3 --num_workers=4 --lr 0.0001 5e-6 21 0.0001 1e-5 21

CUDA_VISIBLE_DEVICES=0 python train_corres.py --model monodepth2 --model_name fast/ab2_cuda0/c3vd_monodepth2_b12e20_raw --data_path rect_c3vd_data \
  --dataset c3vdcorres --split c3vd --batch_size 8 --height 256 --width 320 --png --disable_matcher  \
  --num_epochs 20 --scales 0 1 2 3 --num_workers=4 --lr 0.0001 5e-6 21 0.0001 1e-5 21

CUDA_VISIBLE_DEVICES=0 python train_corres.py --model monodepth2 --model_name fast/ab2_cuda0/c3vd_monodepth2_b8e20_raw --data_path rect_c3vd_data \
  --dataset c3vdcorres --split c3vd --batch_size 16 --height 256 --width 320 --png --disable_matcher  \
  --num_epochs 20 --scales 0 1 2 3 --num_workers=4 --lr 0.0001 5e-6 21 0.0001 1e-5 21

CUDA_VISIBLE_DEVICES=0 python train_corres.py --model monodepth2 --model_name fast/ab2_cuda0/c3vd_monodepth2_b16e25_raw --data_path rect_c3vd_data \
  --dataset c3vdcorres --split c3vd --batch_size 12 --height 256 --width 320 --png --disable_matcher  \
  --num_epochs 25 --scales 0 1 2 3 --num_workers=4 --lr 0.0001 5e-6 26 0.0001 1e-5 26

CUDA_VISIBLE_DEVICES=0 python train_corres.py --model monodepth2 --model_name fast/ab2_cuda0/c3vd_monodepth2_b12e25_raw --data_path rect_c3vd_data \
  --dataset c3vdcorres --split c3vd --batch_size 12 --height 256 --width 320 --png --disable_matcher  \
  --num_epochs 25 --scales 0 1 2 3 --num_workers=4 --lr 0.0001 5e-6 26 0.0001 1e-5 26

CUDA_VISIBLE_DEVICES=0 python train_corres.py --model monodepth2 --model_name fast/ab2_cuda0/c3vd_monodepth2_b8e25_raw --data_path rect_c3vd_data \
  --dataset c3vdcorres --split c3vd --batch_size 8 --height 256 --width 320 --png --disable_matcher  \
  --num_epochs 25 --scales 0 1 2 3 --num_workers=4 --lr 0.0001 5e-6 26 0.0001 1e-5 26

CUDA_VISIBLE_DEVICES=0 python train_corres.py --model monodepth2 --model_name fast/ab2_cuda0/c3vd_monodepth2_b16e30_raw --data_path rect_c3vd_data \
  --dataset c3vdcorres --split c3vd --batch_size 16 --height 256 --width 320 --png --disable_matcher  \
  --num_epochs 30 --scales 0 1 2 3 --num_workers=4 --lr 0.0001 5e-6 31 0.0001 1e-5 31

CUDA_VISIBLE_DEVICES=0 python train_corres.py --model monodepth2 --model_name fast/ab2_cuda0/c3vd_monodepth2_b12e30_raw --data_path rect_c3vd_data \
  --dataset c3vdcorres --split c3vd --batch_size 12 --height 256 --width 320 --png --disable_matcher  \
  --num_epochs 30 --scales 0 1 2 3 --num_workers=4 --lr 0.0001 5e-6 31 0.0001 1e-5 31

CUDA_VISIBLE_DEVICES=0 python train_corres.py --model monodepth2 --model_name fast/ab2_cuda0/c3vd_monodepth2_b8e30_raw --data_path rect_c3vd_data \
  --dataset c3vdcorres --split c3vd --batch_size 8 --height 256 --width 320 --png --disable_matcher  \
  --num_epochs 30 --scales 0 1 2 3 --num_workers=4 --lr 0.0001 5e-6 31 0.0001 1e-5 31
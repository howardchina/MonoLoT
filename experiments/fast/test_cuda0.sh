CUDA_VISIBLE_DEVICES=0 python train_corres.py --model monodepth2 --model_name fast/test_cuda0/c3vd_monodepth2_b16e20_raw --data_path rect_c3vd_data \
  --dataset c3vdcorres --split c3vd --batch_size 12 --height 256 --width 320 --png  \
  --num_epochs 20 --scales 0 1 2 3 --num_workers=4 --lr 0.0001 5e-6 21 0.0001 1e-5 21 \
  --matcher_result "playground/heqi/C3VD/matcher_result.npy" \
  --disable_matcher

CUDA_VISIBLE_DEVICES=0 python train_corres.py --model monodepth2 --model_name fast/test_cuda0/c3vd_monodepth2_b16e20_me --data_path rect_c3vd_data \
  --dataset c3vdcorres --split c3vd --batch_size 12 --height 256 --width 320 --png \
  --num_epochs 20 --scales 0 1 2 3 --num_workers=4 --lr 0.0001 5e-6 21 0.0001 1e-5 21 \
  --matcher_result "playground/heqi/C3VD/matcher_result.npy" \
  --half_epoch_matcher --matcher_loss_alpha 0.20 

CUDA_VISIBLE_DEVICES=0 python train_corres.py --model lite-mono --model_name fast/test_cuda0/c3vd_litemono_b16e20_raw --data_path rect_c3vd_data \
  --dataset c3vdcorres --split c3vd --batch_size 12 --height 256 --width 320 --png  \
  --num_epochs 20 --scales 0 1 2 --num_workers=4 --lr 0.0001 5e-6 21 0.0001 1e-5 21 \
  --matcher_result "playground/heqi/C3VD/matcher_result.npy" \
  --disable_matcher \
  --mypretrain models/lite-mono-pretrain.pth 

CUDA_VISIBLE_DEVICES=0 python train_corres.py --model lite-mono --model_name fast/test_cuda0/c3vd_litemono_b16e20_me --data_path rect_c3vd_data \
  --dataset c3vdcorres --split c3vd --batch_size 12 --height 256 --width 320 --png \
  --num_epochs 20 --scales 0 1 2 --num_workers=4 --lr 0.0001 5e-6 21 0.0001 1e-5 21 \
  --matcher_result "playground/heqi/C3VD/matcher_result.npy" \
  --half_epoch_matcher --matcher_loss_alpha 0.20 \
  --mypretrain models/lite-mono-pretrain.pth 
  
CUDA_VISIBLE_DEVICES=0 python train_corres.py --model monodepth2 --model_name fast/test_cuda0/simcol_monodepth2_b16e20_raw --data_path simcol_data \
  --dataset simcolcorres --split simcol --batch_size 12 --height 448 --width 448 --png \
  --num_epochs 20 --scales 0 1 2 3 --num_workers=4 --lr 0.0001 5e-6 21 0.0001 1e-5 21 \
  --matcher_result "playground/heqi/Simcol/matcher_result.npy" \
  --disable_matcher

CUDA_VISIBLE_DEVICES=0 python train_corres.py --model monodepth2 --model_name fast/test_cuda0/simcol_monodepth2_b16e20_me --data_path simcol_data \
  --dataset simcolcorres --split simcol --batch_size 12 --height 448 --width 448 --png  \
  --num_epochs 20 --scales 0 1 2 3 --num_workers=4 --lr 0.0001 5e-6 21 0.0001 1e-5 21 \
  --matcher_result "playground/heqi/Simcol/matcher_result.npy" \
  --half_epoch_matcher --matcher_loss_alpha 0.20

CUDA_VISIBLE_DEVICES=0 python train_corres.py --model lite-mono --model_name fast/test_cuda0/simcol_litemono_b16e20_raw --data_path simcol_data \
  --dataset simcolcorres --split simcol --batch_size 12 --height 448 --width 448 --png  \
  --num_epochs 20 --scales 0 1 2 --num_workers=4 --lr 0.0001 5e-6 21 0.0001 1e-5 21 \
  --matcher_result "playground/heqi/Simcol/matcher_result.npy" \
  --disable_matcher \
  --mypretrain models/lite-mono-pretrain.pth

CUDA_VISIBLE_DEVICES=0 python train_corres.py --model lite-mono --model_name fast/test_cuda0/simcol_litemono_b16e20_me --data_path simcol_data \
  --dataset simcolcorres --split simcol --batch_size 12 --height 448 --width 448 --png  \
  --num_epochs 20 --scales 0 1 2 --num_workers=4 --lr 0.0001 5e-6 21 0.0001 1e-5 21 \
  --matcher_result "playground/heqi/Simcol/matcher_result.npy" \
  --half_epoch_matcher --matcher_loss_alpha 0.20 \
  --mypretrain models/lite-mono-pretrain.pth

CUDA_VISIBLE_DEVICES=0 python train_corres.py --model monodepth2 --model_name fast/test_cuda0/kitti_monodepth2_b16e20_raw --data_path kitti_data_standard \
  --dataset kitticorres --split eigen_zhou --batch_size 12 --height 192 --width 640 \
  --num_epochs 20 --scales 0 1 2 3 --num_workers=4 --lr 0.0001 5e-6 21 0.0001 1e-5 21 \
  --matcher_result "playground/heqi/KITTI/matcher_result.npy" \
  --disable_matcher

CUDA_VISIBLE_DEVICES=0 python train_corres.py --model monodepth2 --model_name fast/test_cuda0/kitti_monodepth2_b16e20_me --data_path kitti_data_standard \
  --dataset kitticorres --split eigen_zhou --batch_size 12 --height 192 --width 640 \
  --num_epochs 20 --scales 0 1 2 3 --num_workers=4 --lr 0.0001 5e-6 21 0.0001 1e-5 21 \
  --matcher_result "playground/heqi/KITTI/matcher_result.npy" \
  --half_epoch_matcher --matcher_loss_alpha 0.20

CUDA_VISIBLE_DEVICES=0 python train_corres.py --model lite-mono --model_name fast/test_cuda0/kitti_litemono_b16e20_raw --data_path kitti_data_standard \
  --dataset kitticorres --split eigen_zhou --batch_size 12 --height 192 --width 640 \
  --num_epochs 20 --scales 0 1 2 --num_workers=4 --lr 0.0001 5e-6 21 0.0001 1e-5 21 \
  --matcher_result "playground/heqi/KITTI/matcher_result.npy" \
  --disable_matcher \
  --mypretrain models/lite-mono-pretrain.pth

CUDA_VISIBLE_DEVICES=0 python train_corres.py --model lite-mono --model_name fast/test_cuda0/kitti_litemono_b16e20_me --data_path kitti_data_standard \
  --dataset kitticorres --split eigen_zhou --batch_size 12 --height 192 --width 640 \
  --num_epochs 20 --scales 0 1 2 --num_workers=4 --lr 0.0001 5e-6 21 0.0001 1e-5 21 \
  --matcher_result "playground/heqi/KITTI/matcher_result.npy" \
  --half_epoch_matcher --matcher_loss_alpha 0.20 \
  --mypretrain models/lite-mono-pretrain.pth
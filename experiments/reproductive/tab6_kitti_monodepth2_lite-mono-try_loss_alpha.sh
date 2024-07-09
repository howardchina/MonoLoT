CUDA_VISIBLE_DEVICES=0 python train_corres.py --model lite-mono --model_name reproductive/tab6_kitti/kitti_lite-mono_b12e30_me_0.05 --data_path kitti_data_standard \
  --dataset kitticorres --split eigen_zhou --batch_size 12 --height 192 --width 640 \
  --num_epochs 30 --scales 0 1 2 --num_workers=4 --lr 1e-4 1e-5 31 1e-4 1e-5 31 \
  --matcher_result "playground/heqi/KITTI/matcher_result.npy" \
  --half_epoch_matcher --matcher_loss_alpha 0.05 \
  --mypretrain models/lite-mono-pretrain.pth

CUDA_VISIBLE_DEVICES=0 python train_corres.py --model lite-mono --model_name reproductive/tab6_kitti/kitti_lite-mono_b12e30_me_0.10 --data_path kitti_data_standard \
  --dataset kitticorres --split eigen_zhou --batch_size 12 --height 192 --width 640 \
  --num_epochs 30 --scales 0 1 2 --num_workers=4 --lr 1e-4 1e-5 31 1e-4 1e-5 31 \
  --matcher_result "playground/heqi/KITTI/matcher_result.npy" \
  --half_epoch_matcher --matcher_loss_alpha 0.10 \
  --mypretrain models/lite-mono-pretrain.pth

CUDA_VISIBLE_DEVICES=0 python train_corres.py --model lite-mono --model_name reproductive/tab6_kitti/kitti_lite-mono_b12e30_me_0.40 --data_path kitti_data_standard \
  --dataset kitticorres --split eigen_zhou --batch_size 12 --height 192 --width 640 \
  --num_epochs 30 --scales 0 1 2 --num_workers=4 --lr 1e-4 1e-5 31 1e-4 1e-5 31 \
  --matcher_result "playground/heqi/KITTI/matcher_result.npy" \
  --half_epoch_matcher --matcher_loss_alpha 0.40 \
  --mypretrain models/lite-mono-pretrain.pth

CUDA_VISIBLE_DEVICES=0 python train_corres.py --model lite-mono --model_name reproductive/tab6_kitti/kitti_lite-mono_b12e30_me_0.80 --data_path kitti_data_standard \
  --dataset kitticorres --split eigen_zhou --batch_size 12 --height 192 --width 640 \
  --num_epochs 30 --scales 0 1 2 --num_workers=4 --lr 1e-4 1e-5 31 1e-4 1e-5 31 \
  --matcher_result "playground/heqi/KITTI/matcher_result.npy" \
  --half_epoch_matcher --matcher_loss_alpha 0.80 \
  --mypretrain models/lite-mono-pretrain.pth

CUDA_VISIBLE_DEVICES=0 python train_corres.py --model lite-mono --model_name reproductive/tab6_kitti/kitti_lite-mono_b12e30_me_1.60 --data_path kitti_data_standard \
  --dataset kitticorres --split eigen_zhou --batch_size 12 --height 192 --width 640 \
  --num_epochs 30 --scales 0 1 2 --num_workers=4 --lr 1e-4 1e-5 31 1e-4 1e-5 31 \
  --matcher_result "playground/heqi/KITTI/matcher_result.npy" \
  --half_epoch_matcher --matcher_loss_alpha 1.60 \
  --mypretrain models/lite-mono-pretrain.pth

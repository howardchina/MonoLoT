
# CUDA_VISIBLE_DEVICES=1 python train_corres.py --model monodepth2 --model_name reproductive/tab6_kitti/kitti_monodepth2_b12e20_raw --data_path kitti_data_standard \
#   --dataset kitticorres --split eigen_zhou --batch_size 12 --height 192 --width 640 \
#   --num_epochs 20 --scales 0 1 2 3 --num_workers=4 --lr 1e-4 1e-5 21 1e-4 1e-5 21 \
#   --matcher_result "playground/heqi/KITTI/matcher_result.npy" \
#   --disable_matcher
  
# CUDA_VISIBLE_DEVICES=1 python train_corres.py --model monodepth2 --model_name reproductive/tab6_kitti/kitti_monodepth2_b12e20_me --data_path kitti_data_standard \
#   --dataset kitticorres --split eigen_zhou --batch_size 12 --height 192 --width 640 \
#   --num_epochs 20 --scales 0 1 2 3 --num_workers=4 --lr 1e-4 1e-5 21 1e-4 1e-5 21 \
#   --matcher_result "playground/heqi/KITTI/matcher_result.npy" \
#   --half_epoch_matcher --matcher_loss_alpha 0.20

CUDA_VISIBLE_DEVICES=0 python train_corres.py --model lite-mono --model_name reproductive/tab6_kitti/kitti_lite-mono_b12e30_raw --data_path kitti_data_standard \
  --dataset kitticorres --split eigen_zhou --batch_size 12 --height 192 --width 640 \
  --num_epochs 30 --scales 0 1 2 --num_workers=4 --lr 1e-4 1e-5 31 1e-4 1e-5 31 \
  --matcher_result "playground/heqi/KITTI/matcher_result.npy" \
  --disable_matcher \
  --mypretrain models/lite-mono-pretrain.pth 

CUDA_VISIBLE_DEVICES=0 python train_corres.py --model lite-mono --model_name reproductive/tab6_kitti/kitti_lite-mono_b12e30_me --data_path kitti_data_standard \
  --dataset kitticorres --split eigen_zhou --batch_size 12 --height 192 --width 640 \
  --num_epochs 30 --scales 0 1 2 --num_workers=4 --lr 1e-4 1e-5 31 1e-4 1e-5 31 \
  --matcher_result "playground/heqi/KITTI/matcher_result.npy" \
  --half_epoch_matcher --matcher_loss_alpha 0.20 \
  --mypretrain models/lite-mono-pretrain.pth
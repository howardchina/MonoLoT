CUDA_VISIBLE_DEVICES=1 python train_corres.py --model monodepth2 --model_name reproductive/tab6_kitti/kitti_monodepth2_b12e20_me_0.05 --data_path kitti_data_standard \
  --dataset kitticorres --split eigen_zhou --batch_size 12 --height 192 --width 640 \
  --num_epochs 20 --scales 0 1 2 3 --num_workers=4 --lr 1e-4 1e-5 21 1e-4 1e-5 21 \
  --matcher_result "playground/heqi/KITTI/matcher_result.npy" \
  --half_epoch_matcher --matcher_loss_alpha 0.05

CUDA_VISIBLE_DEVICES=1 python train_corres.py --model monodepth2 --model_name reproductive/tab6_kitti/kitti_monodepth2_b12e20_me_0.10 --data_path kitti_data_standard \
  --dataset kitticorres --split eigen_zhou --batch_size 12 --height 192 --width 640 \
  --num_epochs 20 --scales 0 1 2 3 --num_workers=4 --lr 1e-4 1e-5 21 1e-4 1e-5 21 \
  --matcher_result "playground/heqi/KITTI/matcher_result.npy" \
  --half_epoch_matcher --matcher_loss_alpha 0.10
  
CUDA_VISIBLE_DEVICES=1 python train_corres.py --model monodepth2 --model_name reproductive/tab6_kitti/kitti_monodepth2_b12e20_me_0.40 --data_path kitti_data_standard \
  --dataset kitticorres --split eigen_zhou --batch_size 12 --height 192 --width 640 \
  --num_epochs 20 --scales 0 1 2 3 --num_workers=4 --lr 1e-4 1e-5 21 1e-4 1e-5 21 \
  --matcher_result "playground/heqi/KITTI/matcher_result.npy" \
  --half_epoch_matcher --matcher_loss_alpha 0.40
  
CUDA_VISIBLE_DEVICES=1 python train_corres.py --model monodepth2 --model_name reproductive/tab6_kitti/kitti_monodepth2_b12e20_me_0.80 --data_path kitti_data_standard \
  --dataset kitticorres --split eigen_zhou --batch_size 12 --height 192 --width 640 \
  --num_epochs 20 --scales 0 1 2 3 --num_workers=4 --lr 1e-4 1e-5 21 1e-4 1e-5 21 \
  --matcher_result "playground/heqi/KITTI/matcher_result.npy" \
  --half_epoch_matcher --matcher_loss_alpha 0.80
  
CUDA_VISIBLE_DEVICES=1 python train_corres.py --model monodepth2 --model_name reproductive/tab6_kitti/kitti_monodepth2_b12e20_me_1.60 --data_path kitti_data_standard \
  --dataset kitticorres --split eigen_zhou --batch_size 12 --height 192 --width 640 \
  --num_epochs 20 --scales 0 1 2 3 --num_workers=4 --lr 1e-4 1e-5 21 1e-4 1e-5 21 \
  --matcher_result "playground/heqi/KITTI/matcher_result.npy" \
  --half_epoch_matcher --matcher_loss_alpha 1.60
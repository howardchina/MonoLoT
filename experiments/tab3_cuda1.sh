CUDA_VISIBLE_DEVICES=1 python train_v2.py --model_name tab3_cuda1/kitti_litemono_b12e30 --data_path kitti_data_standard \
  --dataset kitti --split eigen_zhou --batch_size 12 --height 192 --width 640 --disable_matcher --use_cudnn \
  --num_epochs 30 --scales 0 1 2 --num_workers=4 --lr 0.0001 5e-6 31 0.0001 1e-5 31

CUDA_VISIBLE_DEVICES=1 python train_v2.py --model_name tab3_cuda1/kitti_litemono_b12e30_me --data_path kitti_data_standard \
  --dataset kitti --split eigen_zhou --batch_size 12 --height 192 --width 640 --use_cudnn \
  --num_epochs 30 --scales 0 1 2 --num_workers=4 --lr 0.0001 5e-6 31 0.0001 1e-5 31 --half_epoch_matcher --matcher_loss_alpha 0.20 

CUDA_VISIBLE_DEVICES=1 python train.py --model_name tab3_cuda1/kitti_monodepth2_b12e20 --data_path kitti_data_standard \
  --dataset kitti --split eigen_zhou --batch_size 12 --height 192 --width 640 --disable_matcher --use_cudnn \
  --num_epochs 20 --scales 0 1 2 3  --num_workers=4 --scheduler_step_size 15

CUDA_VISIBLE_DEVICES=1 python train.py --model_name tab3_cuda1/kitti_monodepth2_b12e20_me --data_path kitti_data_standard \
  --dataset kitti --split eigen_zhou --batch_size 12 --height 192 --width 640 --use_cudnn \
  --num_epochs 20 --scales 0 1 2 3  --num_workers=4 --scheduler_step_size 15 --half_epoch_matcher --matcher_loss_alpha 0.20 

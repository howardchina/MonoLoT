CUDA_VISIBLE_DEVICES=0 python train_v2.py --model_name tab3_cuda0/kitti_litemono_b12e30 --data_path kitti_data_standard \
  --dataset kitti --split eigen_zhou --batch_size 12 --height 192 --width 640 --disable_matcher  \
  --num_epochs 30 --scales 0 1 2 --num_workers=12 --lr 0.0001 5e-6 31 0.0001 1e-5 31 --mypretrain models/lite-mono-pretrain.pth

CUDA_VISIBLE_DEVICES=0 python train_v2.py --model_name tab3_cuda0/kitti_litemono_b12e30_me --data_path kitti_data_standard \
  --dataset kitti --split eigen_zhou --batch_size 12 --height 192 --width 640 \
  --num_epochs 30 --scales 0 1 2 --num_workers=12 --lr 0.0001 5e-6 31 0.0001 1e-5 31 --mypretrain models/lite-mono-pretrain.pth --half_epoch_matcher --matcher_loss_alpha 0.20 
CUDA_VISIBLE_DEVICES=1 python train_corres.py --model lite-mono --model_name c3vd/lite-mono_b8e35_raw/depth_0.0001_0.1/ --data_path rect_c3vd_data \
  --dataset c3vdcorres --split c3vd --batch_size 8 --height 256 --width 320 --png \
  --num_epochs 35 --scales 0 1 2 --num_workers=4 --lr 0.0001 1e-5 36 0.0001 1e-5 36 \
  --matcher_result "playground/heqi/C3VD/matcher_result.npy" \
  --disable_matcher   --seed 3407 \
  --min_depth 0.0001 --max_depth 0.1 \
  --mypretrain models/lite-mono-pretrain.pth 
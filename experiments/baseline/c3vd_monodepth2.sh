CUDA_VISIBLE_DEVICES=0 python train_corres.py --model monodepth2 --model_name baseline/c3vd_monodepth2 --data_path rect_c3vd_data \
  --dataset c3vdcorres --split c3vd --batch_size 12 --height 256 --width 320 --png \
  --num_epochs 35 --scales 0 1 2 3 --num_workers=4 --lr 0.0001 1e-5 36 0.0001 1e-5 36 \
  --matcher_result "playground/heqi/C3VD/matcher_result.npy" \
  --min_depth 0.1 --max_depth 100.0 \
  --disable_matcher

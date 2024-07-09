# [JBHI'24] MonoLoT
Official Pytorch implementation of **MonoLoT: Self-Supervised Monocular Depth Estimation in Low-Texture Scenes for Automatic Robotic Endoscopy**.

[Qi HE](https://howardchina.github.io), 
Guang Feng, 
[Sophia Bano](https://sophiabano.github.io/),
[Danail Stoyanov](https://profiles.ucl.ac.uk/32290),
[Siyang Zuo](https://scholar.google.co.uk/citations?user=FPFY0CgAAAAJ&hl=en)

[[`PDF`](https://drive.google.com/file/d/12DyEv0R_qfWXiO7U9K04PKGjAxE6sLnw/view?usp=sharing)] [[`Youtube`](https://youtu.be/7mAyTpjI13Q?si=JvnwjsjJyxkJ7LiD)] [[`Github`](https://github.com/howardchina/MonoLoT)]


## Overview
<p align="left">
<img src="assets/overview.jpg" width=80% height=80% 
class="center">
</p>

Our research has introduced an innovative approach that addresses the challenges associated with self-supervised monocular depth estimation in digestive endoscopy. We have addressed two critical aspects: the limitations of self-supervised depth estimation in low-texture scenes and the application of
depth estimation in visual servoing for digestive endoscopy.
Our investigation has revealed that the struggles of self-supervised depth estimation in low-texture scenes stem from inaccurate photometric reconstruction losses. To overcome this, we have introduced the point-matching loss, which refines the reprojected points. Furthermore, during the training process, data augmentation is achieved through batch image shuffle loss, significantly improving the accuracy and generalisation capability of the depth model. The combined contributions of the point matching loss and batch image shuffle loss have boosted the baseline accuracy by a minimum of 5% on both the C3VD and SimCol datasets, surpassing the generalisability of ground truth depth-supervised baselines when applied to upper-GI datasets. Moreover, the successful implementation of our robotic platform for automatic intervention in digestive endoscopy demonstrates the practical and impactful application of monocular depth estimation technology.

## Performance
<p align="left">
<img src="assets/performance.jpg" width=80% height=80% 
class="center">
</p>

## Supplementrary Material
<iframe width="560" height="315" src="https://www.youtube.com/embed/7mAyTpjI13Q?si=JvnwjsjJyxkJ7LiD" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>


## Installation

We tested our code on a server with Ubuntu 18.04.6, cuda 11.1, gcc 7.5.0
1. Clone project
```
git clone https://github.com/howardchina/MonoLoT.git
cd MonoLoT
```
2. Install environment
```
conda create --name monolot --file requirements.txt
conda activate monolot
```

## Data

First, create a ```data/``` folder inside the project path by 
```
mkdir data
```

The data structure will be organised as follows:

```
data/
├── dataset_name
│   ├── scene1/
│   │   ├── images
│   │   │   ├── IMG_0.jpg
│   │   │   ├── IMG_1.jpg
│   │   │   ├── ...
│   │   ├── sparse/
│   │       └──0/
│   ├── scene2/
│   │   ├── images
│   │   │   ├── IMG_0.jpg
│   │   │   ├── IMG_1.jpg
│   │   │   ├── ...
│   │   ├── sparse/
│   │       └──0/
...
```

 - For instance: `./data/blending/drjohnson/`
 - For instance: `./data/bungeenerf/amsterdam/`
 - For instance: `./data/mipnerf360/bicycle/`
 - For instance: `./data/nerf_synthetic/chair/`
 - For instance: `./data/tandt/train/`


### Public Data (We follow suggestions from [Scaffold-GS](https://github.com/city-super/Scaffold-GS))

 - The **BungeeNeRF** dataset is available in [Google Drive](https://drive.google.com/file/d/1nBLcf9Jrr6sdxKa1Hbd47IArQQ_X8lww/view?usp=sharing)/[百度网盘[提取码:4whv]](https://pan.baidu.com/s/1AUYUJojhhICSKO2JrmOnCA). 
 - The **MipNeRF360** scenes are provided by the paper author [here](https://jonbarron.info/mipnerf360/). And we test on its entire 9 scenes ```bicycle, bonsai, counter, garden, kitchen, room, stump, flowers, treehill```. 
 - The SfM datasets for **Tanks&Temples** and **Deep Blending** are hosted by 3D-Gaussian-Splatting [here](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip). Download and uncompress them into the ```data/``` folder.

### Custom Data

For custom data, you should process the image sequences with [Colmap](https://colmap.github.io/) to obtain the SfM points and camera poses. Then, place the results into ```data/``` folder.

## Training

To train scenes, we provide the following training scripts: 
 - Tanks&Temples: ```run_shell_tnt.py```
 - MipNeRF360: ```run_shell_mip360.py```
 - BungeeNeRF: ```run_shell_bungee.py```
 - Deep Blending: ```run_shell_db.py```
 - Nerf Synthetic: ```run_shell_blender.py```

 run them with 
 ```
 python run_shell_xxx.py
 ```

The code will automatically run the entire process of: **training, encoding, decoding, testing**.
 - Training log will be recorded in `output.log` of the output directory. Results of **detailed fidelity, detailed size, detailed time** will all be recorded
 - Encoded bitstreams will be stored in `./bitstreams` of the output directory.
 - Evaluated output images will be saved in `./test/ours_30000/renders` of the output directory.
 - Optionally, you can change `lmbda` in these `run_shell_xxx.py` scripts to try variable bitrate.
 - **After training, the original model `point_cloud.ply` is losslessly compressed as `./bitstreams`. You should refer to `./bitstreams` to get the final model size, but not `point_cloud.ply`. You can even delete `point_cloud.ply` if you like :).**


## Contact

- Yihang Chen: yhchen.ee@sjtu.edu.cn

## Citation

If you find our work helpful, please consider citing:

```bibtex
@inproceedings{hac2024,
  title={HAC: Hash-grid Assisted Context for 3D Gaussian Splatting Compression},
  author={Chen, Yihang and Wu, Qianyi and Lin, Weiyao and Harandi, Mehrtash and Cai, Jianfei},
  booktitle={European Conference on Computer Vision},
  year={2024}
}
```


## LICENSE

Please follow the LICENSE of [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting).

## Acknowledgement

 - We thank all authors from [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting) for presenting such an excellent work.
 - We thank all authors from [Scaffold-GS](https://github.com/city-super/Scaffold-GS) for presenting such an excellent work.
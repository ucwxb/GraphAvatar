# GraphAvatar: Compact Head Avatars with GNN-Generated 3D Gaussians

> [GraphAvatar: Compact Head Avatars with GNN-Generated 3D Gaussians](https://arxiv.org/abs/2412.13983)  
> [Xiaobao Wei](https://ucwxb.github.io/), [Peng Chen](https://chenvoid.github.io/), [Ming Lu](https://lu-m13.github.io/), Hui Chen $^\dagger$ , Feng Tian  
> AAAI2025 Main Conference Paper  
> $\dagger$ Corresponding author

![vis](./assets/teaser.png)

We propose a compact method named GraphAvatar that leverages Graph Neural Networks (GNN) to generate the 3D Gaussians for head avatar animation.

## News
- **[2024/12/20]** Code & example data release!
- **[2024/12/10]** GraphAvatar is accepted by AAAI2025!

## Overview
![overview](./assets/pipeline.png)

Rendering photorealistic head avatars from arbitrary viewpoints is crucial for various applications like virtual reality. Although previous methods based on Neural Radiance Fields (NeRF) can achieve impressive results, they lack fidelity and efficiency. Recent methods using 3D Gaussian Splatting (3DGS) have improved rendering quality and real-time performance but still require significant storage overhead. In this paper, we introduce a method called GraphAvatar that utilizes Graph Neural Networks (GNN) to generate 3D Gaussians for the head avatar. Specifically, GraphAvatar trains a geometric GNN and an appearance GNN to generate the attributes of the 3D Gaussians from the tracked mesh. Therefore, our method can store the GNN models instead of the 3D Gaussians, significantly reducing the storage overhead to just 10MB. To reduce the impact of face-tracking errors, we also present a novel graph-guided optimization module to refine face-tracking parameters during training. Finally, we introduce a 3D-aware enhancer for post-processing to enhance the rendering quality. We conduct comprehensive experiments to demonstrate the advantages of GraphAvatar, surpassing existing methods in visual fidelity and storage consumption. The ablation study sheds light on the trade-offs between rendering quality and model size.

## Getting Started

### Environmental Setups
Our code is developed on Ubuntu 20.04 using Python 3.8 and pytorch=1.12.0+cu113. We recommend using conda for the installation of dependencies.

```bash

git clone https://github.com/ucwxb/GraphAvatar
cd GraphAvatar
conda env create -f env.yaml
conda activate GraphAvatar

pip install -r requirements.txt
sudo apt-get install -y \
    freeglut3-dev \
    python3-opengl \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    mesa-common-dev \
    libxmu-dev \
    libxi-dev
pip install PyOpenGL PyOpenGL-accelerate
pip install https://data.pyg.org/whl/torch-1.12.0%2Bcu116/torch_cluster-1.6.0%2Bpt112cu116-cp39-cp39-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.12.0%2Bcu116/torch_scatter-2.1.0%2Bpt112cu116-cp39-cp39-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.12.0%2Bcu116/torch_sparse-0.6.16%2Bpt112cu116-cp39-cp39-linux_x86_64.whl

pip install https://data.pyg.org/whl/torch-2.0.0%2Bcu118/torch_cluster-1.6.3%2Bpt20cu118-cp39-cp39-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-2.0.0%2Bcu118/torch_sparse-0.6.18%2Bpt20cu118-cp39-cp39-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-2.0.0%2Bcu118/torch_scatter-2.1.2%2Bpt20cu118-cp39-cp39-linux_x86_64.whl


pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

cd submodules/diff-gaussian-rasterization
pip install -e .

cd ../..
cd submodules/simple-knn
pip install -e .

cd ../..
cd submodules/mesh
python setup.py install

sed -i 's/out = op(src, index, dim, None, dim_size, fill_value)/out = op(src, index, dim=dim, out=None, dim_size=dim_size)/' $CONDA_PREFIX/lib/python3.9/site-packages/torch_geometric/utils/scatter.py
```

### Preparing Dataset and checkpoint of SAM
To validate the performance of binary polyp segmentation, we have provided the [link](https://drive.google.com/drive/folders/101LDnr7Gget7ehZQkHCNH1csD2WCCBX6?usp=sharing) for datasets sessile-Kvasir and CVC. 
Please create a new folder named "dataset", download and unzip the two datasets into the folder.

```bash
mkdir dataset
# download and move the zip files into the folder

unzip sessile-Kvasir.zip
unzip CVC.zip
```
Please donwload ViT-B SAM checkpoint from this [link](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) and place it into the "sam_ckp" folder.
```bash
mkdir sam_ckp
cd sam_ckp
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```
Finally the file structure is organized as:
```
I-MedSAM
├── dataset
│   ├── sessile-Kvasir
│   |   ├── train
│   |   ├── val
│   |   ├── test
│   ├── CVC
│   |   ├── PNG
│   |   |   ├── Ground Truth
│   |   |   ├── Original
├── sam_ckp
│   ├── sam_vit_b_01ec64.pth
└── other codes...
```

### Training

For training on sessile-Kvasir with the image shape 384x384, please run:
```bash
# for single GPU
bash scripts/train/train_sessile.sh

# for multi GPU. The current settings are for 8 GPUs. If you have less GPUs, please change CUDA_VISIBLE_DEVICES and nproc_per_node.
bash scripts/train/train_sessile_multi.sh
```
Then you can find checkpoints and training logs into the folder "work_dir".

### Evaluation and Visualization
The checkpoint trained on sessile-Kvasir can be found [here](https://drive.google.com/file/d/1qd1FNoc3Io2g8t9HCjaELNYxOHLoiVO0/view?usp=sharing). 
Please download it and place it into the folder "work_dir".
You can follow the test scripts for testing on different experiments settings:
```bash
# test with shape 384x384
bash scripts/test/test_sessile.sh

# cross resolution: from 384 to 128, it changes the param "--label_size" from 384 to 128
bash scripts/test/test_sessile_384_to_128.sh

# cross resolution: from 384 to 896, it changes the param "--label_size" from 384 to 896
bash scripts/test/test_sessile_384_to_896.sh

# cross domain: from sessile-Kvasir to CVC, it set the param "--data_path" to the path of CVC
bash scripts/test/test_sessile_to_CVC.sh
```
You can also modify the param "--resume" as the path of your trained checkpoint.
The test process also supports running on multi GPU, which is the same as training. 
Please refer to the test scripts to change from single GPU into multi GPU.

to visualize the segmentation masks, you can add the argument "--save_pic" into the scripts to save results.


## Citation

If you find this project helpful, please consider citing the our paper:
```
@article{wei2025graphavatar,
  title={GraphAvatar: Compact Head Avatars with GNN-Generated 3D Gaussians},
  author={Wei, Xiaobao and Chen, Peng and Lu, Ming and Chen, Hui and Tian, Feng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2025}
}
```

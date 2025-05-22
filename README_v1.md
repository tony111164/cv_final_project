## Installation

1. Clone DUSt3R.
```bash
git clone --recursive https://github.com/tony111164/cv_final_project.git
cd cv_final_project
# if you have already cloned dust3r:
# git submodule update --init --recursive
```

2. Create the environment, here we show an example using conda.
```bash
conda create -n dust3r python=3.11 cmake=3.14.0
conda activate dust3r 
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia  # use the correct version of cuda for your system
pip install -r requirements.txt
# Optional: you can also install additional packages to:
# - add support for HEIC images
# - add pyrender, used to render depthmap in some datasets preprocessing
# - add required packages for visloc.py
pip install -r requirements_optional.txt
```

3. Optional, compile the cuda kernels for RoPE (as in CroCo v2).
```bash
# DUST3R relies on RoPE positional embeddings for which you can compile some cuda kernels for faster runtime.
cd croco/models/curope/
python setup.py build_ext --inplace
cd ../../../
```

## Checkpoints

You can obtain the checkpoints by two ways:

1) You can use our huggingface_hub integration: the models will be downloaded automatically.

2) Otherwise, We provide several pre-trained models:

| Modelname   | Training resolutions | Head | Encoder | Decoder |
|-------------|----------------------|------|---------|---------|
| [`DUSt3R_ViTLarge_BaseDecoder_224_linear.pth`](https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth) | 224x224 | Linear | ViT-L | ViT-B |
| [`DUSt3R_ViTLarge_BaseDecoder_512_linear.pth`](https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_linear.pth)   | 512x384, 512x336, 512x288, 512x256, 512x160 | Linear | ViT-L | ViT-B |
| [`DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth`](https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth) | 512x384, 512x336, 512x288, 512x256, 512x160 | DPT | ViT-L | ViT-B |

You can check the hyperparameters we used to train these models in the [section: Our Hyperparameters](#our-hyperparameters)

To download a specific model, for example `DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth`:
```bash
mkdir -p checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth -P checkpoints/
```

For the checkpoints, make sure to agree to the license of all the public training datasets and base checkpoints we used, in addition to CC-BY-NC-SA 4.0. Again, see [section: Our Hyperparameters](#our-hyperparameters) for details.


## Key Modifications

### Data

- `dust3r/datasets/sevenscenes.py`
    
    + Automatically extracts image pairs, corresponding depth maps, and camera poses from the 7Scenes dataset, and formats them into the input structure required for training the DUSt3R model.

- `/home/tony/dust3r/dust3r/datasets/__init__.py`

    + add "from .sevenscenes import SevenScenes  # noqa"

### Training & testing

- `/home/tony/dust3r/train_dust3r.sh`

    + Command to execute fine-tuning
    + You can execute it using the following command: `bash train_dust3r.sh`

### inference

- `/home/tony/dust3r/infer_dust3r.py`

    + Reads test image sequences from the 7Scenes dataset, uses a fine-tuned Dust3r model to infer dense 3D point clouds from image pairs, and saves the world-aligned point clouds as .ply files.


# cv final project

## Installation

1. Clone DUSt3R.
```bash
git clone --recursive https://github.com/tony111164/cv_final_project.git
cd cv_final_project
```

2. Create the environment
```bash
conda create -n dust3r python=3.11 cmake=3.14.0
conda activate dust3r 
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia  # use the correct version of cuda for your system
pip install -r requirements.txt
```
## Checkpoints

| Modelname   | Training resolutions | Head | Encoder | Decoder |
|-------------|----------------------|------|---------|---------|
| [`DUSt3R_ViTLarge_BaseDecoder_224_linear.pth`](https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth) | 224x224 | Linear | ViT-L | ViT-B |
| [`DUSt3R_ViTLarge_BaseDecoder_512_linear.pth`](https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_linear.pth)   | 512x384, 512x336, 512x288, 512x256, 512x160 | Linear | ViT-L | ViT-B |
| [`DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth`](https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth) | 512x384, 512x336, 512x288, 512x256, 512x160 | DPT | ViT-L | ViT-B |

To download a specific model, for example `DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth`:
```bash
mkdir -p checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth -P checkpoints/
```

## Key Modifications

### Data preprocessing

- `dust3r/datasets/sevenscenes.py`
    
    + Automatically extracts image pairs, corresponding depth maps, and camera poses from the 7Scenes dataset, and formats them into the input structure required for training the DUSt3R model.

- `dust3r/datasets/__init__.py`

    + add "from .sevenscenes import SevenScenes  # noqa"

### Training & testing

- `train_dust3r.sh`

    + Command to execute fine-tuning
    + You can execute it using the following command: `bash train_dust3r.sh`

### inference

- `infer_dust3r.py`

    + Reads test image sequences from the 7Scenes dataset, uses a fine-tuned Dust3r model to infer dense 3D point clouds from image pairs, and saves the world-aligned point clouds as .ply files.
    + You can execute it using the following command: `python infer_dust3r.py`

### visualize

- `ply_visualize.py`

    + Execute it to visualize PLY files in the test folder
    + First go to the PLY folder with `cd PLY`, then you can run it using the command: `python3 ply_visualize.py test/{PLY_file}`

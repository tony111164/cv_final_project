# CV Final Project - 2025
# TEAM: 像素動力

## Slides
- [📊 Presentation Slides](https://3dreconstruction.my.canva.site/)

## Installation

1. Clone the repository:
```bash
git clone --branch final --recursive https://github.com/tony111164/cv_final_project.git
cd cv_final_project
```

2. Create the conda environment and install dependencies:
```bash
conda create -n dust3r python=3.11 cmake=3.14.0
conda activate dust3r 
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia  # use the correct version of cuda for your system
pip install -r requirements.txt
```
## Usage Flow
1. Run Fine-tuning
```bash
bash train_dust3r.sh
```
2. Run Dense Inference
```bash
python infer_dust3r.py \
  --scenes_root <7SCENES_dir> \
  --output_dir PLY/seq_output \
  --model_ckpt checkpoints/dust3r_7scenes/train_3000/checkpoint-best.pth \
  --step 20 \
  --voxel_size 7.5e-3 \
  --seq_pattern seq-*
```
3. Run Sparse Inference
```bash
python infer_dust3r.py \
  --scenes_root <7SCENES_dir> \
  --output_dir PLY/sparse_seq_output \
  --model_ckpt checkpoints/dust3r_7scenes/train_3000/checkpoint-best.pth \
  --step 1 \
  --voxel_size 7.5e-3 \
  --seq_pattern sparse-seq-*
```

## Inference Results
- Dense seq:

| Scene           | Accuracy  | Completeness |
|----------------|-----------|--------------|
| **AVERAGE**    | **0.25** | **0.62** |

- Sparse seq:

| Scene           | Accuracy  | Completeness |
|----------------|-----------|--------------|
| CHESS-SEQ-05   | 0.201421  | 0.194138     |
| FIRE-SEQ-04    | 0.457314  | 0.166710     |
| PUMPKIN-SEQ-07 | 0.211218  | 0.294592     |
| STAIRS-SEQ-04  | 0.264569  | 0.083909     |
| **AVERAGE**    | **0.283631** | **0.184837** |


----------------------------------------------------------------------------------------------------------

## Implementation Details
- Three DUSt3R pretrained models:

| Modelname   | Training resolutions | Head | Encoder | Decoder |
|-------------|----------------------|------|---------|---------|
| [`DUSt3R_ViTLarge_BaseDecoder_224_linear.pth`](https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth) | 224x224 | Linear | ViT-L | ViT-B |
| [`DUSt3R_ViTLarge_BaseDecoder_512_linear.pth`](https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_linear.pth)   | 512x384, 512x336, 512x288, 512x256, 512x160 | Linear | ViT-L | ViT-B |
| [`DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth`](https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth) | 512x384, 512x336, 512x288, 512x256, 512x160 | DPT | ViT-L | ViT-B |

To download a specific model, for example `DUSt3R_ViTLarge_BaseDecoder_224_dpt.pth`:
```bash
mkdir -p checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth -P checkpoints/
```

#### Data Preprocessing
- `dust3r/datasets/sevenscenes.py`
    
    + Extracts image pairs, depth maps, and camera poses from the 7Scenes dataset.
    + Formats data into the structure required for DUSt3R fine-tuning.

- `dust3r/datasets/__init__.py`

    + Added `from .sevenscenes import SevenScenes` to register the dataset class.

#### Fine-tuning
- `train_dust3r.sh`

    + Bash script to run fine-tuning with 7Scenes.
    + Run with:
    ```bash
    bash train_dust3r.sh
    ```

#### Inference
- `infer_dust3r.py`

    + Loads a fine-tuned model to infer dense 3D point clouds from RGB image pairs.
    + Applies coordinate transformation and exports `.ply` files.
    + Run with:
        + Dense Inference
        ```bash
        python infer_dust3r.py \
        --scenes_root <7SCENES_dir> \
        --output_dir PLY/seq_output \
        --model_ckpt checkpoints/dust3r_7scenes/train_3000/checkpoint-best.pth \
        --step 20 \
        --voxel_size 7.5e-3 \
        --seq_pattern seq-*
        ```
        + Sparse Inference
        ```bash
        python infer_dust3r.py \
        --scenes_root <7SCENES_dir> \
        --output_dir PLY/sparse_seq_output \
        --model_ckpt checkpoints/dust3r_7scenes/train_3000/checkpoint-best.pth \
        --step 1 \
        --voxel_size 7.5e-3 \
        --seq_pattern sparse-seq-*
        ```

#### PLY Point Cloud Evaluation
- `evaluate_file.py`
    
    + This script evaluates the quality of predicted 3D point clouds (`.ply`) against ground-truth point clouds (`.ply`)
    + Usage:
    ```bash
    cd PLY
    python evaluate_file.py <predicted_dir> <ground_truth_dir> --output <result_file>
    ```

#### Visualize
- `ply_visualize.py`

    + Visualizes `.ply` files generated during inference using Open3D.
    + Usage:
    ```bash
    cd PLY
    python3 ply_visualize.py test/{PLY_file}
    ```

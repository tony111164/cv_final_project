<div align="center">

# ⚡️Fast3R: Towards 3D Reconstruction of 1000+ Images in One Forward Pass


${{\color{Red}\Huge{\textsf{  CVPR\ 2025\ \}}}}\$


[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=b31b1b)](https://arxiv.org/abs/2501.13928)
[![Project Website](https://img.shields.io/badge/Fast3R-Website-4CAF50?logo=googlechrome&logoColor=white)](https://fast3r-3d.github.io/)
[![Gradio Demo](https://img.shields.io/badge/Gradio-Demo-orange?style=flat&logo=Gradio&logoColor=red)](https://fast3r.ngrok.app/)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/jedyang97/Fast3R_ViT_Large_512/)
</div>

![Teaser Image](assets/teaser.png)

Official implementation of **Fast3R: Towards 3D Reconstruction of 1000+ Images in One Forward Pass**, CVPR 2025

*[Jianing Yang](https://jedyang.com/), [Alexander Sax](https://alexsax.github.io/), [Kevin J. Liang](https://kevinjliang.github.io/), [Mikael Henaff](https://www.mikaelhenaff.net/), [Hao Tang](https://tanghaotommy.github.io/), [Ang Cao](https://caoang327.github.io/), [Joyce Chai](https://web.eecs.umich.edu/~chaijy/), [Franziska Meier](https://fmeier.github.io/), [Matt Feiszli](https://www.linkedin.com/in/matt-feiszli-76b34b/)*

## Installation

```bash
# clone project
git clone ......
cd fast3r

# create conda environment
conda create -n fast3r python=3.11 cmake=3.14.0 -y
conda activate fast3r

# install PyTorch (adjust cuda version according to your system)
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 nvidia/label/cuda-12.4.0::cuda-toolkit -c pytorch -c nvidia

# install requirements
pip install -r requirements.txt

# install fast3r as a package (so you can import fast3r and use it in your own project)
pip install -e .
```

Note: Please make sure to NOT install the cuROPE module like in DUSt3R - it would mess up Fast3R's prediction.

## Dataset

Download the 7-Scenes dataset provided by the TA

## Training

To train Fast3R on the 7-Scenes dataset using the `super_long_training` experiment configuration:
```bash
CUDA_VISIBLE_DEVICES=1 python fast3r/train.py \
    experiment=super_long_training/super_long_training \
    data.data_root=/home/tony/7SCENES \
    trainer.plugins=null \
    model.pretrained=null
```
- Use GPU 1 (`CUDA_VISIBLE_DEVICES=1`) (can skip it!)

- Load data from your local directory: `/home/7SCENES`

- Disable plugins (e.g., SLURM, DDP) by setting `trainer.plugins=null`

- Disable loading any pretrained weights (`model.pretrained=null`)

Note: For more details, please refer to the original README.md !!!

## Evaluation

To evaluate on 3D reconstruction tasks, run:

```bash
python fast3r/eval.py \
    eval=ablation_recon_better_inference_hp/ablation_recon_better_inference_hp \
    ckpt_path=/home/tony/fast3r/logs/super_long_training/runs/super_long_training_99999/checkpoints/last-v3.ckpt \
    slurm_job_id=local \
    data.data_root=/home/tony/7SCENES \
    trainer.plugins=null
```
- `ablation_recon_better_inference_hp/ablation_recon_better_inference_hp` evaluates the 3D reconstruction on DTU, 7-Scenes and Neural-RGBD datasets.

- Load the trained model checkpoint from your training logs

- Run locally without using SLURM (`slurm_job_id=local`)

- Load the 7-Scenes dataset from `/home/tony/7SCENES`

- Disable plugins (`trainer.plugins=null`) to simplify execution

Note: For more details, please refer to the original README.md !!!

## Key Modifications
- `fast3r/data/components/spann3r_datasets/seven_scenes.py`
    
    Train, validation, and test dataset handling

- `fast3r/configs/experiment/super_long_training/super_long_training.yaml`

    Adjusted some config parameters, hyperparameters

- `fast3r/configs/eval/ablation_recon_better_inference_hp/ablation_recon_better_inference_hp.yaml`
    
    Adjusted some config parameters

- `commands for fast3r/train.py & fast3r/eval.py`

    Adjusted some config parameters

Note: It is recommended to compare with the original code.

## Question

1. We're not allowed to use pose information during the inference/testing stage, but it seems that eval.py requires it?

2. How to generate .ply files from the test set?

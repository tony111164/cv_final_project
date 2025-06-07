import torch
from spann3r.datasets.seven_scenes import SevenScenes
import random

# 模擬你的資料資料夾
DATA_ROOT = "./data/7SCENES"   # 這裡換成你的資料夾

# 建立 Dataset
dataset = SevenScenes(
    ROOT=DATA_ROOT,
    split='train',          # or 'test'
    num_seq=1,
    num_frames=5,
    resolution=(512, 512),  # 可選，根據你想要的輸出解析度
)

# 建立 DataLoader
DL = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

# 印出一筆資料看看
for batch in DL:
    view = batch[0]  # 單一筆資料
    print("圖像 shape:", view['img'].shape)
    print("深度圖 shape:", view['depthmap'].shape)
    print("相機內參:", view['camera_intrinsics'])
    print("相機姿態:", view['camera_pose'])
    break

import os
import torch
import numpy as np
from tqdm import tqdm
from glob import glob
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.inference import inference
import open3d as o3d

# Create image pairs（每隔 step 張圖配成一組）
def make_sparse_pairs(images, step=20):
    return [(images[i], images[i + step]) for i in range(0, len(images) - step, step)]

# === Parameters ===
scenes_root = "7SCENES"
output_dir = "PLY/test"
model_ckpt = "checkpoints/dust3r_7scenes/checkpoint-best.pth"

# === Initialize model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AsymmetricCroCo3DStereo.from_pretrained(model_ckpt).to(device).eval()

# Output directory
os.makedirs(output_dir, exist_ok=True)

# Get each sequence directory (e.g., chess/test/seq-01)
scene_dirs = sorted(glob(os.path.join(scenes_root, "*", "test", "seq-*")))

# === Process each sequence ===
for seq_dir in tqdm(scene_dirs, desc="Processing Sequences"):
    
    # Current scene name and sequence name
    scene = os.path.basename(os.path.dirname(os.path.dirname(seq_dir)))
    seq = os.path.basename(seq_dir)
    
    # Get all RGB image paths in the sequence
    color_paths = sorted(glob(os.path.join(seq_dir, "*.color.png")))
    
    # Check if pose information exists (required to align to world coordinates)
    pose_path = os.path.join(seq_dir, "frame-000000.pose.txt")
    if not os.path.exists(pose_path):
        print(f"[Warning] Missing pose.txt for {seq_dir}, skipping")
        continue

    # Need at least two images for pairing
    if len(color_paths) < 2:
        print(f"[Warning] Skip {seq_dir}, not enough images")
        continue
    
    # Load images
    images = load_images(color_paths, size=640)
    # Create image pairs every 'step' frames
    pairs = make_sparse_pairs(images, step=20)

    all_pts3d, all_rgb = [], []

    # === Inference ===
    with torch.no_grad():
        output = inference(pairs, model, device=device)

    # === Extract 3D points and corresponding colors from model output ===
    for pts3d_tensor, img in zip(output['pred1']['pts3d'], output['view1']['img']):
        pts3d = pts3d_tensor.detach().cpu().numpy()
        
        # # If in [3, H, W] format, convert to [H, W, 3]
        if pts3d.shape[0] == 3:
            pts3d = np.moveaxis(pts3d, 0, -1)  # [3, H, W] -> [H, W, 3]

        # Convert image format to numpy array in [H, W, 3] format
        if isinstance(img, torch.Tensor):
            img_np = img.cpu().numpy()
        else:
            img_np = np.asarray(img)

        if img_np.shape[0] == 3:
            rgb = np.transpose(img_np, (1, 2, 0))  # -> [H, W, 3]
        else:
            rgb = img_np

        # Only keep valid values (non-NaN/Inf) positions
        valid = np.isfinite(pts3d).all(axis=2)
        all_pts3d.append(pts3d[valid])
        all_rgb.append((rgb[valid] * 255).astype(np.uint8))
        
    # If there are no valid points, skip this sequence
    if len(all_pts3d) == 0:
        print(f"[Warning] No valid points in {seq_dir}, skipping")
        continue
    
    # Merge point clouds and color information from all pairs
    pts3d = np.concatenate(all_pts3d, axis=0)
    rgb = np.concatenate(all_rgb, axis=0)

    # Use the pose from the first image to transform point cloud to world coordinates
    T0 = np.loadtxt(pose_path, dtype=np.float32)  # (4, 4)
    pts3d_homo = np.concatenate([pts3d, np.ones((pts3d.shape[0], 1), dtype=np.float32)], axis=1)
    pts3d = (T0 @ pts3d_homo.T).T[:, :3]
    
    # Create and process open3d point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts3d.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64) / 255.0)

    print(f"[{scene}-{seq}] Original point count: {len(pcd.points)}")

    # Downsample
    voxel_size = 7.5e-3
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    print(f"[{scene}-{seq}] Downsampled point count: {len(pcd_down.points)}")

    # Save as binary .ply
    ply_path = os.path.join(output_dir, f"{scene}-{seq}.ply")
    o3d.io.write_point_cloud(ply_path, pcd_down, write_ascii=False)



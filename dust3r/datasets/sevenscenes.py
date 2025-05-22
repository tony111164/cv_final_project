import os
import os.path as osp
import numpy as np
import cv2
from PIL import Image

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2


class SevenScenes(BaseStereoViewDataset):
    def __init__(self, *args, split, ROOT, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, split=split, **kwargs)
        self.scenes = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]
        self.pairs = self._load_data()

    def _load_data(self):
        # Automatically scan frame files under each sequence directory listed in the split file
        pairs = []
        
        for scene in self.scenes:
            scene_dir = osp.join(self.ROOT, scene) # e.g., "7SCENES/chess"
            split_file = osp.join(scene_dir, f"{self.split.capitalize()}Split.txt") # e.g., "7SCENES/chess/TrainSplit.txt"  

            with open(split_file, 'r') as f:
                sequence_dirs = [f"seq-{int(line.strip().replace('sequence', '')):02d}" for line in f.readlines()] # e.g., ["seq-01", "seq-02", ...]

            for seq in sequence_dirs: # e.g., "seq-01"
                seq_dir = osp.join(scene_dir, self.split, seq) # e.g., "7SCENES/chess/train/seq-01"
                
                # skip "sparse" sequences
                if not os.path.exists(seq_dir) or 'sparse' in seq_dir:
                    print(f"[Skip] Missing or unwanted folder: {seq_dir}")
                    continue

                # Get all color image paths in the sequence
                color_images = sorted([
                    fn[:-len('.color.png')]  # Remove the ".color.png" suffix
                    for fn in os.listdir(seq_dir)
                    if fn.endswith('.color.png') # e.g., "frame-000000.color.png"
                ]) 
                
                # pair images every 'offset' frames
                offset = 5
                for i in range(len(color_images) - offset):
                    frame1 = osp.join(seq, color_images[i])
                    frame2 = osp.join(seq, color_images[i + offset])
                    pairs.append({
                        'scene': scene,
                        'frame1': frame1,
                        'frame2': frame2
                    })

        # shuffle pairs
        if self.seed is not None:
            rng = np.random.RandomState(self.seed)
            rng.shuffle(pairs)
        else:
            np.random.shuffle(pairs)

        return pairs

    def __len__(self):
        return len(self.pairs)

    def get_stats(self):
        return f'{len(self)} pairs from {len(self.scenes)} scenes'

    def _get_views(self, pair_idx, resolution, rng):
        pair = self.pairs[pair_idx]
        scene = pair['scene']
        
        views = []
        
        for frame_key, view_idx in zip(['frame1', 'frame2'], [0, 1]):
            frame = pair[frame_key]

            # paths
            color_path = osp.join(self.ROOT, scene, self.split, f"{frame}.color.png") # e.g., "7SCENES/chess/train/seq-01/frame-000000.color.png"
            depth_path = osp.join(self.ROOT, scene, self.split, f"{frame}.depth.png") # e.g., "7SCENES/chess/train/seq-01/frame-000000.depth.png"
            pose_path = osp.join(self.ROOT, scene, self.split, f"{frame}.pose.txt") # e.g., "7SCENES/chess/train/seq-01/frame-000000.pose.txt"

            # Load image and depth
            image = imread_cv2(color_path)
            depthmap = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH).astype(np.float32) / 1000.0
            
            # camera_pose = np.loadtxt(pose_path, dtype=np.float32)
            
            # if osp.exists(pose_path):
            #     camera_pose = np.loadtxt(pose_path).astype(np.float32)
            # else:
            #     camera_pose = np.eye(4, dtype=np.float32)
            
            camera_pose = np.eye(4, dtype=np.float32)
            
            intrinsics = np.array([
                [585, 0, 320],
                [0, 585, 240],
                [0, 0, 1]
            ], dtype=np.float32)

            # Resize and crop if necessary
            image, depthmap, intrinsics = self._crop_resize_if_necessary(
                image, depthmap, intrinsics, resolution, rng, info=(scene, frame)
            )

            views.append(dict(
                img=image,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
                dataset='7-Scenes',
                label=scene,
                instance=frame
            ))
            
        return views

if __name__ == "__main__":
    # Test the dataset
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb
    import numpy as np
    
    dataset = SevenScenes(split='train', ROOT="7SCENES", resolution=(640, 480), aug_crop=16)
    print(f"Dataset contains {len(dataset)} pairs")
    
    for idx in np.random.permutation(len(dataset))[:3]:  # Show first 3 random pairs
        views = dataset[idx]
        assert len(views) == 2
        print(idx, view_name(views[0]), view_name(views[1]))
        viz = SceneViz()
        poses = [views[view_idx]['camera_pose'] for view_idx in [0, 1]]
        cam_size = max(auto_cam_size(poses), 0.001)
        for view_idx in [0, 1]:
            pts3d = views[view_idx]['pts3d']
            valid_mask = views[view_idx]['valid_mask']
            colors = rgb(views[view_idx]['img'])
            viz.add_pointcloud(pts3d, colors, valid_mask)
            viz.add_camera(pose_c2w=views[view_idx]['camera_pose'],
                           focal=views[view_idx]['camera_intrinsics'][0, 0],
                           color=(view_idx*255, (1 - view_idx)*255, 0),
                           image=colors,
                           cam_size=cam_size)
        viz.show()
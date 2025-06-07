import os
import shutil
import subprocess
from pathlib import Path
import zipfile

# === 1. 定義輸入資料夾與路徑 ===
demo_paths = [
    "./data/7scenes/chess/test/chess-seq-03",
    "./data/7scenes/fire/test/fire-seq-03",
    "./data/7scenes/heads/test/heads-seq-01",
    "./data/7scenes/office/test/office-seq-02",
    "./data/7scenes/office/test/office-seq-06",
    "./data/7scenes/office/test/office-seq-07",
    "./data/7scenes/office/test/office-seq-09",
    "./data/7scenes/pumpkin/test/pumpkin-seq-01",
    "./data/7scenes/redkitchen/test/redkitchen-seq-03",
    "./data/7scenes/redkitchen/test/redkitchen-seq-04",
    "./data/7scenes/redkitchen/test/redkitchen-seq-06",
    "./data/7scenes/redkitchen/test/redkitchen-seq-12",
    "./data/7scenes/redkitchen/test/redkitchen-seq-14",
    "./data/7scenes/stairs/test/stairs-seq-01",
]

# === 2. 定義輸出位置 ===
output_dir = Path("output/demo")
collect_dir = Path("test")  # 收集所有 .ply 的資料夾
collect_dir.mkdir(exist_ok=True)

# === 3. 執行 demo.py 並搬運 .ply ===
for demo_path in demo_paths:
    print(f"▶ Running demo on: {demo_path}")
    subprocess.run(["python", "demo.py", "--demo_path", demo_path, "--conf_thresh", "1e-3", "--kf_every", "10"], check=True)

    seq_name = Path(demo_path).name
    ply_path = output_dir / seq_name / f"{seq_name}.ply"
    
    if ply_path.exists():
        target_path = collect_dir / f"{seq_name}.ply"
        shutil.copy(ply_path, target_path)
        print(f"✅ Copied: {ply_path} -> {target_path}")
    else:
        print(f"⚠️ Warning: {ply_path} not found.")

# === 4. 壓縮 test 資料夾為 test.zip ===
zip_name = "test.zip"
print(f"📦 Zipping {collect_dir} to {zip_name}")
with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for file_path in collect_dir.glob("*.ply"):
        zipf.write(file_path, arcname=str(Path("test") / file_path.name))  # 保留 test/ 資料夾
print("✅ Done.")
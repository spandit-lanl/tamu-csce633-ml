from PIL import Image
import os

data_dir = "./data"  # Adjust if needed

for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith(".jpg"):
            path = os.path.join(root, file)
            try:
                with Image.open(path) as img:
                    if img.size == (228, 228):
                        print(f"224x224: {path}")
                    print(f" {path}, {img.size}")
            except Exception as e:
                print(f"Error reading {path}: {e}")

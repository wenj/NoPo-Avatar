import os
import imageio
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, default='datasets/HuGe100K/all/deepfashion')
args = parser.parse_args()

root_dir = args.root_dir


def load_images(example_path):
    """Load JPG images as raw bytes (do not decode)."""
    image_dir = str(example_path).replace('.mp4', '').replace('videos', 'images')
    if os.path.exists(image_dir):
        print(f'skip {example_path}')
        return

    input_video = imageio.get_reader(example_path)
    input_frames = [frame for frame in input_video]
    os.makedirs(image_dir, exist_ok=True)
    for i, frame in enumerate(input_frames):
        imageio.imwrite(f"{image_dir}/{i:06d}.png", frame)


for subdir in sorted(os.listdir(root_dir)):
    for video_name in tqdm(sorted(os.listdir(os.path.join(root_dir, subdir, 'videos')))):
        if video_name.endswith('.mp4'):
            load_images(os.path.join(root_dir, subdir, 'videos', video_name))
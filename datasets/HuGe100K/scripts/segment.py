from segment_anything import SamPredictor, sam_model_registry
import numpy as np
import os
import os.path as osp
import cv2
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, default='datasets/HuGe100K/all/deepfashion')
args = parser.parse_args()

root = args.root_dir

ckpt_path = 'pretrained_weights/sam_vit_h_4b8939.pth'
model_type = "vit_h"
assert osp.isfile(ckpt_path), 'Please download sam_vit_h_4b8939.pth'
sam = sam_model_registry[model_type](checkpoint=ckpt_path).cuda()
predictor = SamPredictor(sam)

# run SAM
for subdir in sorted(os.listdir(root)):
    root_dir = osp.join(root, subdir, 'images')
    save_dir = osp.join(root, subdir, 'masks')
    for video_name in tqdm(sorted(os.listdir(root_dir))):
        os.makedirs(os.path.join(save_dir, video_name), exist_ok=True)
        for img_name in sorted(os.listdir(os.path.join(root_dir, video_name))):
            img_path = os.path.join(root_dir, video_name, img_name)
            img = cv2.imread(img_path)

            ys, xs, _ = (img < 200).nonzero()
            y1, y2 = ys.min(), ys.max()
            x1, x2 = xs.min(), xs.max()
            bbox = np.array([x1, y1, x2, y2])

            # use keypoints as prompts
            img_input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            predictor.set_image(img_input)
            masks, scores, logits = predictor.predict(box=bbox[None, :],
                                                      multimask_output=False)
            # mask_input = logits[np.argmax(scores), :, :]
            # masks, _, _ = predictor.predict(box=bbox[None, :],
            #                                 multimask_output=False, mask_input=mask_input[None])
            mask = masks.sum(0) > 0
            mask = (mask * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(save_dir, video_name, img_name), mask)

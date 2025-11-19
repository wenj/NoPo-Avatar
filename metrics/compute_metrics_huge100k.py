import sys, glob
import imageio.v2 as imageio
import skimage.metrics
import numpy as np
import torch
import cv2
import os
from lpips import LPIPS
import shutil
import pdb



tmp_ours = './tmp/eval_huge100k/pred'
tmp_gt = './tmp/eval_huge100k/gt'
tmperr = './tmp/eval_huge100k/error'

if not os.path.exists(tmperr):
    os.makedirs(tmperr, exist_ok=True)

if not os.path.exists(tmp_ours):
    os.makedirs(tmp_ours, exist_ok=True)

if not os.path.exists(tmp_gt):
    os.makedirs(tmp_gt, exist_ok=True)


def mae(imageA, imageB):
    err = np.sum(np.abs(imageA.astype("float") - imageB.astype("float")))
    err /= float(imageA.shape[0] * imageA.shape[1] * imageA.shape[2])

    return err


###########################################

def mse(imageA, imageB):
    errImage = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2, 2)
    errImage = np.sqrt(errImage)

    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1] * imageA.shape[2])

    return err, errImage



root_dir = 'datasets/HuGe100K/all'
ours_dir = 'outputs/test/train_huge100k_inputs1_res1024'
# os.makedirs(save_dir, exist_ok=True)
# os.makedirs(os.path.join(save_dir, 'images'), exist_ok=True)
# os.makedirs(os.path.join(save_dir, 'motions'), exist_ok=True)

test_list = []
for filename in sorted(os.listdir('datasets/HuGe100K/splits')):
    if 'test' in filename:
        test_list.extend(np.load(os.path.join('datasets/HuGe100K/splits', filename), allow_pickle=True))


psnrs, ssims, mses, maes = [], [], [], []

for i, test_file in enumerate(test_list):
    image_path = test_file['video_path'][:-4].replace('videos', 'images')
    infos = test_file['video_path'][:-4].split('/')
    subset, subdir, subject = infos[-4], infos[-3], infos[-1]

    for i, frame_id in enumerate([19, 23, 3, 7, 11, 15]):
        gt_img = imageio.imread(os.path.join(root_dir, subset, subdir, 'images', subject, f'{frame_id:06d}.png')) / 255.
        mask_img = imageio.imread(os.path.join(root_dir, subset, subdir, 'masks', subject, f'{frame_id:06d}.png')) / 255.
        mask_img = mask_img[..., None]
        gt_img = np.array(gt_img) * mask_img + (1 - mask_img) * 1.
        t = gt_img.astype(np.float32)

        ours_img = imageio.imread(os.path.join(ours_dir, f'{subset}_{subdir}_{subject}', 'color', f'{frame_id:06d}.png')) / 255.
        g = ours_img.astype(np.float32)

        mseValue, errImg = mse(g, t)

        errImg = (errImg * 255.0).astype(np.uint8)
        errImg = cv2.applyColorMap(errImg, cv2.COLORMAP_JET)

        sample_name = f'{subset}_{subdir}_{subject}_{i:06d}'
        cv2.imwrite(os.path.join(tmperr, sample_name + '.png'), errImg)

        mseValue_ours_gt, errImg_ours_gt = mse(g, t)
        maeValue = mae(g, t)
        psnr = 10 * np.log10((1 ** 2) / mseValue_ours_gt)

        imageio.imsave("{}/{}_source.png".format(tmp_ours, sample_name),
                       (g * 255).astype('uint8'))  # ours

        imageio.imsave("{}/{}_target.png".format(tmp_gt, sample_name),
                       (t * 255).astype('uint8'))  # gt

        psnrs += [psnr]
        ssims += [skimage.metrics.structural_similarity(g, t, channel_axis=2, data_range=1)]
        # maes += [maeValue]
        mses += [mseValue]


psnrs = np.array(psnrs)
ssims = np.array(ssims)

# PSNR & SSIM
psnr = psnrs.mean()
print(f"PSNR mean {psnr}", flush=True)
ssim = ssims.mean()
print(f"SSIM mean {ssim}", flush=True)

###########################################
# LPIPS

lpips = LPIPS(net='alex', version='0.1')
if torch.cuda.is_available():
    lpips = lpips.cuda()

g_files = sorted(glob.glob(tmp_ours + '/*_source.png'))
t_files = sorted(glob.glob(tmp_gt + '/*_target.png'))

lpipses = []
for i in range(len(g_files)):

    g = imageio.imread(g_files[i]).astype('float32') / 255.
    t = imageio.imread(t_files[i]).astype('float32') / 255.
    g = 2 * torch.from_numpy(g).unsqueeze(-1).permute(3, 2, 0, 1) - 1
    t = 2 * torch.from_numpy(t).unsqueeze(-1).permute(3, 2, 0, 1) - 1
    if torch.cuda.is_available():
        g = g.cuda()
        t = t.cuda()
    lpipses += [lpips(g, t).item()]
lpips = np.mean(lpipses)
print(f"LPIPS Alex Mean {lpips}", flush=True)


###########

lpips = LPIPS(net='vgg', version='0.1')
if torch.cuda.is_available():
    lpips = lpips.cuda()

g_files = sorted(glob.glob(tmp_ours + '/*_source.png'))
t_files = sorted(glob.glob(tmp_gt + '/*_target.png'))

lpipses = []
for i in range(len(g_files)):
    g = imageio.imread(g_files[i]).astype('float32') / 255.
    t = imageio.imread(t_files[i]).astype('float32') / 255.
    g = 2 * torch.from_numpy(g).unsqueeze(-1).permute(3, 2, 0, 1) - 1
    t = 2 * torch.from_numpy(t).unsqueeze(-1).permute(3, 2, 0, 1) - 1
    if torch.cuda.is_available():
        g = g.cuda()
        t = t.cuda()
    lpipses += [lpips(g, t).item()]
lpips = np.mean(lpipses)
print(f"LPIPS VGG mean {lpips}", flush=True)


###########################################
# FID

os.system('python -m pytorch_fid --device cuda {} {}'.format(tmp_ours, tmp_gt))


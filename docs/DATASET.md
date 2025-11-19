# Data instruction

-----

To facilitate reproduction, we provide access to the processed dataset. You can acquire it only upon request. Before doing so, you must first obtain permission from the original provider. We list the procedure for obtaining each dataset here:

* __THuman2.0 / THuman2.1__: Fill out the [request form](https://github.com/ytrock/THuman2.0-Dataset/blob/main/THUman2.1_Agreement.pdf) and send it to Yebin Liu (liuyebin@mail.tsinghua.edu.cn) and cc Tao Yu (ytrock@126.com) to request the download link.  By requesting the link, you acknowledge that you have read the agreement, understand it, and agree to be bound by it. If you do not agree with these terms and conditions, you must not download and use the Dataset.
* __XHuman__: Follow [the instruction](https://xhumans.ait.ethz.ch/) to get the access to the XHuman dataset.
* __HuGe100K__: Follow [the instruction](https://yiyuzhuang.github.io/IDOL/) to get the access to the HuGe100K dataset.

We provide a rendered THuman2.0 / THuman2.1 following [GHG](https://github.com/humansensinglab/Generalizable-Human-Gaussians). After getting the access, please email Jing Wen (jw116@illinois.edu) with the title "Request for NoPo-Avatar's data". Please attach the screenshot of the comfirmation email from authors of the original datasets.

We also provide the processed XHuman dataset for evaluating novel view and novel pose synthesis. Please email Jing Wen (jw116@illinois.edu) with the title "Request for NoPo-Avatar's data" and attach the screenshot of the confirmation email.

Organize the original HuGe100K dataset and our rendered THuman2.0 / THuman2.1 following the structure:
```
├── $ROOT/datasets
    ├── thuman
        ├── train
        ├── val
        ├── thuman2.0_train.json
        ├── thuman2.0_val.json
        ├── thuman2.1_train.json
    ├── HuGe100K
        ├── all
            ├── deepfashion
                ├── images0
                    ├── images
                    ├── masks
                    ├── param
                    ├── videos
        ├── scripts
        ├── splits
```

## THuman2.0 / THuman2.1
Run the following command to generate data for _THuman2.0_:
```bash
python -m src.scripts.convert_thuman
```

Set `THuman21 = True` in Line 27 of src/scripts/convert_thuman to generate _THuman2.1_.

Please note that we also render the LBS weights for training (`RASTERIZE_LBS_WEIGHTS = True`) in the resolution 1024x1024. This step will takes much time and storage. Feel free to disable LBS rendering for evaluation only.

## HuGe100K
Put HuGe100K raw data under `datasets/HuGe100K/all`.

Setup SAM following [the instruction](https://github.com/facebookresearch/segment-anything) and download [the checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) for ViT-H. Put the checkpoint under `pretrained_weights/`

Run the following script to extract images from videos and segment the subjects, then prepare to the target format for training. Note than HuGe100K is a large dataset. Unpacking videos, segmenting and rendering LBS weights will cost time and storage. 
```bash
./datasets/HuGe100K/scripts/prepare_huge100k.sh
```

For reproduction, we provide our train/val/test splits using IDOL's codes in `datasets/HuGe100K/splits`.
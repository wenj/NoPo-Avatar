# NoPo-Avatar: Generalizable and Animatable Avatars from Sparse Inputs without Human Poses

NeurIPS 2025

[Paper]() | [Project Page](https://wenj.github.io/NoPo-Avatar/)

```bibtex
@inproceedings{wen2025nopoavatar,
    title={{NoPo-Avatar: Generalizable and Animatable Avatars from Sparse Inputs without Human Poses}},
    author={Jing Wen and Alex Schwing and Shenlong Wang},
    booktitle={NeurIPS},
    year={2025}
}
```

## Environment
```bash
conda create -n NoPo-Avatar python=3.10 -y
conda activate NoPo-Avatar
pip install -r requirements.txt
```


## Data preparation

Please refer to [DATASET.md](docs/DATASET.md) for data preparation.

## Inference
You can find the checkpoints [here](https://uofi.box.com/s/euwzfrfkuo3j5zpb4z2a6a3e6w37r2yj).

Download [SMPL-X models](https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=models_smplx_v1_1.zip) and put them under `datasets/smplx/`. Download [SMPLX's UV map](https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=smplx_uv.zip) and put them under `assets/templates`. 

and run
```bash
python src/scripts/generate_template.py
```

### Multiview inputs
We take three input views on THuman2.0 / THuman2.1.
```bash
python -m src.main \
        mode=test \
        dataset/view_sampler@dataset.thuman.view_sampler=evaluation \
        dataset.thuman.view_sampler.index_path=assets/evaluation_thuman_view3.json \
        dataset.thuman.background_color=[0,0,0] \
        checkpointing.load=checkpoint/thuman2.0_inputs3_res1024_iter50000.ckpt \
        test.save_image=true \
        test.save_compare=true \
        +experiment=train_thuman2.0_3views_res1024 \
        wandb.name=train_thuman2.0_3views_res1024 \
        test.output_path=outputs/test
```
Set `test.align_pose=true test.align_human_pose=target` to enable test-time pose optimization.

Checkpoints (Set `checkpointing.load`):
* `checkpoint/thuman2.0_inputs3_res1024_iter50000.ckpt`: trained on THuman2.0
* `checkpoint/thuman2.1_inputs3_res1024_iter50000.ckpt`: trained on THuman2.1 (~2000 more subjects)
* `checkpoint/thuman2.1_huge100k_inputs3_res1024_iter90000.ckpt`: trained on THuman2.1 + HuGe100K

Larger training datasets yield better generalization.

To evaluate, run
```bash
python metrics/compute_metrics_thuman.py
```

On XHuman, run
```bash
# Novel view synthesis
python -m src.main \
        mode=test \
        dataset/view_sampler@dataset.thuman.view_sampler=evaluation \
        dataset.thuman.view_sampler.index_path=assets/evaluation_xhuman.json \
        dataset.thuman.roots=[datasets/xhuman] \
        dataset.thuman.background_color=[0,0,0] \
        checkpointing.load=checkpoint/thuman2.1_huge100k_inputs3_res1024_iter90000.ckpt \
        test.save_image=true \
        test.save_compare=true \
        +experiment=train_thuman2.0_3views_res1024 \
        wandb.name=train_thuman2.0_3views_res1024_xhuman \
        test.output_path=outputs/test

# Novel pose synthesis
python -m src.main \
        mode=test \
        dataset/view_sampler@dataset.thuman.view_sampler=evaluation \
        dataset.thuman.view_sampler.index_path=assets/evaluation_xhuman_nps.json \
        dataset.thuman.roots=[datasets/xhuman_nps] \
        dataset.thuman.background_color=[0,0,0] \
        checkpointing.load=checkpoint/thuman2.1_huge100k_inputs3_res1024_iter90000.ckpt \
        test.save_image=true \
        test.save_compare=true \
        +experiment=train_thuman2.0_3views_res1024 \
        wandb.name=train_thuman2.0_3views_res1024_xhuman_nps \
        test.output_path=outputs/test
```

To evaluate, run
```bash
python metrics/compute_metrics_xhuman.py
python metrics/compute_metrics_xhuman_nps.py
```

### Single-view input

```bash
python -m src.main \
        mode=test \
        dataset/view_sampler@dataset.huge100k.view_sampler=evaluation \
        dataset.huge100k.view_sampler.index_path=assets/evaluation_huge100k_view1.json \
        dataset.huge100k.background_color=[1,1,1] \
        checkpointing.load=checkpoint/huge100k_inputs1_res896_iter80000.ckpt \
        test.save_image=true \
        test.save_compare=true \
        +experiment=train_huge100k_inputs1_res1024 \
        wandb.name=train_huge100k_inputs1_res1024 \
        test.output_path=outputs/test
```

To evaluate, run
```bash
python metrics/compute_metrics_huge100k.py
```

## Training

Download the [pretrained checkpoint](https://huggingface.co/botaoye/NoPoSplat/resolve/main/mixRe10kDl3dv_512x512.ckpt) from NoPoSplat and put it under `pretrained_weights/`.

Download [SMPL-X models](https://smpl-x.is.tue.mpg.de/download.php) and put them under `datasets/smplx/`.

### Multiview inputs
Train on THuman2.0 (or THuman2.1, change the dataset path for THuman2.1):
```bash
python -m src.main +experiment=train_thuman2.0_3views_res256
python -m src.main +experiment=train_thuman2.0_3views_res512 model.encoder.pretrained_weights=<PATH OF THE LAST .ckpt FROM LAST STEP>
python -m src.main +experiment=train_thuman2.0_3views_res1024 model.encoder.pretrained_weights=<PATH OF THE LAST .ckpt FROM LAST STEP>
```

Train on THuman2.0 (or THuman2.1) + HuGe100K:
```bash
python -m src.main +experiment=train_thuman2.1_huge100k_3views_res256
python -m src.main +experiment=train_thuman2.1_huge100k_3views_res512 model.encoder.pretrained_weights=<PATH OF THE LAST .ckpt FROM LAST STEP>
python -m src.main +experiment=train_thuman2.1_huge100k_3views_res1024 model.encoder.pretrained_weights=<PATH OF THE LAST .ckpt FROM LAST STEP>
```

### Single-view input
Train on HuGe100K:
```bash
python -m src.main +experiment=train_huge100k_inputs1_res256
python -m src.main +experiment=train_huge100k_inputs1_res512 model.encoder.pretrained_weights=<PATH OF THE LAST .ckpt FROM LAST STEP>
python -m src.main +experiment=train_huge100k_inputs1_res1024 model.encoder.pretrained_weights=<PATH OF THE LAST .ckpt FROM LAST STEP>
```

## Acknowledgement
This project builds on [GHG](https://github.com/humansensinglab/Generalizable-Human-Gaussians) and [NoPoSplat](https://github.com/cvg/NoPoSplat). We thank the authors for releasing codes.
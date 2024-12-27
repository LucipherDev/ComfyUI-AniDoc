# ComfyUI-AniDoc
ComfyUI Custom Nodes for ["AniDoc: Animation Creation Made Easier"](https://arxiv.org/abs/2412.14173). These nodes, adapted from [the official implementations](https://github.com/yihao-meng/AniDoc), enables automated line art video colorization using a novel model that aligns color information from references, ensures temporal consistency, and reduces manual effort in animation production.

## Installation

1. Navigate to your ComfyUI's custom_nodes directory:
```bash
cd ComfyUI/custom_nodes
```

2. Clone this repository:
```bash
git clone https://github.com/LucipherDev/ComfyUI-AniDoc
```

3. Install requirements:
```bash
cd ComfyUI-AniDoc
python install.py
```

### Or Install via ComfyUI Manager

****Custom nodes from [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) are required for these nodes to function properly.***

## Usage

**If you installed from the ComfyUI-Manager, all the necessary models should be automatically downloaded into the `models/diffusers` folder.**

**Otherwise they will be automatically downloaded when the LoadAniDoc node is used for the first time.**

Manually Download the [CoTracker Checkpoint](https://huggingface.co/facebook/cotracker/blob/main/cotracker2.pth) and place it in `models/cotracker` folder to use AniDoc with tracking enabled.

The nodes can be found in "AniDoc" category as "AniDocLoader", "LoadCoTracker", "GetAniDocControlnetImages", "AniDocSampler".

Take a look at the example workflow for more info.

> Currently our model expects `14 frames` video as input, so if you want to colorize your own lineart sequence, you should preprocess it into 14 frames

> However, in our test, we found that in most cases our model works well for more than 14 frames (`72 frames`)

## Citation

```bibtex
@article{meng2024anidoc,
      title={AniDoc: Animation Creation Made Easier},
      author={Yihao Meng and Hao Ouyang and Hanlin Wang and Qiuyu Wang and Wen Wang and Ka Leong Cheng and Zhiheng Liu and Yujun Shen and Huamin Qu},
      journal={arXiv preprint arXiv:2412.14173},
      year={2024}
}
```

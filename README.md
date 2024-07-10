# S2C
Official repository for CVPR 2024 Oral paper: "**From SAM to CAMs: Exploring Segment Anything Model for Weakly Supervised Semantic Segmentation**" by [Hyeokjun Kweon](https://scholar.google.com/citations?user=em3aymgAAAAJ&hl=en&oi=ao).

## Prerequisite
* Tested on Ubuntu 18.04, with Python 3.8, PyTorch 1.8.2, CUDA 11.4, 4 GPUs.
* [The PASCAL VOC 2012 development kit](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/):
You need to specify place VOC2012 under ./data folder.
* ImageNet-pretrained weights for resnet38d are from [[resnet_38d.params]](https://drive.google.com/file/d/1fpb4vah3e-Ynx4cv5upUcqnpJFY_FTja/view?usp=sharing). You need to place the weights as ./pretrained/resnet_38d.params.

# Prerequisite on SAM 
* Please install [SAM](https://github.com/facebookresearch/segment-anything) and download vit_h version as ./pretrained/sam_vit_h.pth
* Note that I slightly modified the original code of SAM for fast batch-wise inference during the training of CAMs.
* After installing SAM properly, you should substitute the files 'mask_decoder.py' and 'sam.py' in the segment_anything/modeling directory with the files in 'modeling' of this repository.
* Additionally, you need to run the Segment-Everything option using SAM as preprocessing. Please refer to get_se_map.py for further details.

## Usage
* This repository generates CAMs (seeds) to train the segmentation network.
* For further refinement, refer [RIB](https://github.com/jbeomlee93/RIB) and [SAM_WSSS](https://github.com/cskyl/SAM_WSSS).

### Training
* Please specify the name of your experiment.
* Training results are saved at ./experiment/[exp_name]
```
python train.py --name [exp_name] --model s2c
```

### Evaluation for CAM
```
python evaluation.py --name [exp_name] --task cam --dict_dir dict
```

## Citation
If our code be useful for you, please consider citing our CVPR 2024 paper using the following BibTeX entry.
```
@inproceedings{kweon2024sam,
  title={From SAM to CAMs: Exploring Segment Anything Model for Weakly Supervised Semantic Segmentation},
  author={Kweon, Hyeokjun and Yoon, Kuk-Jin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={19499--19509},
  year={2024}
}
```

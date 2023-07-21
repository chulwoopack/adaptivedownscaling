# Adaptive Downscaling

This repo is for developing a novel image downscaling method. This work is currently under review by International Journal International Journal on Document Analysis and Recognition (IJDAR).

## Table of Contents
- [Project Overview](#project-overview)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Contact](#contact)

## Project Overview
In this work, we propose a novel image downscaling approach that combines the strengths of both content-independent and content-aware strategies. The approach limits the sampling space per the content-independent strategy, adaptively relocating such sampled pixel points, and amplifying their intensities based on the local gradient and texture via the content-aware strategy. 

The overview of our approach is shown in the figure below:
![workflow](/assets/workflow.png)

Comparison between downscaled image by ours and conventional method is shown below:
![ds_results](/assets/ds_results.png)

Segmentation quality improvements gained by our downscaled images are shown below:
![seg_results](/assets/seg_results.png)

## Prerequisites
The required software systems and libraries are:
* Python >= 3.6
* numpy >= 1.21.6
* matplotlib >= 3.1.1
* opencv-python >= 4.0.1
* tqdm >= 4.31.1
* scikit-learn >= 0.20.3
* scikit-image >= 0.15.0

## Usage
```
python adaptive-downscaling-v9 --image_list <image_list.txt> --output_dir <output_dir> [--mode <uniform | adaptive>] [--opt <log|entropy|contrast|homogeneity|scharr|canny>]
```
- `--image_list`: Specifies the path to the input file, which should contain a list of image paths.
- `--output_dir`: Specifies the path to save output images.
- `--mode`      : Specifies the image downscaling mode either `uniform` or `adaptive` (`uniform` by default).
- `--opt`       : Specifies the mapping option among `log`, `entropy`, `contrast`, `homogeneity`, `scharr`, and `canny` (`log` by default).

## Contact
Please [email](chulwoo.pack@huskers.unl.edu) me if you have any question.

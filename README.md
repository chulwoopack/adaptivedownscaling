# Adaptive Downscaling

This repo is for developing a novel image downscaling method. This work is currently under review by International Journal International Journal on Document Analysis and Recognition (IJDAR).

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
In this work, we propose a novel image downscaling approach that combines the strengths of both content-independent and content-aware strategies. The approach limits the sampling space per the content-independent strategy, adaptively relocating such sampled pixel points, and amplifying their intensities based on the local gradient and texture via the content-aware strategy. 

The overview of our approach is shown in the figure below:
![workflow](/assets/workflow.png)

Comparison between downscaled image by ours and conventional method is shown below:
![ds_results](/assets/ds_results.png)

Segmentation quality improvements gained by our downscaled images are shown below:
![seg_results](/assets/seg_results.png)

## Installation
Please run `setup.py` to match the dependencies.

## Usage

```
python adaptive-downscaling-v9 --image_list [image_list.txt] --output_dir [output_dir]
```
- `--image_list`: Specifies the path to the input file, which should contain a list of image paths.
- `--output_dir`: Specifies the path to save output images.

## Contact
Please [email](chulwoo.pack@huskers.unl.edu) me if you have any question.

# pero-quality

This repository provides tools for quality assessment of digitalized documents . The estimated quality scores closely correnspond to readability by humans. The tools provide quality score heatmaps and an overall quality score for a whole document page. The module computes local perceptual qulity scores based on confidenc scores  from Optical Character Recognition (OCR). More detailed description is provided below.

![](images/image0.jpg) ![](images/image2.jpg)

This module is build on top of OCR developed in project [PERO]([https://pero.fit.vutbr.cz/](https://pero.fit.vutbr.cz/) ([pero-ocr]([https://github.com/DCGM/pero-ocr](https://github.com/DCGM/pero-ocr)). The text recognition works in multiple stages. Firstly, locations and heights of text lines are determined using a fully convolutional neural network (modified U-NET).  The individual text lines are processed by covolutional-recurrent networks trained using CTC loss. These networks provide confidences of recognized characters which are locally mapped to perceptual scores. The mapping to perceptual scores was calibrated on a large dataset of readability ratings by human readers. 

## Install

This module requires python 3.

```bash
git clone https://github.com/DCGM/pero-quality.git
cd pero-quality
pip install -r requirements.txt
```

**Note**: One of the requirements is `tensorflow-gpu`. If you don't have CUDA capable GPU installed in your machine, you can instead use `tensorflow` module to run the computation on CPU. If you have a CUDA capable GPU, but don't have CUDA drivers installed , refer to instructions for [tensorflow GPU support](https://www.tensorflow.org/install/gpu).

## Usage

Before processing a document, you need to download configuration and models for line detection and character recongnition. This is done by launching `models/download_models.py` script, or using the link located below.

All quality assessment functionality is encapsulated in `QualityEvaluator` class  located in `quality_evaluator` module. Working with the module is shown below:

```python
import cv2
from quality_evaluator import QualityEvaluator
quality = QualityEvaluator(config_path)
image = cv2.imread(filepath, cv2.IMREAD_COLOR)
score, heatmap = quality.evaluate_image(image)
```

The output `score` holds scalar quality value of the whole page and `heatmap` is a numpy array of local quality at the input image resolution. More verbose example is located in `pero_quality` module.

**Note**: pero and pero_quality modules must be in `PYTHONPATH` to use `QualityEvaluator` class:

```
export PYTHONPATH=/abs/path/pero-quality/pero:/abs/path/pero-quality/pero_quality:$PYTHONPATH
```

To process multiple images from command line and render local quality overlays, launch the module as: 

```bash
python3 pero-quality.py config/path input/path -o output/path -v
```

Both input path and output path can be directories, in which case all images in the input directory are processed. Alternatively single input and output file can be specified. Additional info about parameters is printed using single `-h`  argument.

## Models

| [Lidove noviny_2019-12-16](http://www.fit.vutbr.cz/~ihradis/pero-models/ocr_quality_LN_2019-12-16.zip) | This model is optimized for low-quality scans of  czech newspapers scanned from microfilms. E.g.  [Lidove noviny](http://www.digitalniknihovna.cz/mzk/periodical/uuid:bdc405b0-e5f9-11dc-bfb2-000d606f5dc6). |
| ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| more will be added                                                                                     |                                                                                                                                                                                                              |

## Docker support

If you don't want to install this package and all dependencies on your system, you can use a provided dockerfile, which will create a separate image and docker container for this module.

`docker` directory contains basic info about how to run this repository in docker. We provide a dockerfile, and scripts for building an image and running docker container.

### Prerequisites

- [docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/#install-docker-engine---community-)

- [nvidia docker](https://github.com/NVIDIA/nvidia-docker) support

### Usage

Note: use sudo with docker commands if user is not in `docker` group

1. build image: `build.sh`

2. create and run container: `run.sh`

3. run `pero-quality.py` or use `QualityEvaluator` as needed

# pero-quality

This repository contains scripts used for digitalized document quality  assessment. The score is evaluated using confidences of neural network for text recognition.

![](images/image0.jpg) ![](images/image2.jpg)

## Usage

- pero and pero_quality modules must be in `PYTHONPATH` to use `QualityEvaluator` class
  
  ```
  export PYTHONPATH=/abs/path/pero-quality/pero:/abs/path/pero-quality/pero_quality:$PYTHONPATH
  ```

Before processing a document, you need to download configuration and models needed for line detection and character recongnition. This is done by launching `models/download_models.py` script.



The class encapsulating quality assessment is located in `quality_evaluator` module. Working with the module is shown below:

```python
quality = QualityEvaluator(config)
image = cv2.imread(filepath, cv2.IMREAD_COLOR)
score, heatmap = quality.evaluate_image(image)
```

More verbose example is located in `pero_quality` module. To process images and save them highlighted as shown above, launch the module as: 

```bash
python3 pero-quality.py config/path input/path -o output/path -v
```

Both input path and output path both must be directory paths or file paths.

## Configuration

Basic configuration for quality assessment is downloaded with the models as described in the previous section. The parameters are tuned for images with size approx. 6000x4000 pixels. 

## Models

| ocr_quality_LN_2019-12-16 | [http://www.fit.vutbr.cz/~ihradis/pero-models/ocr_quality_LN_2019-12-16.zip](http://www.fit.vutbr.cz/~ihradis/pero-models/ocr_quality_LN_2019-12-16.zip) |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| more will be added        |                                                                                                                                                          |

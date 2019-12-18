# pero-quality

Tools for digitalized document quality  assessment.

![](images/image0.jpg) ![](images/image2.jpg)

## Usage

- pero and pero_quality modules_ must be in `PYTHONPATH` to use `QualityEvaluator` class
  
  ```
  export PYTHONPATH=/abs/path/pero-quality/pero:/abs/path/pero-quality/pero_quality:$PYTHONPATH
  ```
  
  

Before processing a document, you need to download configuration and models needed for line detection and character recongnition. This is done by launching `models/download_models.py` script.

The class encapsulating quality assessment is located in `quality_evaluator` module. Example, how the class is used, is located in `pero_quality` module. To process images and save them highlighted as shown above, launch the module as: 

```bash
python3 pero-quality.py config/path input/path -o output/path -v
```

Both input path and output path both must be directory paths or file paths.

## Configuration

Basic configuration for quality assessment is downloaded with the models as described in the previous section. The parameters are tuned for images with size approx. 6000x4000 pixels. 

#!/usr/bin/env python3

import cv2
import numpy as np
import os

from configparser import ConfigParser
from os.path import isfile, isabs, dirname, join, realpath
from sys import stderr
from typing import Dict


try:
    from heatmap import locate_confidences, load_encoding, compute_heatmap
    from pero.document_ocr import PageParser, PageLayout
except ModuleNotFoundError as e:
    print("Make sure \"pero\" directory is in PYTHONPATH.", file=stderr)
    raise e


class QualityEvaluator:
    """
    Class which computes image quality and outputs a heatmap.
    """

    def __init__(self, config_path: str):
        # suppress tensorflow warnings on loading models
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        assert isfile(config_path), f"Config file not found. Given path: \"{config_path}\""

        self.config = ConfigParser()
        self.config.optionxform = str
        self.config.read(config_path)

        for section, key in [['LINE_PARSER', 'MODEL_PATH'], ['OCR', 'OCR_JSON']]:
            if not isabs(self.config[section][key]):
                self.config[section][key] = realpath(join(dirname(config_path), self.config[section][key]))

        self.page_parser = PageParser(self.config)
        self.encoding = load_encoding(self.config['OCR']['OCR_JSON'])

    def _evaluate_image_with_layout(self, image: np.ndarray, page_layout: PageLayout) \
            -> (float, Dict, PageLayout):
        """
        :param image: loaded image as numpy array
        :param page_layout:
        :return:
            float: global image score
            Dict[(l, t, r, b) -> float]: dictionary containing score for each evaluated crop with its coordinates
        """
        page_layout = self.page_parser.process_page(image, page_layout)
        confidences_dict = locate_confidences(page_layout, self.encoding)

        img_score, heatmaps_scores = compute_heatmap(confidences_dict, image.shape[:2])

        return img_score, heatmaps_scores

    def evaluate_image(self, image: np.ndarray) -> (float, np.ndarray):
        """
        Compute quality and heatmap for given image.

        :param image: loaded image as numpy array
        :return:
            float: global image score in interval <0, 1>
            np.ndarray: heatmap with each pixel having value <0, 1> for score
        """

        h, w = image.shape[:2]
        page_layout = PageLayout(id="ID", page_size=(h, w))

        img_score, heatmap_scores = self._evaluate_image_with_layout(image, page_layout)

        # map scores to crops
        heatmap = np.zeros((h, w), dtype=np.float)

        for crop, score in heatmap_scores.items():
            l, t, r, b = crop
            heatmap[t:b + 1, l:r + 1] = score

        return img_score, heatmap

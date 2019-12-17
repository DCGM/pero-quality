#!/usr/bin/env python3

import cv2
import numpy as np

from argparse import ArgumentParser, Namespace
from os import listdir
from os.path import isdir, isfile, join, dirname, realpath
from sys import path, stderr
from typing import Dict, Tuple

dir_path = realpath(dirname(__file__))
path.append(join(dir_path, "pero"))

from heatmap import confidence_to_rgb
from quality_evaluator import QualityEvaluator


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("config", help="Path to OCR configuration file.")
    parser.add_argument("input", help="Path to image or folder.")
    parser.add_argument("-o", "--output", help="Path to image or folder.")
    parser.add_argument("-v", "--visual-heatmap", help="Colored heatmap created.", action="store_true")

    args = parser.parse_args()

    assert isdir(args.input) == isdir(args.output), "Both paths must be a file or a directory."
    return args


def draw_heatmap(image: np.ndarray, heatmap_scores: Dict[Tuple, float]) -> np.ndarray:
    heatmap = image.copy()

    for crop, score in heatmap_scores.items():
        l, t, r, b = crop
        heatmap[t:b + 1, l:r + 1] = confidence_to_rgb(score)

    alpha = 0.5
    return cv2.addWeighted(heatmap, alpha, image, 1 - alpha, 0, image)


def main():
    args = get_args()

    if isdir(args.input):
        files = [join(args.input, file) for file in listdir(args.input)]
    else:
        files = [args.input]

    evaluator = QualityEvaluator(args.config)

    for file in files:
        filename = file.split("/")[-1]
        image = cv2.imread(file)

        if image is None:
            print(f"File {filename} can't be read.", file=stderr)
            continue

        score, heatmap_scores = evaluator.evaluate_image(image)
        print(f"{filename}: {score}")

        if args.visual_heatmap:
            # draw colored heatmap
            heatmap = draw_heatmap(image, heatmap_scores)
        else:
            # map scores to crops
            heatmap = np.zeros_like(image)

            for crop, score in heatmap_scores.items():
                l, t, r, b = crop

                # interpolate <0,1> between 0-255
                heatmap[t:b + 1, l:r + 1] = score * 255

        if args.output is not None:
            # directory path given
            if isdir(args.output):
                cv2.imwrite(join(args.output, filename), heatmap)

            # file path given
            else:
                cv2.imwrite(args.output, heatmap)


if __name__ == "__main__":
    main()

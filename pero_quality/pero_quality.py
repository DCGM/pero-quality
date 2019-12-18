#!/usr/bin/env python3

import cv2
import numpy as np

from argparse import ArgumentParser, Namespace
from os import listdir
from os.path import isdir, isfile, join, dirname, realpath, basename
from sys import path, stderr

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


def create_colour_lut():
    """ Lookup table for effective converting of scores to colors for heatmap

    :return: numpy array with shape (256, 3)
    """
    lut = [_ for _ in range(256)]

    for i in range(256):
        if i / 255 < 0.1:
            # too low scores mean no detection
            lut[i] = (255, 255, 255)
        else:
            lut[i] = confidence_to_rgb(i / 255)

    return np.array(lut)


def draw_heatmap(image: np.ndarray, heatmap_scores: np.ndarray, colour_lut: np.ndarray) -> np.ndarray:
    """ Draw colored heatmap on image showing text quality.

    :param image: image which is processed
    :param heatmap_scores: score for every pixel
    :param colour_lut: lookup table for effective mapping scores to colors
    :return: heatmap
    """

    # change values to <0, 255> interval
    heatmap = (heatmap_scores * 255).astype(np.int)

    colors = np.full_like(image, 255)

    # compute color for every channel using lookup table
    for channel in range(3):
        colors[:, :, channel] = colour_lut[:, channel][heatmap[:, :]]

    alpha = 0.5
    return cv2.addWeighted(colors, alpha, image, 1 - alpha, 0, image)


def main():
    args = get_args()

    if isdir(args.input):
        files = [join(args.input, file) for file in listdir(args.input)]
    elif isfile(args.input):
        files = [args.input]
    else:
        raise FileNotFoundError("Input file not found.")

    evaluator = QualityEvaluator(args.config)
    colour_lut = create_colour_lut()

    for file in files:
        filename = basename(file)
        image = cv2.imread(file, cv2.IMREAD_COLOR)

        if image is None:
            print(f"File {filename} can't be read.", file=stderr)
            continue

        score, heatmap_scores = evaluator.evaluate_image(image)
        print(f"{filename}: {score}")

        if args.visual_heatmap:
            # draw colored heatmap
            heatmap = draw_heatmap(image, heatmap_scores, colour_lut)
        else:
            heatmap = heatmap_scores * 255

        if args.output is not None:
            # directory path given
            if isdir(args.output):
                cv2.imwrite(join(args.output, filename), heatmap)

            # file path given
            else:
                cv2.imwrite(args.output, heatmap)


if __name__ == "__main__":
    main()

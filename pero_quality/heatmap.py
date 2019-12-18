#!/usr/bin/env python3

import cv2
import numpy as np
import pickle
import sys

from argparse import ArgumentParser, Namespace
from os import listdir
from os.path import isdir, dirname, realpath, join
from os import mkdir
from scipy.special import softmax
from sys import path
from typing import List, Tuple, Dict, Optional, Iterable
from xml.etree import ElementTree

# TODO: remove when pero-ocr becomes downloadable package
dir_path = realpath(dirname(dirname(__file__)))
path.append(join(dir_path, "pero"))

from pero.document_ocr import PageLayout
from pero.force_alignment import force_align
from pero.ocr_engine.postprocess import narrow_label
from pero.char_confidences import greedy_filtration


class Coords:
    """
    For cleaner storing of coordinates.
    """
    def __init__(self, l: int, t: int, r: int, b: int):
        self.l, self.t, self.r, self.b = map(lambda x: int(x), [l, t, r, b])

    def __repr__(self):
        return f"(l:{self.l} t:{self.t} r:{self.r} b:{self.b})"

    def tuple(self) -> Tuple[int, int, int, int]:
        return self.l, self.t, self.r, self.b


def get_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("input_dir", help="Path to folder containing images.")
    parser.add_argument("encoding", help="Path to OCR engine .json file containing character table.")
    parser.add_argument("maps", help="Output directory.")
    parser.add_argument("ocr_data", help="Directory containing OCR output")

    parser.add_argument("output_file", help="File where image scores will be stored.")
    parser.add_argument("-c", "--confidence-threshold", type=float, default=0.,
                        help="Line is skipped if average confidence is smaller than given value.")
    parser.add_argument("-s", "--show", action="store_true", help="Show output image.")

    args = parser.parse_args()

    assert 0 <= args.confidence_threshold <= 1, "Confidence threshold not in interval <0, 1>."

    if not isdir(args.maps):
        try:
            mkdir(args.maps)
        except FileNotFoundError:
            ex_type, ex_inst, ex_tb = sys.exc_info()
            raise ex_type.with_traceback(
                FileNotFoundError("Directory for storing heatmaps can't be found nor created."),
                ex_tb)
        except FileExistsError:
            ex_type, ex_inst, ex_tb = sys.exc_info()
            raise ex_type.with_traceback(
                FileExistsError("Heatmap directory can't be created. File with that name already exists."),
                ex_tb)

    return args


def load_encoding(path: str) -> List[str]:
    """
    Load character table from OCR engine configuration
    :param path: Path to OCR engine config file.
    :return: array containing characters from encoding
    """
    import json
    with open(path, "r") as f:
        engine = json.load(f)
    return engine['characters']


def get_lines_coords(page_path) -> Dict[str, Coords]:
    """
    Parse line coordinates from page.xml file (output of OCR in
        page directory).

    :param page_path: Path to xml file.
    :return: Dictionary: line_id -> Coords object - coordinates of line on page
    """

    lines = ElementTree.parse(page_path).getroot()[0][0][1:]
    lines_coords = {}

    for line in lines:

        # points = lt rt rb lb
        line_id = line.attrib['id']
        points = line[0].attrib['points'].split(" ")

        # possible problem if there is odd number if points (forced to take the latter one)
        # skewed lines are not implemented
        l, t = points[0].split(",")
        r, b = points[(len(points)+1)//2].split(",")

        lines_coords[line_id] = Coords(l, t, r, b)

    return lines_coords


def confidence_to_rgb(confidence: float) -> Tuple[int, int, int]:
    """
    Interpolate between red and green based on
    confidence parameter <0, 1>.
    
    :param confidence: factor for choosing color between red and green
    :return: Tuple containing BGR code for color
    """

    if confidence <= 0.5:
        color_amount = int(confidence * 2 * 255)
        return 0, color_amount, 255
    else:
        color_amount = int((confidence - 0.5) * 2 * 255)
        return 0, 255, 255 - color_amount


def get_confidence_alignment(line_logits: np.ndarray, confidence_threshold: float, encoding: List[str]) -> Tuple[List, np.ndarray]:
    """
    Compute text from logits and align it to frames.

    :return: Tuple
        narrowly aligned logits (only peaks are left)
        probabilities of each character in text
    """
    blank_char_index = len(encoding)

    # set zeroes to "big" negative values
    line_logits[line_logits == 0] = -80

    line_probs = softmax(line_logits, axis=1)

    # transcribe probabilities to text
    text, text_probs = greedy_filtration(line_probs, encoding)

    # empty line
    if text == "" or np.average(text_probs) < confidence_threshold:
        return [], np.array([])

    # map letters from list to indices from encoding array
    text_indices = [encoding.index(c) for c in list(text)]

    # this is where the magic happens
    forced_alignment = force_align(-np.log(line_probs), text_indices, blank_char_index)

    # make single letter out of repeating letters in multiple frames
    narrow_forced_alignment = narrow_label(forced_alignment, line_probs, blank_char_index, False)

    return text_probs, narrow_forced_alignment


def locate_confidences(page_layout: PageLayout, encoding: List[str], crop_coords: Optional[Coords] = None,
                       confidence_threshold: float = 0.) -> \
        Dict[Tuple[int, int], float]:
    """
    Create dictionary containing detected letters with their confidences.

    :param page_layout: object encapsulating regions and lines
    :param crop_coords: coordinates of crop in original image
    :param encoding: character encoding table from OCR config
    :param confidence_threshold: parameter for filtering lines by average confidence
    :return: dictionary mapping coordinates of each heatmap to it's corresponding confidence
    """
    heatmap_dict: Dict[Tuple[int, int], float] = {}

    for region in page_layout.regions:
        for line_idx, line in enumerate(region.lines):
            # get line coords
            t, l = line.polygon[0]
            b, r = line.polygon[(len(line.polygon) + 1) // 2]
            line_coords = Coords(l, t, r, b)

            if crop_coords is not None and (line_coords.t > crop_coords.b or line_coords.b < crop_coords.t):
                # line is not located in specified crop
                continue

            blank_char_index = len(encoding)
            text_probs, narrow_alignment = get_confidence_alignment(line.logits.toarray(),
                                                                    confidence_threshold, encoding)

            # in case of empty line
            if len(text_probs) == 0:
                continue

            # pixels_per_frame =  size of line (in pixels) / frame count
            frame_size = (line_coords.r - line_coords.l) / (len(narrow_alignment))

            # starting pixel from where we're iterating
            frame_coord = line_coords.l

            # text probabilities are over each letter (without blanks)
            confidence_iter = iter(text_probs)

            for frame_idx, character_idx in enumerate(narrow_alignment):
                if character_idx != blank_char_index:

                    try:
                        confidence = next(confidence_iter)
                    except StopIteration:
                        ex_type, ex_inst, ex_tb = sys.exc_info()
                        raise ex_type.with_traceback(
                            StopIteration("Iterating over more valid characters than expected. "
                                          "Confidence iterator is empty."),
                            ex_tb)

                    if crop_coords is not None and frame_coord < crop_coords.l:
                        # before crop -> skip to next frame
                        frame_coord += frame_size
                        continue
                    if crop_coords is not None and frame_coord > crop_coords.r:
                        # after crop -> end
                        break

                    # compute center coordinates of frame
                    center_x = int(2 * frame_coord + frame_size) // 2
                    center_y = (line_coords.b + line_coords.t) // 2

                    # heatmap is stored only if its center is inside given crop
                    if crop_coords is None or (crop_coords.l < center_x < crop_coords.r and
                                               crop_coords.t < center_y < crop_coords.b):
                        heatmap_dict[(center_x, center_y)] = confidence
                frame_coord += frame_size

    return heatmap_dict


def compute_global_score(scores: List) -> float:
    return np.average(scores)


def compute_local_score(heatmap: Iterable, denom=5):
    values = np.array(sorted(heatmap))

    if len(values) == 0:
        return 0

    values = values[:len(values)//denom]

    p = 6
    return ((values ** p).sum()) ** (1/p)


def compute_heatmap(heatmap: Dict, img_size: Tuple, crop_size=200, min_logit_per_crop=5) \
        -> Tuple[float, Dict[Tuple, float]]:
    """
    Compute perceptual score from OCR confidences of each letter detection.

    :param heatmap: dictionary contaning letter detections with their confidences
    :param img_size: (h, w) format
    :param crop_size: size of each crop which will be evaluated
    :param min_logit_per_crop: if crop contains less logits than given value, the crop is not evaluated
    :return:
        float: score of whole image
        Dict[(l, t, r, b) -> float]: score of every crop
    """

    if len(heatmap) == 0:
        return 0, {}

    h, w = img_size

    crops: List[Coords] = []
    crop_scores: Dict[Tuple, float] = {}

    for y in range(0, h - (h % crop_size) + crop_size, crop_size):
        for x in range(0, w - (w % crop_size + crop_size), crop_size):
            crops.append(Coords(x, y, x + crop_size - 1, y + crop_size - 1))

    for crop in crops:
        # filter heatmap to contain only scores from given crop
        submap = {k: v for k, v in heatmap.items() if crop.l <= k[0] <= crop.r and crop.t <= k[1] <= crop.b}

        if len(submap) < min_logit_per_crop:
            continue

        score = np.clip(compute_local_score(submap.values()), 0, 1)
        crop_scores[crop.tuple()] = score

    global_score = np.clip(compute_global_score(list(crop_scores.values())), 0, 1)

    return global_score, crop_scores


def main(args: Namespace):
    score_file = open(args.output_file, "w+")
    encoding = load_encoding(args.encoding)

    file_cnt = len(listdir(args.input_dir))

    for file_counter, filepath in enumerate(listdir(args.input_dir)):
        img_name_prefix = filepath.split(".")[0]
        img_path = args.input_dir + "/" + filepath

        page_image: np.ndarray = cv2.imread(img_path)
        assert page_image is not None, f"Image with path \"{img_path}\" can't be loaded."

        h, w = page_image.shape[:2]

        page_path = f"{args.ocr_data}/pages/{img_name_prefix}.xml"
        page_layout = PageLayout(file=page_path)

        logits_path = f"{args.ocr_data}/logits/{img_name_prefix}.logits"
        page_layout.load_logits(logits_path)

        heatmap_dict = locate_confidences(page_layout, encoding)
        score, submaps_overlay, heatmaps_scores = compute_heatmap(heatmap_dict, page_image.copy(), (h, w))

        score_file.write(f"{img_name_prefix}: {score:.3f}\n")

        if len(heatmap_dict) == 0:
            continue

        alpha = 0.5
        cv2.addWeighted(submaps_overlay, alpha, page_image, 1 - alpha, 0, page_image)

        if args.show:
            cv2.imshow(f"colored_img.jpg", page_image)
            cv2.waitKey()

        output_path = f"{args.maps}/{img_name_prefix}"
        cv2.imwrite(f"{output_path}.jpg", page_image)

        # save pickled heatmap
        with open(f"{output_path}.pkl", "wb") as f:
            pickle.dump(heatmap_dict, f)

        print(f"Extracted heatmaps: {file_counter + 1}/{file_cnt}", flush=True, end="\r")

    score_file.close()


if __name__ == "__main__":
    args = get_args()
    main(args)

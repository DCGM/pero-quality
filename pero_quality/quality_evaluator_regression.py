#!/usr/bin/env python3

import cv2
import numpy as np
import os

from configparser import ConfigParser
from os.path import isfile, isabs, dirname, join, realpath
from sys import stderr
from typing import Dict, Optional

import tensorflow as tf
from tensorflow.python.platform import gfile

try:
    from heatmap import divide_chunks
except ModuleNotFoundError as e:
    print("Make sure \"pero\" directory is in PYTHONPATH.", file=stderr)
    raise e

class QualityEvaluatorRegression:
    """
    Class which computes image quality and outputs a heatmap.
    """
    def __init__(self, config_path, config):
        # suppress tensorflow warnings on loading models
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        self.config = config

        for section in ['REGRESSION', 'SEGMENTATION']:
            if not isabs(self.config[section]['PATH']):
                self.config[section]['PATH'] = realpath(join(dirname(config_path), self.config[section]['PATH']))

        self.size = int(self.config['REGRESSION']['SIZE'])

        #regression model
        with tf.gfile.GFile(self.config['REGRESSION']['PATH'], "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def)

        self.sess_reg = tf.Session(graph=graph)

        #segmentation model        
        with tf.gfile.GFile(self.config['SEGMENTATION']['PATH'], "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def)

        self.sess_seg = tf.Session(graph=graph)

    def _parse_image(self, image, out_map):
        """
        :param image: loaded image as numpy array
        :param out_map: map from segmentation
        :return:
            list: crops
            list: coordinates of crops
            list: mask for skiping of crops
        """
        batch = []
        shape = []
        skip = []
        row = True
        column = True
        row_end = self.size
        column_end = self.size

        while row:
            column = True
            column_end = self.size
            while column:
                crop = image[column_end - self.size:column_end, row_end - self.size:row_end]
                crop_ = out_map[column_end - self.size:column_end, row_end - self.size:row_end]

                if np.shape(crop)[0] != self.size:
                    column = False
                elif np.shape(crop)[1] != self.size:
                    row = False
                    column = False
                else:
                    seg_text = np.mean(crop_)
                    if seg_text < 0.33:
                        skip.append(True)
                    else:
                        skip.append(False)
                    batch.append(crop)
                    shape.append((int((column_end-self.size)/self.size), int((row_end-self.size)/self.size)))
                column_end += self.size
            row_end += self.size

        return batch, shape, skip

    def _compute_map(self, preds, shape, image, skip):
        """
        :param preds: predictions from regression network
        :param shape: coordinates of crops
        :param image: loaded image as numpy array
        :param skip: mask for skiping of crops
        :return:
            np.ndarray: heatmap with each pixel having value <0, 1> for score
        """
        matrix = np.zeros((shape[-1][0]+1, shape[-1][1]+1))
        skip_matrix = np.zeros((shape[-1][0]+1, shape[-1][1]+1))
        for i, item in enumerate(shape):
            matrix[item[0]][item[1]] = preds[i]
            skip_matrix[item[0]][item[1]] = skip[i]

        heatmap = np.zeros(shape=(np.shape(image)[0], np.shape(image)[1]))
        for y in range(np.shape(matrix)[0]):
            for x in range(np.shape(matrix)[1]):
                if not skip_matrix[y][x]:
                    heatmap[y*self.size:y*self.size+self.size, x*self.size:x*self.size+self.size] = matrix[y][x]

        return heatmap

    def evaluate_image(self, image):
        """
        Compute quality and heatmap for given image.

        :param image: loaded image as numpy array
        :return:
            float: global image score in interval <0, 1>
            np.ndarray: heatmap with each pixel having value <0, 1> for score
        """        
        resized = cv2.resize(image, (0,0), fx=1/4, fy=1/4)
        
        new_shape_x = resized.shape[0]
        new_shape_y = resized.shape[1]
        while not new_shape_x % 8 == 0:
            new_shape_x += 1
        while not new_shape_y % 8 == 0:
            new_shape_y += 1
        test_img_canvas = np.zeros((1, new_shape_x, new_shape_y, 3))
        test_img_canvas[0, :resized.shape[0], :resized.shape[1], :] = resized/256.

        self.sess_seg.graph.as_default()
        out_map = self.sess_seg.run(self.config['SEGMENTATION']['OUTPUT'], feed_dict={self.config['SEGMENTATION']['INPUT']: test_img_canvas})
        out_map = cv2.resize(cv2.cvtColor(out_map[0], cv2.COLOR_BGR2GRAY), (np.shape(image)[1], np.shape(image)[0]))

        batch, shape, skip = self._parse_image(image, out_map)

        self.sess_reg.graph.as_default()
        batches = list(divide_chunks(batch, int(self.config['REGRESSION']['BATCH_SIZE'])))

        output = self.sess_reg.graph.get_tensor_by_name(self.config['REGRESSION']['OUTPUT'])
        preds = []
        for i, item in enumerate(batches):
            temp = self.sess_reg.run(output, {self.config['REGRESSION']['INPUT']: np.array(item)})
            if len(preds) == 0:
                preds = temp
            else:
                preds = np.concatenate((preds, temp))

        img_score = np.average(np.delete(preds, np.where(np.array(skip) == 1.0)))
        heatmap = self._compute_map(preds, shape, image, skip)

        return img_score, heatmap

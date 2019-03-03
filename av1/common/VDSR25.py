## Copyright (c) 2019, Alliance for Open Media. All rights reserved
##
## This source code is subject to the terms of the BSD 2 Clause License and
## the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
## was not distributed with this source code in the LICENSE file, you can
## obtain it at www.aomedia.org/license/software. If the Alliance for Open
## Media Patent License 1.0 was not distributed with this source code in the
## PATENTS file, you can obtain it at www.aomedia.org/license/patent.
##

import numpy as np
import tensorflow as tf


def model25(input_tensor):
#VDSR25
#    with tf.device("/gpu:0"):
        weights = []
        tensor = None
        convId = 0

        conv_00_w = tf.get_variable("conv_%02d_w" % (convId), [3,3,1,64], initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/9)))
        conv_00_b = tf.get_variable("conv_%02d_b" % (convId), [64], initializer=tf.constant_initializer(0))
        convId += 1
        weights.append(conv_00_w)
        weights.append(conv_00_b)
        tensor = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input_tensor, conv_00_w, strides=[1,1,1,1], padding='SAME'), conv_00_b))

        for i in range(23):
            conv_w = tf.get_variable("conv_%02d_w" % (convId), [3,3,64,64], initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/9/64)))
            conv_b = tf.get_variable("conv_%02d_b" % (convId), [64], initializer=tf.constant_initializer(0))
            convId += 1
            weights.append(conv_w)
            weights.append(conv_b)
            tensor = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(tensor, conv_w, strides=[1,1,1,1], padding='SAME'), conv_b))

        conv_w = tf.get_variable("conv_%02d_w" % (convId), [3,3,64,1], initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/9/64)))
        conv_b = tf.get_variable("conv_%02d_b" % (convId), [1], initializer=tf.constant_initializer(0))
        convId += 1
        weights.append(conv_w)
        weights.append(conv_b)
        tensor = tf.nn.bias_add(tf.nn.conv2d(tensor, conv_w, strides=[1,1,1,1], padding='SAME'), conv_b)

        tensor = tf.add(tensor, input_tensor)

        return tensor, weights

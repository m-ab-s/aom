## Copyright (c) 2019, Alliance for Open Media. All rights reserved
##
## This source code is subject to the terms of the BSD 2 Clause License and
## the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
## was not distributed with this source code in the LICENSE file, you can
## obtain it at www.aomedia.org/license/software. If the Alliance for Open
## Media Patent License 1.0 was not distributed with this source code in the
## PATENTS file, you can obtain it at www.aomedia.org/license/patent.
##

import tensorflow as tf
import numpy as np

def resblock(temp_tensor, convId, weights):
	out_tensor = None
	skip_tensor = None
	conv_secondID = 0

	skip_tensor = temp_tensor

	# Conv, 1x1, filters=192 ,+ ReLU
	conv_w = tf.get_variable("conv_%02d_%02d_w" % (convId, conv_secondID), [1, 1, 32, 192],
							 initializer=tf.contrib.layers.xavier_initializer())
	conv_b = tf.get_variable("conv_%02d_%02d_b" % (convId, conv_secondID), [192], initializer=tf.constant_initializer(0))
	weights.append(conv_w)
	weights.append(conv_b)
	out_tensor = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(temp_tensor, conv_w, strides=[1, 1, 1, 1], padding='SAME'), conv_b))
	conv_secondID += 1

	# Conv, 1x1, filters=25
	conv_w = tf.get_variable("conv_%02d_%02d_w" % (convId, conv_secondID), [1, 1, 192, 25],
							 initializer=tf.contrib.layers.xavier_initializer())
	conv_b = tf.get_variable("conv_%02d_%02d_b" % (convId, conv_secondID), [25], initializer=tf.constant_initializer(0))
	weights.append(conv_w)
	weights.append(conv_b)
	out_tensor = tf.nn.bias_add(tf.nn.conv2d(out_tensor, conv_w, strides=[1, 1, 1, 1], padding='SAME'), conv_b)
	conv_secondID += 1

	# Conv, 3x3, filters=32
	conv_w = tf.get_variable("conv_%02d_%02d_w" % (convId, conv_secondID), [3, 3, 25, 32],
							 initializer=tf.contrib.layers.xavier_initializer())
	conv_b = tf.get_variable("conv_%02d_%02d_b" % (convId, conv_secondID), [32], initializer=tf.constant_initializer(0))
	weights.append(conv_w)
	weights.append(conv_b)
	out_tensor = tf.nn.bias_add(tf.nn.conv2d(out_tensor, conv_w, strides=[1, 1, 1, 1], padding='SAME'), conv_b)
	conv_secondID += 1

	#skip + out_tensor
	out_tensor = tf.add(skip_tensor, out_tensor)

	return out_tensor

def model8(input_tensor):
	weights = []
	tensor = None
	convId = 0

	'''conv_00_w = tf.get_variable("conv_%02d_w" % (convId), [3, 3, 1, 32],
								initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9)))'''
	'''conv_00_w = tf.get_variable("conv_%02d_w" % (convId), [3, 3, 1, 32],
								initializer=tf.random_uniform_initializer(minval=-1 * np.sqrt(6.0 / 9), maxval = np.sqrt(6.0 / 9)))'''
	conv_00_w = tf.get_variable("conv_%02d_w" % (convId), [3, 3, 1, 32],
								initializer=tf.contrib.layers.xavier_initializer())
	conv_00_b = tf.get_variable("conv_%02d_b" % (convId), [32], initializer=tf.constant_initializer(0))
	weights.append(conv_00_w)
	weights.append(conv_00_b)
	tensor = tf.nn.bias_add(tf.nn.conv2d(input_tensor, conv_00_w, strides=[1, 1, 1, 1], padding='SAME'), conv_00_b)
	convId += 1

	#Residual Block x 8
	for i in range(8):
		tensor = resblock(tensor, convId, weights)
		convId += 1

	conv_w = tf.get_variable("conv_%02d_w" % (convId), [3, 3, 32, 1],
								initializer=tf.contrib.layers.xavier_initializer())
	conv_b = tf.get_variable("conv_%02d_b" % (convId), [1], initializer=tf.constant_initializer(0))
	weights.append(conv_w)
	weights.append(conv_b)
	tensor = tf.nn.bias_add(tf.nn.conv2d(tensor, conv_w, strides=[1, 1, 1, 1], padding='SAME'), conv_b)

	tensor = tf.add(tensor, input_tensor)

	return tensor, weights

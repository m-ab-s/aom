/*
 * Copyright (c) 2019, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <assert.h>
#include <math.h>
#include <stdio.h>

#include "third_party/googletest/src/googletest/include/gtest/gtest.h"

#include "av1/common/cnn.h"
#include "config/av1_rtcd.h"

#define SQR(x) ((x) * (x))
#define FLOAT_TOL 1E-6
#define INT_TOL 0

namespace {

class CNNTest : public ::testing::Test {
 protected:
  static void RunCNNTest(int image_width, int image_height, float *input,
                         float *expected, CNN_CONFIG cnn_config, int in_stride,
                         double tolerance, int use_rounding) {
    int out_width, out_height;
    av1_find_cnn_output_size(image_width, image_height, &cnn_config, &out_width,
                             &out_height);
    const int out_size = out_width * out_height;
    float *output = (float *)aom_malloc(sizeof(*output) * out_size);
    const int out_stride = out_width;

    av1_cnn_predict((const float **)&input, image_width, image_height,
                    in_stride, &cnn_config, &output, out_stride);

    if (use_rounding) {
      for (int i = 0; i < out_size; ++i) {
        output[i] = roundf(output[i]);
      }
    }

    double mse = 0;
    for (int i = 0; i < out_size; ++i) {
      EXPECT_LE(fabsf(expected[i] - output[i]), 1)
          << i << ": " << expected[i] << "/" << output[i] << std::endl;
      mse += SQR(expected[i] - output[i]);
    }
    mse /= out_size;
    EXPECT_LE(mse, tolerance);

    aom_free(output);
  }

  static void AssignLayerWeightsBiases(CNN_CONFIG *cnn_config, float *weights,
                                       float *bias) {
    size_t weight_offset = 0;
    size_t bias_offset = 0;
    for (int layer = 0; layer < cnn_config->num_layers; ++layer) {
      CNN_LAYER_CONFIG *layer_config = &cnn_config->layer_config[layer];
      layer_config->weights = weights + weight_offset;
      layer_config->bias = bias + bias_offset;
      weight_offset += layer_config->filter_width *
                       layer_config->filter_height * layer_config->in_channels *
                       layer_config->out_channels;
      bias_offset += layer_config->out_channels;

      ASSERT_NE(layer_config->weights, nullptr);
      ASSERT_NE(layer_config->bias, nullptr);
    }
  }
};

}  // namespace

TEST_F(CNNTest, TestNonActivationSingleLayerSingleKernel) {
  int image_width = 8;
  int image_height = 8;
  float input[] = {
    166, 125, 167, 37,  25,  16,  104, 136, 163, 37,  89,  162, 164,
    177, 108, 182, 46,  119, 64,  144, 100, 33,  157, 113, 183, 129,
    121, 42,  177, 221, 241, 150, 9,   70,  0,   96,  189, 208, 205,
    48,  117, 142, 202, 39,  35,  152, 100, 41,  225, 65,  213, 147,
    201, 18,  18,  191, 153, 23,  109, 134, 158, 12,  99,  120,
  };
  float expected_same[] = {
    1182, 2221, 827,  1890, 2635, 3076, 2994, 1266, 451,  3041, 2935,
    3128, 2427, 2040, 2463, 1491, 1578, 3932, 3122, 2774, 3169, 5148,
    5426, 2396, 941,  867,  1973, 2382, 3967, 5163, 3118, 1953, 1504,
    3536, 3523, 4372, 2400, 3542, 2882, 1897, 2148, 3290, 2486, 3594,
    4152, 3589, 1686, 2171, 912,  3931, 1896, 4044, 2188, 1767, 2998,
    1382, -667, 2881, 1009, 2496, -16,  1847, 1442, -902,
  };
  float expected_valid[] = {
    3041, 2935, 3128, 2427, 2040, 2463, 3932, 3122, 2774, 3169, 5148, 5426,
    867,  1973, 2382, 3967, 5163, 3118, 3536, 3523, 4372, 2400, 3542, 2882,
    3290, 2486, 3594, 4152, 3589, 1686, 3931, 1896, 4044, 2188, 1767, 2998,
  };
  float weights[] = { 7, -3, 5, -1, -3, 6, 8, 5, 3 };
  float bias[] = { 4 };

  CNN_CONFIG cnn_config = { .num_layers = 1,
                            .is_residue = 0,
                            .ext_width = 0,
                            .ext_height = 0,
                            .strict_bounds = 0,
                            { {
                                .deconvolve = 0,
                                .in_channels = 1,
                                .filter_width = 3,
                                .filter_height = 3,
                                .out_channels = 1,
                                .skip_width = 1,
                                .skip_height = 1,
                                .maxpool = 0,
                                .weights = weights,
                                .bias = bias,
                                .pad = PADDING_SAME_ZERO,
                                .activation = NONE,
                                .input_copy = 0,
                                .skip_combine = SKIP_NONE,
                            } } };

  RunCNNTest(image_width, image_height, input, expected_same, cnn_config,
             image_width, INT_TOL, 1);

  // Change padding to valid
  cnn_config.layer_config[0].pad = PADDING_VALID;

  RunCNNTest(image_width, image_height, input, expected_valid, cnn_config,
             image_width, INT_TOL, 1);
}

TEST_F(CNNTest, TestDeconvolveNonActivationSingleLayerSingleKernel) {
  int image_width = 4;
  int image_height = 4;
  float input[] = {
    160, 80, 118, 226, 117, 171, 134, 77, 120, 117, 76, 137, 175, 187, 56, 137,
  };
  float expected_same[] = {
    -475, -475,  -235,  -547, -349,  -1017, -673,  -221, 1784, 162,  1897,
    930,  1196,  887,   1820, -134,  -666,  -1436, -988, -866, -793, -575,
    -914, -1202, 1547,  602,  1835,  427,   1611,  15,   1716, 733,  -589,
    -826, -922,  -1003, -833, -1135, -828,  -517,  1950, 730,  1741, 1003,
    503,  169,   1886,  553,  -760,  -1284, -1030, -839, -549, -705, -832,
    -817, 1055,  -146,  2002, -444,  1276,  111,   1107, -406,
  };
  float expected_valid[] = {
    -315, 1125, 965,   -235,  329,   431,  379,   997,  1587,  -1125,
    -635, -475, -475,  -235,  -547,  -349, -1017, -673, -221,  5,
    91,   1784, 162,   1897,  930,   1196, 887,   1820, -134,  750,
    -143, -666, -1436, -988,  -866,  -793, -575,  -914, -1202, -447,
    -1,   1547, 602,   1835,  427,   1611, 15,    1716, 733,   -295,
    -241, -589, -826,  -922,  -1003, -833, -1135, -828, -517,  -149,
    -105, 1950, 730,   1741,  1003,  503,  169,   1886, 553,   5,
    -455, -760, -1284, -1030, -839,  -549, -705,  -832, -817,  -269,
    355,  1055, -146,  2002,  -444,  1276, 111,   1107, -406,  690,
    355,  -345, -496,  -719,  -818,  -481, -1,    -381, -680,  -269,
  };
  float weights[] = { -2, 7, 7, -5, -4, -3, -1, 0, 2, 6, -3, 5, 2, -2, -5, -2 };
  float bias[] = { 5 };

  CNN_CONFIG cnn_config = { .num_layers = 1,
                            .is_residue = 0,
                            .ext_width = 0,
                            .ext_height = 0,
                            .strict_bounds = 0,
                            { {
                                .deconvolve = 1,
                                .in_channels = 1,
                                .filter_width = 4,
                                .filter_height = 4,
                                .out_channels = 1,
                                .skip_width = 2,
                                .skip_height = 2,
                                .maxpool = 0,
                                .weights = weights,
                                .bias = bias,
                                .pad = PADDING_SAME_ZERO,
                                .activation = NONE,
                                .input_copy = 0,
                                .skip_combine = SKIP_NONE,
                            } } };

  RunCNNTest(image_width, image_height, input, expected_same, cnn_config,
             image_width, INT_TOL, 1);

  // Change padding to valid
  cnn_config.layer_config[0].pad = PADDING_VALID;

  RunCNNTest(image_width, image_height, input, expected_valid, cnn_config,
             image_width, INT_TOL, 1);
}

TEST_F(CNNTest, TestRELUMultiLayerMultiKernel) {
  int image_width = 8;
  int image_height = 8;
  float input[] = { 1, 8, 2, 2, 4, 8, 1, 8, 3, 3, 7, 1, 3, 3, 2, 6,
                    3, 6, 0, 6, 2, 4, 9, 2, 8, 2, 0, 4, 8, 3, 9, 3,
                    2, 7, 1, 7, 6, 0, 2, 5, 2, 7, 0, 7, 0, 5, 5, 8,
                    7, 8, 4, 5, 1, 5, 6, 6, 8, 5, 5, 1, 2, 9, 3, 9 };
  float expected[] = {
    1377431, 2173407, 2435745, 2471195, 2626654, 2734721, 2482994, 1513223,
    2152462, 3496400, 3977867, 4146647, 4441683, 4586838, 4090693, 2476698,
    2473040, 4021092, 4676039, 4978473, 5348027, 5489855, 4786816, 2901849,
    2605592, 4290798, 5007352, 5291078, 5588990, 5626708, 4904796, 2983677,
    2849105, 4608427, 5275136, 5340961, 5559243, 5600541, 5035205, 3090147,
    3059302, 4828189, 5325228, 5101868, 5277427, 5383493, 5012109, 3098909,
    2773077, 4309552, 4577133, 4273240, 4465622, 4670977, 4454622, 2768211,
    1651264, 2588284, 2694330, 2500518, 2627716, 2758369, 2646960, 1649032,
  };
  float weights[] = {
    7, 0, 4, 1, 2, 0, 4, 6, 6, 0, 9, 2, 9, 2, 0, 2, 4, 5, 4, 8, 4, 8, 9, 2,
    7, 5, 8, 9, 2, 8, 8, 3, 8, 8, 9, 1, 9, 8, 8, 8, 0, 3, 3, 5, 2, 4, 0, 7,
    5, 8, 9, 8, 7, 2, 5, 8, 6, 2, 8, 6, 8, 6, 1, 3, 4, 2, 0, 4, 3, 9, 9, 8,
    5, 9, 2, 4, 9, 7, 6, 5, 9, 6, 6, 4, 9, 2, 7, 6, 0, 8, 5, 7, 9, 6, 6, 5,
    5, 2, 4, 1, 5, 3, 6, 5, 8, 6, 6, 9, 8, 9, 9, 4, 1, 7, 5, 5, 8, 0, 8, 3,
    3, 0, 6, 3, 7, 2, 5, 1, 9, 7, 0, 3, 7, 0, 6, 0, 3, 5, 7, 2, 5, 5, 7, 9,
    2, 1, 5, 5, 3, 9, 6, 2, 4, 9, 7, 6, 2, 3, 3, 2, 1, 3, 2, 8, 0, 4, 7, 2,
    2, 6, 9, 0, 9, 8, 9, 8, 4, 1, 4, 3, 8, 2, 7, 1, 0, 7, 1, 7, 8, 3, 2, 3,
    9, 0, 5, 4, 4, 4, 8, 5, 7, 5, 9, 1, 1, 6, 1, 6, 2, 8, 8, 9, 2, 1, 4, 6,
  };
  float bias[] = {
    9, 6, 6, 7, 9, 1, 2, 9, 5,
  };

  CNN_CONFIG cnn_config = { .num_layers = 3,
                            .is_residue = 0,
                            .ext_width = 0,
                            .ext_height = 0,
                            .strict_bounds = 0,
                            {
                                {
                                    .deconvolve = 0,
                                    .in_channels = 1,
                                    .filter_width = 3,
                                    .filter_height = 3,
                                    .out_channels = 4,
                                    .skip_width = 1,
                                    .skip_height = 1,
                                    .maxpool = 0,
                                    .weights = NULL,
                                    .bias = NULL,
                                    .pad = PADDING_SAME_ZERO,
                                    .activation = RELU,
                                    .input_copy = 0,
                                    .skip_combine = SKIP_NONE,
                                },
                                {
                                    .deconvolve = 0,
                                    .in_channels = 4,
                                    .filter_width = 3,
                                    .filter_height = 3,
                                    .out_channels = 4,
                                    .skip_width = 1,
                                    .skip_height = 1,
                                    .maxpool = 0,
                                    .weights = NULL,
                                    .bias = NULL,
                                    .pad = PADDING_SAME_ZERO,
                                    .activation = RELU,
                                    .input_copy = 0,
                                    .skip_combine = SKIP_NONE,
                                },
                                {
                                    .deconvolve = 0,
                                    .in_channels = 4,
                                    .filter_width = 3,
                                    .filter_height = 3,
                                    .out_channels = 1,
                                    .skip_width = 1,
                                    .skip_height = 1,
                                    .maxpool = 0,
                                    .weights = NULL,
                                    .bias = NULL,
                                    .pad = PADDING_SAME_ZERO,
                                    .activation = RELU,
                                    .input_copy = 0,
                                    .skip_combine = SKIP_NONE,
                                },
                            } };

  // Weights and biases need to be specified separately because
  // of the offset.
  AssignLayerWeightsBiases(&cnn_config, weights, bias);

  RunCNNTest(image_width, image_height, input, expected, cnn_config,
             image_width, INT_TOL, 1);
}

TEST_F(CNNTest, TestSoftsignMultiLayerMultiKernel) {
  int image_width = 8;
  int image_height = 8;
  float input[] = { 0.517f, 0.505f, 0.769f, 0.537f, 0.55f,  0.264f, 0.991f,
                    0.282f, 0.87f,  0.63f,  0.165f, 0.463f, 0.075f, 0.46f,
                    0.098f, 0.954f, 0.592f, 0.439f, 0.389f, 0.316f, 0.921f,
                    0.551f, 0.815f, 0.512f, 0.784f, 0.65f,  0.417f, 0.472f,
                    0.509f, 0.258f, 0.631f, 0.235f, 0.353f, 0.541f, 0.538f,
                    0.148f, 0.683f, 0.957f, 0.294f, 0.269f, 0.15f,  0.773f,
                    0.404f, 0.279f, 0.076f, 0.693f, 0.536f, 0.055f, 0.868f,
                    0.605f, 0.288f, 0.024f, 0.424f, 0.924f, 0.476f, 0.031f,
                    0.728f, 0.972f, 0.543f, 0.701f, 0.56f,  0.726f, 0.37f,
                    0.046f };
  float expected[] = {
    0.864f, 0.91f,  0.911f, 0.911f, 0.911f, 0.911f, 0.91f,  0.871f,
    0.915f, 0.939f, 0.94f,  0.94f,  0.94f,  0.94f,  0.938f, 0.902f,
    0.916f, 0.94f,  0.94f,  0.94f,  0.94f,  0.94f,  0.939f, 0.904f,
    0.916f, 0.94f,  0.941f, 0.941f, 0.941f, 0.941f, 0.939f, 0.903f,
    0.916f, 0.94f,  0.941f, 0.941f, 0.941f, 0.94f,  0.939f, 0.903f,
    0.916f, 0.94f,  0.94f,  0.94f,  0.941f, 0.94f,  0.939f, 0.903f,
    0.915f, 0.939f, 0.94f,  0.94f,  0.94f,  0.939f, 0.938f, 0.901f,
    0.878f, 0.904f, 0.904f, 0.904f, 0.904f, 0.904f, 0.902f, 0.846f,
  };
  float weights[] = {
    0.44f,  0.863f, 0.551f, 0.281f, 0.727f, 0.97f,  0.48f,  0.751f, 0.976f,
    0.836f, 0.067f, 0.486f, 0.015f, 0.06f,  0.189f, 0.674f, 0.617f, 0.359f,
    0.251f, 0.262f, 0.245f, 0.369f, 0.369f, 0.689f, 0.195f, 0.079f, 0.357f,
    0.086f, 0.873f, 0.339f, 0.878f, 0.507f, 0.547f, 0.054f, 0.097f, 0.085f,
    0.617f, 0.159f, 0.639f, 0.946f, 0.103f, 0.958f, 0.423f, 0.349f, 0.131f,
    0.149f, 0.29f,  0.782f, 0.513f, 0.523f, 0.229f, 0.638f, 0.939f, 0.245f,
    0.942f, 0.421f, 0.683f, 0.642f, 0.937f, 0.193f, 0.559f, 0.962f, 0.413f,
    0.421f, 0.052f, 0.414f, 0.398f, 0.196f, 0.2f,   0.76f,  0.645f, 0.893f,
    0.201f, 0.584f, 0.901f, 0.009f, 0.664f, 0.749f, 0.979f, 0.303f, 0.409f,
    0.972f, 0.483f, 0.375f, 0.021f, 0.798f, 0.728f, 0.881f, 0.298f, 0.51f,
    0.167f, 0.257f, 0.212f, 0.342f, 0.458f, 0.284f, 0.187f, 0.733f, 0.164f,
    0.358f, 0.247f, 0.403f, 0.829f, 0.816f, 0.294f, 0.446f, 0.64f,  0.791f,
    0.926f, 0.064f, 0.28f,  0.087f, 0.83f,  0.069f, 0.656f, 0.082f, 0.985f,
    0.845f, 0.117f, 0.487f, 0.436f, 0.767f, 0.43f,  0.524f, 0.259f, 0.735f,
    0.295f, 0.698f, 0.765f, 0.595f, 0.783f, 0.715f, 0.226f, 0.314f, 0.373f,
    0.398f, 0.819f, 0.506f, 0.718f, 0.529f, 0.622f, 0.762f, 0.375f, 0.081f,
    0.257f, 0.159f, 0.32f,  0.706f, 0.021f, 0.707f, 0.683f, 0.921f, 0.785f,
    0.372f, 0.034f, 0.424f, 0.375f, 0.413f, 0.623f, 0.375f, 0.582f, 0.33f,
    0.186f, 0.356f, 0.688f, 0.967f, 0.782f, 0.707f, 0.818f, 0.134f, 0.757f,
    0.148f, 0.409f, 0.908f, 0.675f, 0.861f, 0.313f, 0.861f, 0.926f, 0.572f,
    0.14f,  0.103f, 0.249f, 0.542f, 0.479f, 0.191f, 0.528f, 0.486f, 0.54f,
    0.728f, 0.936f, 0.883f, 0.152f, 0.237f, 0.65f,  0.335f, 0.372f, 0.109f,
    0.971f, 0.705f, 0.398f, 0.028f, 0.315f, 0.206f, 0.742f, 0.466f, 0.618f,
    0.943f, 0.314f, 0.346f, 0.465f, 0.104f, 0.962f, 0.1f,   0.831f, 0.793f,
  };
  float bias[] = { 0.988f, 0.336f, 0.038f, 0.06f, 0.001f,
                   0.391f, 0.519f, 0.689f, 0.1f };

  CNN_CONFIG cnn_config = { .num_layers = 3,
                            .is_residue = 0,
                            .ext_width = 0,
                            .ext_height = 0,
                            .strict_bounds = 0,
                            {
                                {
                                    .deconvolve = 0,
                                    .in_channels = 1,
                                    .filter_width = 3,
                                    .filter_height = 3,
                                    .out_channels = 4,
                                    .skip_width = 1,
                                    .skip_height = 1,
                                    .maxpool = 0,
                                    .weights = 0,
                                    .bias = 0,
                                    .pad = PADDING_SAME_ZERO,
                                    .activation = SOFTSIGN,
                                    .input_copy = 0,
                                    .skip_combine = SKIP_NONE,
                                },
                                {
                                    .deconvolve = 0,
                                    .in_channels = 4,
                                    .filter_width = 3,
                                    .filter_height = 3,
                                    .out_channels = 4,
                                    .skip_width = 1,
                                    .skip_height = 1,
                                    .maxpool = 0,
                                    .weights = NULL,
                                    .bias = NULL,
                                    .pad = PADDING_SAME_ZERO,
                                    .activation = SOFTSIGN,
                                    .input_copy = 0,
                                    .skip_combine = SKIP_NONE,
                                },
                                {
                                    .deconvolve = 0,
                                    .in_channels = 4,
                                    .filter_width = 3,
                                    .filter_height = 3,
                                    .out_channels = 1,
                                    .skip_width = 1,
                                    .skip_height = 1,
                                    .maxpool = 0,
                                    .weights = NULL,
                                    .bias = NULL,
                                    .pad = PADDING_SAME_ZERO,
                                    .activation = SOFTSIGN,
                                    .input_copy = 0,
                                    .skip_combine = SKIP_NONE,
                                },
                            } };

  // Weights and biases need to be specified separately because
  // of the offset.
  AssignLayerWeightsBiases(&cnn_config, weights, bias);

  RunCNNTest(image_width, image_height, input, expected, cnn_config,
             image_width, FLOAT_TOL, 0);
}

TEST_F(CNNTest, TestSingleLayerEvenStrideEvenDimImage) {
  int image_width = 4;
  int image_height = 4;
  int stride = 2;
  float input[] = {
    -4, 4, -2, 8, -5, 9, 4, -2, 8, 7, 8, 4, -1, -5, 2, 6,
  };
  float expected_horizontal[] = {
    57, -42, 20, 14, -85, -52, -23, -60,
  };
  float expected_vertical[] = {
    13, 20, -81, 14, 7, -23, -21, -60,
  };
  float expected_ver_hor[] = {
    20,
    14,
    -23,
    -60,
  };
  float weights[] = {
    -4, 2, -3, -5, -4, -2, -2, 5, -1,
  };
  float bias[] = { -2 };

  CNN_CONFIG cnn_config = { .num_layers = 1,
                            .is_residue = 0,
                            .ext_width = 0,
                            .ext_height = 0,
                            .strict_bounds = 0,
                            {
                                {
                                    .deconvolve = 0,
                                    .in_channels = 1,
                                    .filter_width = 3,
                                    .filter_height = 3,
                                    .out_channels = 1,
                                    .skip_width = stride,
                                    .skip_height = 1,
                                    .maxpool = 0,
                                    .weights = weights,
                                    .bias = bias,
                                    .pad = PADDING_SAME_ZERO,
                                    .activation = NONE,
                                    .input_copy = 0,
                                    .skip_combine = SKIP_NONE,
                                },
                            } };

  RunCNNTest(image_width, image_height, input, expected_horizontal, cnn_config,
             image_width, INT_TOL, 0);

  cnn_config.layer_config[0].skip_width = 1;
  cnn_config.layer_config[0].skip_height = stride;

  RunCNNTest(image_width, image_height, input, expected_vertical, cnn_config,
             image_width, INT_TOL, 0);

  cnn_config.layer_config[0].skip_width = stride;
  cnn_config.layer_config[0].skip_height = stride;

  RunCNNTest(image_width, image_height, input, expected_ver_hor, cnn_config,
             image_width, INT_TOL, 0);
}

TEST_F(CNNTest, TestSingleLayerEvenStrideOddDimImage) {
  int image_width = 5;
  int image_height = 5;
  int stride = 2;
  float input[] = {
    8, 9, 0,  5, 9,  7,  -4, 7, 0, 1, 2, 9,  6,
    9, 4, -5, 8, -2, -1, 2,  5, 3, 6, 8, -5,
  };
  float expected_horizontal[] = {
    1, -27, -59, -27, 6, 42, -136, -42, -50, 54, -15, -43, 2, -109, -5,
  };
  float expected_vertical[] = {
    1, -99, -27, -16, -59, -136, -39, -42, -142, -50, 2, -45, -109, -31, -5,
  };
  float expected_ver_hor[] = {
    1, -27, -59, -136, -42, -50, 2, -109, -5,
  };
  float weights[] = {
    -5, -1, 4, -4, -5, -3, 6, 7, -5,
  };
  float bias[] = { -1 };

  CNN_CONFIG cnn_config = { .num_layers = 1,
                            .is_residue = 0,
                            .ext_width = 0,
                            .ext_height = 0,
                            .strict_bounds = 0,
                            {
                                {
                                    .deconvolve = 0,
                                    .in_channels = 1,
                                    .filter_width = 3,
                                    .filter_height = 3,
                                    .out_channels = 1,
                                    .skip_width = stride,
                                    .skip_height = 1,
                                    .maxpool = 0,
                                    .weights = weights,
                                    .bias = bias,
                                    .pad = PADDING_SAME_ZERO,
                                    .activation = NONE,
                                    .input_copy = 0,
                                    .skip_combine = SKIP_NONE,
                                },
                            } };

  RunCNNTest(image_width, image_height, input, expected_horizontal, cnn_config,
             image_width, INT_TOL, 0);

  cnn_config.layer_config[0].skip_width = 1;
  cnn_config.layer_config[0].skip_height = stride;

  RunCNNTest(image_width, image_height, input, expected_vertical, cnn_config,
             image_width, INT_TOL, 0);

  cnn_config.layer_config[0].skip_width = stride;
  cnn_config.layer_config[0].skip_height = stride;

  RunCNNTest(image_width, image_height, input, expected_ver_hor, cnn_config,
             image_width, INT_TOL, 0);
}

TEST_F(CNNTest, TestSingleLayerOddStrideEvenDimImage) {
  int image_width = 8;
  int image_height = 8;
  int stride = 3;
  float input[] = {
    1, 3, 1,  -3, 1,  8,  7,  -1, 4,  2,  -1, -2, 0,  -4, -5, -3,
    5, 0, -3, -1, -1, -3, 0,  7,  -4, 1,  8,  2,  1,  0,  1,  6,
    5, 1, 2,  8,  0,  0,  -2, 2,  -1, -3, -1, 8,  6,  6,  -3, 4,
    5, 9, 9,  2,  -3, -4, 2,  7,  1,  6,  0,  -4, -5, 1,  1,  3,
  };
  float expected_horizontal[] = {
    22, -28, -49, 46,  86, -20, -53, -11, 18, -31, 12,  23,
    38, -16, 1,   -15, 0,  25,  -39, 36,  32, 104, -56, 22,
  };
  float expected_vertical[] = {
    19,  46, -15, -20, 86, 67,  -31, -20, 26,  38, -44, 2,
    -16, 40, 44,  1,   44, 104, 30,  -16, -56, 17, 36,  22,
  };
  float expected_ver_hor[] = {
    46, 86, -20, 38, -16, 1, 104, -56, 22,
  };
  float weights[] = {
    0, 2, 7, -3, 5, -5, 5, -2, -3,
  };
  float bias[] = { -4 };

  CNN_CONFIG cnn_config = { .num_layers = 1,
                            .is_residue = 0,
                            .ext_width = 0,
                            .ext_height = 0,
                            .strict_bounds = 0,
                            {
                                {
                                    .deconvolve = 0,
                                    .in_channels = 1,
                                    .filter_width = 3,
                                    .filter_height = 3,
                                    .out_channels = 1,
                                    .skip_width = stride,
                                    .skip_height = 1,
                                    .maxpool = 0,
                                    .weights = weights,
                                    .bias = bias,
                                    .pad = PADDING_SAME_ZERO,
                                    .activation = NONE,
                                    .input_copy = 0,
                                    .skip_combine = SKIP_NONE,
                                },
                            } };

  RunCNNTest(image_width, image_height, input, expected_horizontal, cnn_config,
             image_width, INT_TOL, 0);

  cnn_config.layer_config[0].skip_width = 1;
  cnn_config.layer_config[0].skip_height = stride;

  RunCNNTest(image_width, image_height, input, expected_vertical, cnn_config,
             image_width, INT_TOL, 0);

  cnn_config.layer_config[0].skip_width = stride;
  cnn_config.layer_config[0].skip_height = stride;

  RunCNNTest(image_width, image_height, input, expected_ver_hor, cnn_config,
             image_width, INT_TOL, 0);
}

TEST_F(CNNTest, TestSingleLayerOddStrideOddDimImage) {
  int image_width = 7;
  int image_height = 7;
  int stride = 3;
  float input[] = {
    -5, -3, -1, 1, 4,  4, 6,  4,  0,  0,  3,  2, -5, 3, -5, 2, 7,
    8,  -4, 7,  5, 9,  2, -5, -3, -2, -2, -5, 9, 8,  7, 9,  7, -1,
    -3, 9,  1,  4, -3, 1, -2, 9,  5,  8,  -1, 4, 2,  7, 9,
  };
  float expected_horizontal[] = {
    -2, 4,  -39, -14, -2,  5,  19,  -37, 7,   -102, 14,
    57, -3, 26,  -37, -72, 35, -65, 50,  -12, 2,
  };
  float expected_vertical[] = {
    -2,  9, -4, 4,  28,  43, -39, -102, -53, -7, 14,
    -16, 8, 57, 50, -19, 79, -12, 50,   -3,  2,
  };
  float expected_ver_hor[] = {
    -2, 4, -39, -102, 14, 57, 50, -12, 2,
  };
  float weights[] = {
    2, 3, -3, 3, -5, 6, 3, -3, -3,
  };
  float bias[] = { 3 };

  CNN_CONFIG cnn_config = { .num_layers = 1,
                            .is_residue = 0,
                            .ext_width = 0,
                            .ext_height = 0,
                            .strict_bounds = 0,
                            {
                                {
                                    .deconvolve = 0,
                                    .in_channels = 1,
                                    .filter_width = 3,
                                    .filter_height = 3,
                                    .out_channels = 1,
                                    .skip_width = stride,
                                    .skip_height = 1,
                                    .maxpool = 0,
                                    .weights = weights,
                                    .bias = bias,
                                    .pad = PADDING_SAME_ZERO,
                                    .activation = NONE,
                                    .input_copy = 0,
                                    .skip_combine = SKIP_NONE,
                                },
                            } };

  RunCNNTest(image_width, image_height, input, expected_horizontal, cnn_config,
             image_width, INT_TOL, 0);

  cnn_config.layer_config[0].skip_width = 1;
  cnn_config.layer_config[0].skip_height = stride;

  RunCNNTest(image_width, image_height, input, expected_vertical, cnn_config,
             image_width, INT_TOL, 0);

  cnn_config.layer_config[0].skip_width = stride;
  cnn_config.layer_config[0].skip_height = stride;

  RunCNNTest(image_width, image_height, input, expected_ver_hor, cnn_config,
             image_width, INT_TOL, 0);
}

TEST_F(CNNTest, TestValidPadding) {
  int image_width = 8;
  int image_height = 8;
  int stride = 2;
  float input[] = {
    -4, 9,  8,  0,  4,  5, -1, -4, 6,  9,  -5, 9,  6,  6, 7, 2,
    -1, 0,  -2, -5, 8,  0, 2,  7,  4,  2,  1,  7,  6,  1, 1, 3,
    3,  -1, -1, 7,  -2, 8, 4,  1,  -4, 4,  -1, -3, 5,  3, 5, -2,
    -5, 8,  3,  -1, 3,  0, 2,  5,  -5, -3, 8,  6,  -3, 9, 7, 0,
  };

  float expected_hor_stride[] = {
    11, 50, 68, -1, -88, 65, 29, 37, 16, -2, 2, 48, -33, -29, 6, -56, 50, -29,
  };

  float expected_ver_stride[] = {
    11, 63, 50, 0, 68, 25, 29, -15, 37, -10, 16, 36, -33, 45, -29, 60, 6, 27,
  };

  float expected_hor_ver_stride[] = {
    11, 50, 68, 29, 37, 16, -33, -29, 6,
  };

  float weights[] = {
    5, -3, 2, 2, 3, 1, 4, -3, -4,
  };

  float bias[] = {
    4,
  };

  CNN_CONFIG cnn_config = { .num_layers = 1,
                            .is_residue = 0,
                            .ext_width = 0,
                            .ext_height = 0,
                            .strict_bounds = 0,
                            {
                                {
                                    .deconvolve = 0,
                                    .in_channels = 1,
                                    .filter_width = 3,
                                    .filter_height = 3,
                                    .out_channels = 1,
                                    .skip_width = stride,
                                    .skip_height = 1,
                                    .maxpool = 0,
                                    .weights = weights,
                                    .bias = bias,
                                    .pad = PADDING_VALID,
                                    .activation = NONE,
                                    .input_copy = 0,
                                    .skip_combine = SKIP_NONE,
                                },
                            } };

  RunCNNTest(image_width, image_height, input, expected_hor_stride, cnn_config,
             image_width, INT_TOL, 0);

  cnn_config.layer_config[0].skip_width = 1;
  cnn_config.layer_config[0].skip_height = stride;

  RunCNNTest(image_width, image_height, input, expected_ver_stride, cnn_config,
             image_width, INT_TOL, 0);

  cnn_config.layer_config[0].skip_width = stride;
  cnn_config.layer_config[0].skip_height = stride;

  RunCNNTest(image_width, image_height, input, expected_hor_ver_stride,
             cnn_config, image_width, INT_TOL, 0);
}

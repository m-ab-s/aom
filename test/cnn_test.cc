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

#define SQR(x) ((x) * (x))
#define TOL 1E-5

struct CNN_TEST_CONFIG {
  const int image_width;
  const int image_height;
  float *input_tensor;
  float *expected_tensor;
  const int in_stride;
  const int out_stride;
  const CNN_CONFIG cnn_config;
};

namespace {

class CNNTest : public ::testing::Test {
 protected:
  static const CNN_TEST_CONFIG SetupCNNTest(
      int image_width, int image_height, float *input, float *expected_output,
      int layers, int is_residue, float *layer_weights, float *layer_bias,
      int *filter_width, int *filter_height, int *deconvolve,
      int *input_channels, int *output_channels, int *skip_width,
      int *skip_height, PADDING_TYPE *pads, ACTIVATION *activations,
      int *input_copy, int *output_add) {
    const size_t input_size = sizeof(*input) * image_width * image_height;

    float *input_tensor = (float *)aom_malloc(input_size);
    assert(input_tensor != NULL);
    memcpy(input_tensor, input, input_size);

    const size_t expected_output_size =
        sizeof(*expected_output) * image_width * image_height;
    float *expected_tensor = (float *)aom_malloc(expected_output_size);
    assert(expected_tensor != NULL);
    memcpy(expected_tensor, expected_output, expected_output_size);

    CNN_CONFIG cnn_config;
    cnn_config.num_layers = layers;
    cnn_config.is_residue = is_residue;

    size_t weight_offset = 0;
    size_t bias_offset = 0;

    for (int i = 0; i < layers; ++i) {
      const size_t weights_size = filter_width[i] * filter_height[i] *
                                  input_channels[i] * output_channels[i] *
                                  sizeof(*layer_weights);

      float *weights = (float *)aom_malloc(weights_size);
      assert(weights != NULL);
      memcpy(weights, (float *)layer_weights + weight_offset, weights_size);

      const int bias_size = output_channels[i] * sizeof(*layer_bias);
      float *bias = (float *)aom_malloc(bias_size);
      assert(bias != NULL);
      memcpy(bias, (float *)layer_bias + bias_offset, bias_size);

      cnn_config.layer_config[i] = (CNN_LAYER_CONFIG){
        .deconvolve = deconvolve[i],
        .in_channels = input_channels[i],
        .filter_width = filter_width[i],
        .filter_height = filter_height[i],
        .out_channels = output_channels[i],
        .skip_width = skip_width[i],
        .skip_height = skip_height[i],
        .weights = weights,
        .bias = bias,
        .pad = pads[i],
        .activation = activations[i],
        .input_copy = input_copy[i],
        .output_add = output_add[i],
      };
      weight_offset += (weights_size / sizeof(*layer_weights));
      bias_offset += (bias_size / sizeof(*layer_bias));
    }
    return { .image_width = image_width,
             .image_height = image_height,
             .input_tensor = input_tensor,
             .expected_tensor = expected_tensor,
             .in_stride = image_width,
             .out_stride = image_width,
             .cnn_config = cnn_config };
  }

  static void TearDownCNNTest(CNN_TEST_CONFIG cnn_test_config) {
    aom_free(cnn_test_config.input_tensor);
    aom_free(cnn_test_config.expected_tensor);
    for (int i = 0; i < cnn_test_config.cnn_config.num_layers; ++i) {
      aom_free(cnn_test_config.cnn_config.layer_config[i].weights);
      aom_free(cnn_test_config.cnn_config.layer_config[i].bias);
    }
  }

  static const CNN_TEST_CONFIG SetupTestNonActivationSingleLayerSingleKernel() {
    int image_width = 8;
    int image_height = 8;
    int layers = 1;
    int is_residue = 0;
    float inputs[] = {
      199, 194, 246, 118, 167, 208, 91,  101, 62,  102, 200, 14,  180,
      191, 85,  37,  3,   35,  87,  37,  230, 109, 17,  43,  121, 145,
      23,  147, 105, 41,  41,  121, 196, 202, 230, 25,  205, 122, 132,
      67,  190, 134, 40,  5,   159, 35,  130, 204, 112, 193, 135, 124,
      143, 246, 110, 207, 218, 14,  49,  42,  97,  96,  214, 115,
    };
    float expected[] = {
      365, 588, 554, 469, 542, 542, 368, 186, 377, 691, 627, 697, 802,
      695, 491, 198, 299, 474, 379, 629, 658, 547, 360, 201, 454, 610,
      479, 643, 613, 490, 376, 234, 653, 693, 608, 504, 537, 520, 570,
      374, 671, 863, 528, 614, 620, 718, 648, 517, 548, 567, 431, 513,
      534, 765, 821, 517, 420, 409, 334, 358, 499, 505, 654, 351,
    };
    float weights[] = { 0.508f, 0.367f, 0.93f,  0.546f, 0.882f,
                        0.476f, 0.24f,  0.713f, 0.516f };
    float bias[] = { 0.529f };
    int filter_width[] = { 3 };
    int filter_height[] = { 3 };
    int deconvolve[] = { 0 };
    int input_channels[] = { 1 };
    int output_channels[] = { 1 };
    int skip_width[] = { 1 };
    int skip_height[] = { 1 };
    int input_copy[] = {
      0,
    };
    int output_add[] = {
      0,
    };
    PADDING_TYPE pads[] = { PADDING_SAME_ZERO };
    ACTIVATION activations[] = { NONE };
    return SetupCNNTest(image_width, image_height, inputs, expected, layers,
                        is_residue, weights, bias, filter_width, filter_height,
                        deconvolve, input_channels, output_channels, skip_width,
                        skip_height, pads, activations, input_copy, output_add);
  }

  static const CNN_TEST_CONFIG SetupTestRELUMultiLayerMultiKernel() {
    int image_width = 8;
    int image_height = 8;
    int layers = 3;
    int is_residue = 0;
    float inputs[] = { 1, 8, 2, 2, 4, 8, 1, 8, 3, 3, 7, 1, 3, 3, 2, 6,
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
    int filter_width[] = { 3, 3, 3 };
    int filter_height[] = { 3, 3, 3 };
    int deconvolve[] = { 0, 0, 0 };
    int input_channels[] = { 1, 4, 4 };
    int output_channels[] = { 4, 4, 1 };
    int skip_width[] = { 1, 1, 1 };
    int skip_height[] = { 1, 1, 1 };
    int input_copy[] = { 0, 0, 0 };
    int output_add[] = { 0, 0, 0 };
    PADDING_TYPE pads[] = { PADDING_SAME_ZERO, PADDING_SAME_ZERO,
                            PADDING_SAME_ZERO };
    ACTIVATION activations[] = { RELU, RELU, RELU };
    return SetupCNNTest(image_width, image_height, inputs, expected, layers,
                        is_residue, weights, bias, filter_width, filter_height,
                        deconvolve, input_channels, output_channels, skip_width,
                        skip_height, pads, activations, input_copy, output_add);
  }

  static const CNN_TEST_CONFIG SetupTestSoftsignMultiLayerMultiKernel() {
    int image_width = 8;
    int image_height = 8;
    int layers = 3;
    int is_residue = 0;
    float inputs[] = { 0.517f, 0.505f, 0.769f, 0.537f, 0.55f,  0.264f, 0.991f,
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
    int filter_width[] = { 3, 3, 3 };
    int filter_height[] = { 3, 3, 3 };
    int deconvolve[] = { 0, 0, 0 };
    int input_channels[] = { 1, 4, 4 };
    int output_channels[] = { 4, 4, 1 };
    int skip_width[] = { 1, 1, 1 };
    int skip_height[] = { 1, 1, 1 };
    PADDING_TYPE pads[] = { PADDING_SAME_ZERO, PADDING_SAME_ZERO,
                            PADDING_SAME_ZERO };
    ACTIVATION activations[] = { SOFTSIGN, SOFTSIGN, SOFTSIGN };
    int input_copy[] = { 0, 0, 0 };
    int output_add[] = { 0, 0, 0 };
    return SetupCNNTest(image_width, image_height, inputs, expected, layers,
                        is_residue, weights, bias, filter_width, filter_height,
                        deconvolve, input_channels, output_channels, skip_width,
                        skip_height, pads, activations, input_copy, output_add);
  }

  static void VerifyMeanSquaredError(int image_size, float *expected,
                                     float *output, double tolerance) {
    double mean_squared_error = 0;
    for (int i = 0; i < image_size; ++i) {
      EXPECT_LE(fabsf(expected[i] - output[i]), 1);
      mean_squared_error += SQR(expected[i] - output[i]);
    }
    EXPECT_LE(mean_squared_error, tolerance);
  }

  static void RoundImage(int image_size, float *image) {
    for (int i = 0; i < image_size; ++i) {
      image[i] = roundf(image[i]);
    }
  }
};
}  // namespace

TEST_F(CNNTest, TestNonActivationSingleLayerSingleKernel) {
  const CNN_TEST_CONFIG cnn_test_config =
      SetupTestNonActivationSingleLayerSingleKernel();
  const int image_size =
      cnn_test_config.image_width * cnn_test_config.image_height;
  float *output_tensor =
      (float *)aom_malloc(image_size * sizeof(cnn_test_config.image_width));
  assert(output_tensor != NULL);

  av1_cnn_predict_c(cnn_test_config.input_tensor, cnn_test_config.image_width,
                    cnn_test_config.image_height, cnn_test_config.in_stride,
                    &cnn_test_config.cnn_config, output_tensor,
                    cnn_test_config.out_stride);

  RoundImage(image_size, output_tensor);
  VerifyMeanSquaredError(image_size, cnn_test_config.expected_tensor,
                         output_tensor, 0);

  TearDownCNNTest(cnn_test_config);
  aom_free(output_tensor);
}

TEST_F(CNNTest, TestRELUMultiLayerMultiKernel) {
  const CNN_TEST_CONFIG cnn_test_config = SetupTestRELUMultiLayerMultiKernel();
  const int image_size =
      cnn_test_config.image_width * cnn_test_config.image_height;
  float *output_tensor =
      (float *)aom_malloc(image_size * sizeof(cnn_test_config.image_width));
  assert(output_tensor != NULL);

  av1_cnn_predict_c(cnn_test_config.input_tensor, cnn_test_config.image_width,
                    cnn_test_config.image_height, cnn_test_config.in_stride,
                    &cnn_test_config.cnn_config, output_tensor,
                    cnn_test_config.out_stride);

  VerifyMeanSquaredError(image_size, cnn_test_config.expected_tensor,
                         output_tensor, TOL);

  TearDownCNNTest(cnn_test_config);
  aom_free(output_tensor);
}

TEST_F(CNNTest, TestSoftsignMultiLayerMultiKernel) {
  const CNN_TEST_CONFIG cnn_test_config =
      SetupTestSoftsignMultiLayerMultiKernel();
  const int image_size =
      cnn_test_config.image_width * cnn_test_config.image_height;
  float *output_tensor =
      (float *)aom_malloc(image_size * sizeof(cnn_test_config.image_width));
  assert(output_tensor != NULL);

  av1_cnn_predict_c(cnn_test_config.input_tensor, cnn_test_config.image_width,
                    cnn_test_config.image_height, cnn_test_config.in_stride,
                    &cnn_test_config.cnn_config, output_tensor,
                    cnn_test_config.out_stride);

  VerifyMeanSquaredError(image_size, cnn_test_config.expected_tensor,
                         output_tensor, TOL);

  TearDownCNNTest(cnn_test_config);
  aom_free(output_tensor);
}

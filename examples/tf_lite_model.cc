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

/*!\file
 * \brief This is an sample binary showing how to load a TF-lite model.
 *
 * 1. Build your model like normal.
 * 2. Follow the steps at https://www.tensorflow.org/lite/convert/python_api
 *    to convert into a TF-lite model. Optionally quantize the model to make
 *    it smaller (see the Quantization section at
 *    https://www.tensorflow.org/lite/guide/get_started)
 * 3. Run `xxd -i model.tflite > model.cc` to make it a CC file.
 * 4. Change the declaration to be const.
 * 5. Create a .h file that exposes the array and length.
 * 6. Add appropriate copyright headers and includes.
 * 7. Update the ops registration as needed (see tf_lite_ops_registration.cc
 *    for more details).
 */

#include <cstdio>
#include <cstdlib>
#include <memory>

#include "examples/tf_lite_model_data.h"
#include "examples/tf_lite_ops_registration.h"
#include "third_party/tensorflow/tensorflow/lite/interpreter.h"
#include "third_party/tensorflow/tensorflow/lite/model.h"

int main() {
  // Toy model that takes in four floats and returns a classification
  // probability.
  auto model = tflite::GetModel(tf_lite_model_data);

  // Build the interpreter.
  tflite::MutableOpResolver resolver;
  RegisterSpecificOps(&resolver);
  tflite::InterpreterBuilder builder(model, resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);

  std::unique_ptr<tflite::ErrorReporter> reporter(
      tflite::DefaultErrorReporter());

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    reporter->Report("Failed");
    return EXIT_FAILURE;
  }

  float *input = interpreter->typed_input_tensor<float>(0);
  for (int i = 0; i < 4; ++i) {
    input[i] = i;
  }
  auto status = interpreter->Invoke();
  if (status != kTfLiteOk) {
    reporter->Report("Failed");
    return EXIT_FAILURE;
  }
  float y0 = interpreter->typed_output_tensor<float>(0)[0];
  float y1 = interpreter->typed_output_tensor<float>(0)[1];
  printf("Sample classification probability: %f, %f\n", y0, y1);

  return EXIT_SUCCESS;
}

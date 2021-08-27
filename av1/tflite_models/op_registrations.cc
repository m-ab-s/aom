#include "av1/tflite_models/op_registrations.h"
#include "tensorflow/lite/kernels/builtin_op_kernels.h"

void RegisterSelectedOpsAllQps(::tflite::MutableOpResolver *resolver) {
  resolver->AddBuiltin(::tflite::BuiltinOperator_ADD,
                       ::tflite::ops::builtin::Register_ADD(), 1, 2);
  resolver->AddBuiltin(::tflite::BuiltinOperator_CONV_2D,
                       ::tflite::ops::builtin::Register_CONV_2D(), 1, 3);
  resolver->AddBuiltin(::tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
                       ::tflite::ops::builtin::Register_DEPTHWISE_CONV_2D(), 1,
                       3);
  resolver->AddBuiltin(::tflite::BuiltinOperator_DEQUANTIZE,
                       ::tflite::ops::builtin::Register_DEQUANTIZE(), 2, 2);
  resolver->AddBuiltin(::tflite::BuiltinOperator_MIRROR_PAD,
                       ::tflite::ops::builtin::Register_MIRROR_PAD());
  resolver->AddBuiltin(::tflite::BuiltinOperator_PAD,
                       ::tflite::ops::builtin::Register_PAD(), 1, 2);
  resolver->AddBuiltin(::tflite::BuiltinOperator_QUANTIZE,
                       ::tflite::ops::builtin::Register_QUANTIZE());
}

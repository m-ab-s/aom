# Copyright (c) 2019, Alliance for Open Media. All rights reserved
#
# This source code is subject to the terms of the BSD 2 Clause License and
# the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
# was not distributed with this source code in the LICENSE file, you can
# obtain it at www.aomedia.org/license/software. If the Alliance for Open
# Media Patent License 1.0 was not distributed with this source code in the
# PATENTS file, you can obtain it at www.aomedia.org/license/patent.
#

r"""Python module that transforms a Tensorflow ckpt to a C header file.

Usage: tf_ckpt_to_c_header.py [-h] [--input_path INPUT_PATH]
                              [--output_path OUTPUT_PATH]
                              [--header_guard HEADER_GUARD]
                              [--config_name CONFIG_NAME]
                              [--var_regex VAR_REGEX]
                              [--trained_qp TRAINED_QP]
                              [--is_residue IS_RESIDUE]
                              [--ext_width EXT_WIDTH]
                              [--ext_height EXT_HEIGHT]
                              [--strict_bounds STRICT_BOUNDS]
                              [--enable_explicit_field_names]
                              [--enable_aligned_declaration]
                              [--architecture {VDSR|WDSR}]

Optional Arguments:
  -h, --help            show this help message and exit
  --input_path INPUT_PATH
                        Path to ckpt. Please include the full prefix of all
                        relevant ckpt files.
  --output_path OUTPUT_PATH
                        Path to output of file.
  --header_guard HEADER_GUARD
                        Name of the header file header guard.
  --config_name CONFIG_NAME
                        Name of model and prefix to all relevant weight, bias,
                        and qp variables.
  --var_regex VAR_REGEX
                        Regex to match tensor names against in the model ckpt.
  --trained_qp TRAINED_QP
                        The primary QP of the model.
  --is_residue IS_RESIDUE
                        Whether we are predicting an image or the residue of
                        an image.
  --ext_width EXT_WIDTH
                        Width of the frame extension.
  --ext_height EXT_HEIGHT
                        Height of the frame extension.
  --strict_bounds STRICT_BOUNDS
                        Whether the input bounds are strict or not.
  --enable_explicit_field_names
                        Whether to print field names along side values in cnn
                        config.
  --enable_aligned_declaration
                        Whether to align all the weights and biases in memory
                        along multiples of four.
  --architecture {VDSR|WDSR}
                        Kind of architecture the network uses.

Example Invocation:
$ python tf_ckpt_to_c_header.py --input_path="./model.ckpt" --output_path=
    "./output.h" --header_guard="MODEL_H_" --trained_qp=32
    --enable_explicit_field_names
"""

import argparse
import collections
import copy
import re
import sys

from tf_ckpt_to_c_header_struct_parser import StructNode

from tensorflow import logging
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.platform import app

FLAGS = None
BRANCH_CONFIG_ORDER = StructNode((
    ("input_to_branches", "0x00"),
    ("copy_to_branches", 0),
    ("branches_to_combine", "0x00")), name="branch_config")
BATCHNORM_PARAMS_ORDER = StructNode((
    ("bn_gamma", None),
    ("bn_beta", None),
    ("bn_mean", None),
    ("bn_std", None)), name="batchnorm_params")
LAYER_CONFIG_ORDER = StructNode((
    ("in_channels", -1),
    ("filter_width", -1),
    ("filter_height", -1),
    ("out_channels", -1),
    ("skip_width", 1),
    ("skip_height", 1),
    ("maxpool", 0),
    ("weights", ""),
    ("bias", ""),
    ("pad", "PADDING_SAME_ZERO"),
    ("activation", "NONE"),
    ("deconvolve", 0),
    ("branch", 0),
    ("branch_copy_type", "BRANCH_NO_COPY"),
    ("branch_combine_type", "BRANCH_NOC"),
    ("branch_config", BRANCH_CONFIG_ORDER),
    ("bn_params", BATCHNORM_PARAMS_ORDER)))
CNN_CONFIG_ORDER = {}
BIT_ALIGNMENT = 32
WEIGHT_STRING = "w"
BIAS_STRING = "b"

logging.set_verbosity("INFO")


def _get_weight_tensor_from_prefix(tensor_prefix):
  return tensor_prefix + WEIGHT_STRING


def _get_bias_tensor_from_prefix(tensor_prefix):
  return tensor_prefix + BIAS_STRING


def _generate_layer_index_tensor_name_map(input_reader, var_regex):
  """Create a map from tuples of indices to tensor names."""
  layer_index_tensor_name_map = {}
  # Create a index-tuple to tensor name map.
  for k, _ in input_reader.get_variable_to_shape_map().iteritems():
    match = re.match(var_regex, k)
    if match:
      # Value is the prefix of the layer variables.
      var_indices = tuple(int(x) for x in match.group(1).split("_")[:-1])
      # Ignore the (b|w) decorator since we cannot control the order they
      # are read.
      layer_index_tensor_name_map[var_indices] = k[:-1]
  logging.info("Gathered {0} variables: {1}".format(
      len(layer_index_tensor_name_map),
      [value for _, value in layer_index_tensor_name_map.iteritems()]))
  return layer_index_tensor_name_map


def _format_tensor(tensor, layer):
  """Reformats the tensor from Python-style array to C-style array."""
  flattened_tensor = tensor.flatten()
  if not sum(flattened_tensor):
    logging.warning("Tensor at layer %d is a zero tensor!", layer)
  parsed_tensor = " ".join([("%ff," % value) for value in flattened_tensor])
  return "{" + parsed_tensor + "}"


def _build_layer_config(shape, layer, num_layers):
  layer_config = copy.deepcopy(LAYER_CONFIG_ORDER)
  layer_config["weights"] = "%s_weight_%s" % (FLAGS.config_name, layer)
  layer_config["bias"] = "%s_bias_%s" % (FLAGS.config_name, layer)
  layer_config["filter_width"] = shape[0]
  layer_config["filter_height"] = shape[1]
  layer_config["in_channels"] = shape[2]
  layer_config["out_channels"] = shape[3]
  if FLAGS.architecture == "WDSR" and 0 < layer and layer < num_layers - 1:
    branch_config = copy.deepcopy(BRANCH_CONFIG_ORDER)
    # If layer belongs to a residual block.
    if (layer % 3) == 1:
      # Input residual block layer.
      layer_config["activation"] = "RELU"
      layer_config["branch_copy_type"] = "BRANCH_INPUT"
      branch_config["input_to_branches"] = "0x02"
    elif (layer % 3) == 0:
      # Output residual block layer.
      layer_config["branch_combine_type"] = "BRANCH_ADD"
      branch_config["branches_to_combine"] = "0x02"
    layer_config["branch_config"] = branch_config
  elif FLAGS.architecture == "VDSR" and layer < num_layers - 1:
    layer_config["activation"] = "RELU"
  return layer_config


def _extract_layer_variables(input_reader):
  """Extracts the graph, weights, and biases from input_file."""
  cnn_config = CNN_CONFIG_ORDER
  layer_index_tensor_name_map = _generate_layer_index_tensor_name_map(
      input_reader, FLAGS.var_regex)
  num_layers = len(layer_index_tensor_name_map)
  cnn_config["num_layers"] = num_layers

  cnn_config["layer_config"] = [None] * num_layers
  weights = [None] * num_layers
  biases = [None] * num_layers

  layer = 0
  # By the end of this process, every layer should have weights and biases.
  for _, tensor_name in sorted(layer_index_tensor_name_map.iteritems()):
    cnn_config["layer_config"][layer] = _build_layer_config(
        input_reader.get_variable_to_shape_map()[
            _get_weight_tensor_from_prefix(tensor_name)],
        layer,
        num_layers)
    weights[layer] = _format_tensor(
        input_reader.get_tensor(_get_weight_tensor_from_prefix(tensor_name)),
        layer)
    biases[layer] = _format_tensor(
        input_reader.get_tensor(_get_bias_tensor_from_prefix(tensor_name)),
        layer)
    layer += 1
  return cnn_config, weights, biases


def _print_header(header_guard, output_file):
  """Writes the header of the header file. Contains the aom license."""
  output_file.write("""/*
 * Copyright (c) 2019, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */
""")
  output_file.write("\n#ifndef %s" % header_guard)
  output_file.write("\n#define %s\n" % header_guard)
  if FLAGS.enable_aligned_declaration:
    output_file.write("\n#include \"aom_ports/mem.h\"")
  output_file.write("\n#include \"av1/common/cnn.h\"\n")


def _print_weights_biases(layers, weights, biases, output_file):
  """Write the contents of the cnn to the header file.

  Each variable has static const prepended to them to ensure they remain
  local to the header and any file included them and need not be modified.
  All variables have FLAGS.config_name prepended to their name to distinguish
  them from other variables in other model header files.

  The layout of the file is as follows:
    - qp of the model,
    - weights and biases in alternating order,
    - the graph.

  The variables printed in the graph are determined by CNN_CONFIG_ORDER
  and LAYER_CONFIG_ORDER. It is especially important to maintain the orders in
  each of these constants because C expects that any initialized struct
  maintains the order of the fields it has in its declaration.

  Args:
    cnn_config: A dictionary containing all the graph parameters.
    weights: An array containing the weights of each layer in order.
    biases: An array containing the bias of each layer in order.
    output_file: The output file.
  """
  output_file.write("\nstatic const int %s_trained_qp = %d;\n" %
                    (FLAGS.config_name, FLAGS.trained_qp))
  if FLAGS.enable_aligned_declaration:
    for layer in range(layers):
      output_file.write("\nDECLARE_ALIGNED(%d, static float, " \
                        "%s_weight_%d[]) = %s;\n" %
                        (BIT_ALIGNMENT, FLAGS.config_name, layer,
                         weights[layer]))
      output_file.write("\nDECLARE_ALIGNED(%d, static float, " \
                        "%s_bias_%d[]) = %s;\n" %
                        (BIT_ALIGNMENT, FLAGS.config_name, layer,
                         biases[layer]))
  else:
    for layer in range(layers):
      output_file.write("\nstatic float %s_weight_%d[] = %s;\n" %
                        (FLAGS.config_name, layer, weights[layer]))
      output_file.write("\nstatic float %s_bias_%d[] = %s;\n" %
                        (FLAGS.config_name, layer, biases[layer]))


def _print_cnn_config(cnn_config, output_file):
  output_file.write("\nconst CNN_CONFIG %s = {" % FLAGS.config_name)
  cnn_config.write_fields_to_output(output_file,
                                    FLAGS.enable_explicit_field_names)
  output_file.write("\n};\n")


def _print_footer(header_guard, output_file):
  """Writes the footer of the header file."""
  output_file.write("\n#endif  // %s" % header_guard)


def generate_header_file():
  """Generates a C header file from a Tensorflow model ckpt."""
  try:
    input_reader = pywrap_tensorflow.NewCheckpointReader(FLAGS.input_path)
  except Exception as e:
    raise e
  cnn_config, weights, biases = _extract_layer_variables(input_reader)

  assert not [weight for weight in weights if weight is None]
  assert not [bias for bias in biases if bias is None]
  assert not [
      config for config in cnn_config["layer_config"] if config is None
  ]

  with open(FLAGS.output_path, "w+") as output_file:
    _print_header(FLAGS.header_guard, output_file)
    _print_weights_biases(cnn_config["num_layers"], weights, biases,
                          output_file)
    _print_cnn_config(cnn_config, output_file)
    _print_footer(FLAGS.header_guard, output_file)

    output_file.close()


def main(_):
  generate_header_file()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--input_path",
      type=str,
      default="",
      help="Path to ckpt. Please include the full prefix of all relevant ckpt "
      " files.")
  parser.add_argument(
      "--output_path",
      type=str,
      default="",
      help="Path to output of file.")
  parser.add_argument(
      "--header_guard",
      type=str,
      default="AOM_AV1_COMMON_MODEL_H_",
      help="Name of the header file header guard.")
  parser.add_argument(
      "--config_name",
      type=str,
      default="model",
      help="Name of model and prefix to all relevant weight, bias, and qp "
      "variables.")
  parser.add_argument(
      "--var_regex",
      type=str,
      default=r".*conv_(([0-9][0-9]*_)*)(w|b)$",
      help="Regex to match tensor names against in the model ckpt.")
  parser.add_argument(
      "--trained_qp",
      type=int,
      default=32,
      help="The primary QP of the model.")
  parser.add_argument(
      "--is_residue",
      type=int,
      default=1,
      help="Whether we are predicting an image or the residue of an image.")
  parser.add_argument(
      "--ext_width",
      type=int,
      default=0,
      help="Width of the frame extension.")
  parser.add_argument(
      "--ext_height",
      type=int,
      default=0,
      help="Height of the frame extension.")
  parser.add_argument(
      "--strict_bounds",
      type=int,
      default=0,
      help="Whether the input bounds are strict or not.")
  parser.add_argument(
      "--enable_explicit_field_names",
      default=False,
      action="store_true",
      help="Whether to print field names along side values in cnn config.")
  parser.add_argument(
      "--enable_aligned_declaration",
      default=False,
      action="store_true",
      help="Whether to align weights and biases in memory along addresses "
           "that are multiples of four.")
  parser.add_argument(
      "--architecture",
      default=None,
      choices=["WDSR", "VDSR"],
      help="Kind of architecture the network uses.")
  FLAGS, unparsed = parser.parse_known_args()

  if FLAGS.architecture:
    logging.info("Using %s architecture.", FLAGS.architecture)
  else:
    logging.info("Using default architecture. You will need to fill network "
                 "details manually.")

  # The order of these fields must reflect the order of the structs in
  # ${AOM_ROOT}/av1/common/cnn.h.

  CNN_CONFIG_ORDER = StructNode((
      ("num_layers", 0),
      ("is_residue", FLAGS.is_residue),
      ("ext_width", FLAGS.ext_width),
      ("ext_height", FLAGS.ext_height),
      ("strict_bounds", FLAGS.strict_bounds),
      ("layer_config", [])
  ))

  app.run(main=main, argv=[sys.argv[0]] + unparsed)

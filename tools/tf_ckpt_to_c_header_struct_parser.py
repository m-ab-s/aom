import collections

class StructNode(collections.OrderedDict):

  _name = None

  def __init__(self, *args, **kwargs):
    self._name = kwargs.get("name")

    super(StructNode, self).__init__(*args, **kwargs)

  def __getitem__(self, key):
    return super(StructNode, self).__getitem__(key)

  def __setitem__(self, key, value):
    super(StructNode, self).__setitem__(key, value)

  def write_fields_to_output(self, output_file, include_names=False):
    if self._name:
      output_file.write(" // %s" % self._name)
    for key, value in super(StructNode, self).items():
      if key == "name":
        continue
      output_file.write("\n  ")
      if include_names:
        output_file.write(".{0} = ".format(key))
      if isinstance(value, StructNode):
        output_file.write("{")
        if value.items():
          value.write_fields_to_output(output_file, include_names)
          output_file.write("\n  },")
        else:
          output_file.write(" 0 },")
      elif isinstance(value, list):
        # If list, then it it is the cnn_config layer_configs.
        output_file.write("{")
        for idx, element in enumerate(value):
          output_file.write("\n    { // layer_%d" % idx)
          element.write_fields_to_output(output_file, include_names)
          output_file.write("\n    },")
        output_file.write("\n    }")
      else:
        if value is not None:
          output_file.write("{0},".format(value))
        else:
          output_file.write("NULL,")

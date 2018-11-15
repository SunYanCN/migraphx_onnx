# usage: python convert_fp16_onnx.py <fp32 ONNX file> <fp16 ONNX file>
#
# This script calls the onnxmltools to convert a fp32 model to fp16.
# Note that for some models, this can take a long time, e.g. measured 8 hours
# to convert resnet50i64 model from torchvision.  Other models are done within
# seconds.
#
import onnx
import sys
from onnx import optimizer
from onnxmltools.utils.float16_converter import convert_float_to_float16
if len(sys.argv) == 3:
    original_model = onnx.load(sys.argv[1])

    new_model = convert_float_to_float16(original_model)
    onnx.save_model(new_model,sys.argv[2])

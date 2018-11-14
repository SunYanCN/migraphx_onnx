This directory contains models derived from the Pytorch torchvision sequence
of models.  Each is generally derived using pytorch 0.4.0 to download a
pre-trained model, setting the eval() flag and then saving an ONNX file.

The file "dockerfile" can be used to create a docker container that includes
pytorch 0.4.0.

The file "get_torchvision.py" was used to generate ONNX files in this
irectory.

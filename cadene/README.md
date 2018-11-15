This directory contains models derived from the Pytorch pretrainedmodels
package maintained by Remi Cadene: http://pypi.org/project/pretrainedmodels/

Each model is generally derived using pytorch 0.4.0 to download a pre-trained
model and then saving to an ONNX file.

The file "dockerfile" can be used to create a docker container that includes
pytorch 0.4.0 and an instance of this package.  Note that the pretrained package
appears to be under active development.

The file "get_pretrained.py" was used to generate ONNX files in this directory.

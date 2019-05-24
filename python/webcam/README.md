Example program using webcam to classify incoming images.

NOTE: This requires a few pre-requisites:
1. The PYTHONPATH must be set properly to point to a /py directory for
   AMDMIGraphX.  Prebuilt packages can do this, otherwise as an example
   source ../PYTHONPATH.src

2. The script refers to resnet50i1.onnx.  This file can be created with a
   script in the ../onnx directory.  Note that this model does not have a
   softmax normalization layer. Thus the number printed is not a percentage.

3. The script requires a webcam. This may make it more difficult to run
   in a docker container.
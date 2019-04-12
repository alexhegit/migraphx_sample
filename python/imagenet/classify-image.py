# Demonstration script that classifies an image using an imagenet model.
#
# Usage: python classify-image.py <ONNX file> <image file>
#
import migraphx
import requests
import sys
import os
import numpy as np
from PIL import Image
from torchvision import models,transforms
from torch.autograd import Variable

if len(sys.argv) == 3:
    onnxfile = sys.argv[1]
    imagefile = sys.argv[2]
else:
    print('Usage: python classify-image.py <ONNX file> <image file>')
    sys.exit(1)

model = migraphx.parse_onnx(onnxfile)
model.compile(migraphx.get_target("gpu"))

normalize = transforms.Normalize(
    mean=[0.485,0.456,0.406],
    std=[0.229,0.224,0.224])

# allocate space
params = {}
for key,value in model.get_parameter_shapes().items():
   params[key] = migraphx.allocate_gpu(value)
   if key == '0':
       if value.lens() == [1L, 3L, 224L, 224L]:
           format='imagenet224'
           preprocess = transforms.Compose([
               transforms.Resize(256),
               transforms.CenterCrop(224),
               transforms.ToTensor(),
               normalize])
       elif value.lens() == [1L, 3L, 299L, 299L]:
           format='imagenet299'
           preprocess = transforms.Compose([
               transforms.Resize(299),
               transforms.CenterCrop(299),
               transforms.ToTensor(),
               normalize])           
       else:
           print('Expecting either 1x3x224x244 or 1x3x299x299 input format')
           sys.exit(1)

LABELS_URL='https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'
labels = {int(key):value for (key,value)
           in requests.get(LABELS_URL).json().items()}

img_pil = Image.open(imagefile)
img_tensor = preprocess(img_pil)
if img_tensor.shape[0] != 3:
    img_pil2 = img_pil.convert("RGB")
    img_tensor = preprocess(img_pil2)
img_tensor.unsqueeze_(0)
img_variable = Variable(img_tensor)

image = img_variable.numpy()

tmp_result = migraphx.to_gpu(migraphx.argument(image))
params['0'] = tmp_result
result = np.array(migraphx.from_gpu(model.run(params)),copy=False)
maxlist = result.argsort()

print(imagefile,
      maxlist[0][999],
      maxlist[0][998],
      maxlist[0][997],
      maxlist[0][996],
      maxlist[0][995])
print(labels[maxlist[0][999]])
print(labels[maxlist[0][998]])
print(labels[maxlist[0][997]])
print(labels[maxlist[0][996]])
print(labels[maxlist[0][995]])

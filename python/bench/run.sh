#!/bin/bash
python3 bench.py --framework tensorflow --save_file /home/mev/source/migraphx_sample/tfpb/mobilenet_v2i1.pb --image_file /home/mev/source/migraphx_sample/python/bench/ILSVRC2012_val_00000001.JPEG
python2 bench.py --framework migraphx --save_file /home/mev/source/migraphx_sample/tfpb/mobilenet_v2i1.pb --image_file /home/mev/source/migraphx_sample/python/bench/ILSVRC2012_val_00000001.JPEG

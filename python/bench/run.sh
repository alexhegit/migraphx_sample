#!/bin/bash
#export MIOPEN_FIND_ENFORCE=4
export TF_ROCM_FUSION_ENABLE=1

# resnet50v2
echo "resnet50v2 ******"
/usr/bin/python3 bench.py --model resnet50v2 --framework tensorflow --save_file /home/mev/source/migraphx_sample/tfpb/slim/resnet50v2_i64.pb --image_file /home/mev/source/migraphx_sample/python/bench/ILSVRC2012_val_00000001.JPEG
exit
/usr/bin/python2 bench_copy.py --model resnet50v2 --framework migraphx --save_file /home/mev/source/migraphx_sample/tfpb/slim/resnet50v2_i64.pb --fp16 --image_file /home/mev/source/migraphx_sample/python/bench/ILSVRC2012_val_00000001.JPEG
/usr/bin/python2 bench_copy.py --model resnet50v2 --framework migraphx --save_file /home/mev/source/migraphx_sample/tfpb/slim/resnet50v2_i64.pb --image_file /home/mev/source/migraphx_sample/python/bench/ILSVRC2012_val_00000001.JPEG
/usr/bin/python3 bench.py --model resnet50v2 --framework tensorflow --save_file /home/mev/source/migraphx_sample/tfpb/slim/resnet50v2_i1.pb --batch 1 --image_file /home/mev/source/migraphx_sample/python/bench/ILSVRC2012_val_00000001.JPEG
/usr/bin/python2 bench_copy.py --model resnet50v2 --framework migraphx --save_file /home/mev/source/migraphx_sample/tfpb/slim/resnet50v2_i1.pb --batch 1 --fp16 --image_file /home/mev/source/migraphx_sample/python/bench/ILSVRC2012_val_00000001.JPEG
/usr/bin/python2 bench_copy.py --model resnet50v2 --framework migraphx --save_file /home/mev/source/migraphx_sample/tfpb/slim/resnet50v2_i1.pb --batch 1 --image_file /home/mev/source/migraphx_sample/python/bench/ILSVRC2012_val_00000001.JPEG
#/usr/bin/python2 bench_nocopy.py --model resnet50v2 --framework migraphx --save_file /home/mev/source/migraphx_sample/tfpb/slim/resnet50v2_i64.pb --image_file /home/mev/source/migraphx_sample/python/bench/ILSVRC2012_val_00000001.JPEG

# resnet50v1
echo "resnet50v1 ******"
/usr/bin/python3 bench.py --model resnet50v1 --framework tensorflow --save_file /home/mev/source/migraphx_sample/tfpb/slim/resnet50v1_i64.pb --image_file /home/mev/source/migraphx_sample/python/bench/ILSVRC2012_val_00000001.JPEG
/usr/bin/python2 bench_copy.py --model resnet50v1 --framework migraphx --save_file /home/mev/source/migraphx_sample/tfpb/slim/resnet50v1_i64.pb --fp16 --image_file /home/mev/source/migraphx_sample/python/bench/ILSVRC2012_val_00000001.JPEG
/usr/bin/python2 bench_copy.py --model resnet50v1 --framework migraphx --save_file /home/mev/source/migraphx_sample/tfpb/slim/resnet50v1_i64.pb --image_file /home/mev/source/migraphx_sample/python/bench/ILSVRC2012_val_00000001.JPEG
/usr/bin/python3 bench.py --model resnet50v1 --framework tensorflow --save_file /home/mev/source/migraphx_sample/tfpb/slim/resnet50v1_i1.pb --batch 1 --image_file /home/mev/source/migraphx_sample/python/bench/ILSVRC2012_val_00000001.JPEG
/usr/bin/python2 bench_copy.py --model resnet50v1 --framework migraphx --save_file /home/mev/source/migraphx_sample/tfpb/slim/resnet50v1_i1.pb --batch 1 --fp16 --image_file /home/mev/source/migraphx_sample/python/bench/ILSVRC2012_val_00000001.JPEG
/usr/bin/python2 bench_copy.py --model resnet50v1 --framework migraphx --save_file /home/mev/source/migraphx_sample/tfpb/slim/resnet50v1_i1.pb --batch 1 --image_file /home/mev/source/migraphx_sample/python/bench/ILSVRC2012_val_00000001.JPEG
#/usr/bin/python2 bench_nocopy.py --model resnet50v1 --framework migraphx --save_file /home/mev/source/migraphx_sample/tfpb/slim/resnet50v1_i64.pb --image_file /home/mev/source/migraphx_sample/python/bench/ILSVRC2012_val_00000001.JPEG

# inceptionv3
echo "inceptionv3 ******"
/usr/bin/python3 bench.py --model inceptionv3 --framework tensorflow --resize_val=299 --save_file /home/mev/source/migraphx_sample/tfpb/slim/inceptionv3_i32.pb --image_file /home/mev/source/migraphx_sample/python/bench/ILSVRC2012_val_00000001.JPEG
/usr/bin/python2 bench_copy.py --model inceptionv3 --framework migraphx --resize_val=299 --save_file /home/mev/source/migraphx_sample/tfpb/slim/inceptionv3_i32.pb --fp16 --image_file /home/mev/source/migraphx_sample/python/bench/ILSVRC2012_val_00000001.JPEG
/usr/bin/python2 bench_copy.py --model inceptionv3 --framework migraphx --resize_val=299 --save_file /home/mev/source/migraphx_sample/tfpb/slim/inceptionv3_i32.pb --image_file /home/mev/source/migraphx_sample/python/bench/ILSVRC2012_val_00000001.JPEG
/usr/bin/python3 bench.py --model inceptionv3 --framework tensorflow --resize_val=299 --save_file /home/mev/source/migraphx_sample/tfpb/slim/inceptionv3_i1.pb --batch 1 --image_file /home/mev/source/migraphx_sample/python/bench/ILSVRC2012_val_00000001.JPEG
/usr/bin/python2 bench_copy.py --model inceptionv3 --framework migraphx --resize_val=299 --save_file /home/mev/source/migraphx_sample/tfpb/slim/inceptionv3_i1.pb --batch 1 --fp16 --image_file /home/mev/source/migraphx_sample/python/bench/ILSVRC2012_val_00000001.JPEG
/usr/bin/python2 bench_copy.py --model inceptionv3 --framework migraphx --resize_val=299 --save_file /home/mev/source/migraphx_sample/tfpb/slim/inceptionv3_i1.pb --batch 1 --image_file /home/mev/source/migraphx_sample/python/bench/ILSVRC2012_val_00000001.JPEG
#/usr/bin/python2 bench_nocopy.py --model inceptionv3 --framework migraphx --resize_val=299 --save_file /home/mev/source/migraphx_sample/tfpb/slim/inceptionv3_i32.pb --image_file /home/mev/source/migraphx_sample/python/bench/ILSVRC2012_val_00000001.JPEG

# vgg16
echo "vgg16 ******"
/usr/bin/python3 bench.py --model vgg16 --framework tensorflow --save_file /home/mev/source/migraphx_sample/tfpb/slim/vgg16_i16.pb --image_file /home/mev/source/migraphx_sample/python/bench/ILSVRC2012_val_00000001.JPEG
/usr/bin/python2 bench_copy.py --model vgg16 --framework migraphx --save_file /home/mev/source/migraphx_sample/tfpb/slim/vgg16_i16.pb --fp16 --image_file /home/mev/source/migraphx_sample/python/bench/ILSVRC2012_val_00000001.JPEG
/usr/bin/python2 bench_copy.py --model vgg16 --framework migraphx --save_file /home/mev/source/migraphx_sample/tfpb/slim/vgg16_i16.pb --image_file /home/mev/source/migraphx_sample/python/bench/ILSVRC2012_val_00000001.JPEG
/usr/bin/python3 bench.py --model vgg16 --framework tensorflow --save_file /home/mev/source/migraphx_sample/tfpb/slim/vgg16_i1.pb --batch 1 --image_file /home/mev/source/migraphx_sample/python/bench/ILSVRC2012_val_00000001.JPEG
/usr/bin/python2 bench_copy.py --model vgg16 --framework migraphx --save_file /home/mev/source/migraphx_sample/tfpb/slim/vgg16_i1.pb --batch 1 --fp16 --image_file /home/mev/source/migraphx_sample/python/bench/ILSVRC2012_val_00000001.JPEG
/usr/bin/python2 bench_copy.py --model vgg16 --framework migraphx --save_file /home/mev/source/migraphx_sample/tfpb/slim/vgg16_i1.pb --batch 1 --image_file /home/mev/source/migraphx_sample/python/bench/ILSVRC2012_val_00000001.JPEG
#/usr/bin/python2 bench_nocopy.py --model vgg16 --framework migraphx --save_file /home/mev/source/migraphx_sample/tfpb/slim/vgg16_i16.pb --image_file /home/mev/source/migraphx_sample/python/bench/ILSVRC2012_val_00000001.JPEG

# mobilenet
echo "mobilenet ******"
/usr/bin/python3 bench.py --model mobilenet --framework tensorflow --save_file /home/mev/source/migraphx_sample/tfpb/slim/mobilenet_i64.pb --image_file /home/mev/source/migraphx_sample/python/bench/ILSVRC2012_val_00000001.JPEG
/usr/bin/python2 bench_copy.py --model mobilenet --framework migraphx --save_file /home/mev/source/migraphx_sample/tfpb/slim/mobilenet_i64.pb --fp16 --image_file /home/mev/source/migraphx_sample/python/bench/ILSVRC2012_val_00000001.JPEG
/usr/bin/python2 bench_copy.py --model mobilenet --framework migraphx --save_file /home/mev/source/migraphx_sample/tfpb/slim/mobilenet_i64.pb --image_file /home/mev/source/migraphx_sample/python/bench/ILSVRC2012_val_00000001.JPEG
/usr/bin/python3 bench.py --model mobilenet --framework tensorflow --save_file /home/mev/source/migraphx_sample/tfpb/slim/mobilenet_i1.pb --batch 1 --image_file /home/mev/source/migraphx_sample/python/bench/ILSVRC2012_val_00000001.JPEG
/usr/bin/python2 bench_copy.py --model mobilenet --framework migraphx --save_file /home/mev/source/migraphx_sample/tfpb/slim/mobilenet_i1.pb --batch 1 --fp16 --image_file /home/mev/source/migraphx_sample/python/bench/ILSVRC2012_val_00000001.JPEG
/usr/bin/python2 bench_copy.py --model mobilenet --framework migraphx --save_file /home/mev/source/migraphx_sample/tfpb/slim/mobilenet_i1.pb --batch 1 --image_file /home/mev/source/migraphx_sample/python/bench/ILSVRC2012_val_00000001.JPEG
#/usr/bin/python2 bench_nocopy.py --model mobilenet --framework migraphx --save_file /home/mev/source/migraphx_sample/tfpb/slim/mobilenet_i64.pb --image_file /home/mev/source/migraphx_sample/python/bench/ILSVRC2012_val_00000001.JPEG
exit 0


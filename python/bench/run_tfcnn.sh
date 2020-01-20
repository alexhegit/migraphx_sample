#!/bin/bash
PYTHON=${PYTHON:="/usr/bin/python3"}
TFCNNDIR=${TFCNNDIR:="/home/mev/source/tensorflow/benchmarks/scripts/tf_cnn_benchmarks"}
export TF_ROCM_FUSION_ENABLE=1
#export TF_ROCM_FUSION_ENABLE=0

echo "XLA"
cd ${TFCNNDIR}
${PYTHON} tf_cnn_benchmarks.py --xla=True --model resnet50 --batch_size=64 --forward_only=True > resnet50.out 2>&1
ips=`grep "total" resnet50.out | tail -1 | awk '{ print $3 }'`
echo "resnet50	$ips"

${PYTHON} tf_cnn_benchmarks.py --xla=True --model inception3 --batch_size=32 --forward_only=True > inception.out 2>&1
ips=`grep "total" inception.out | tail -1 | awk '{ print $3 }'`
echo "inception3	$ips"

${PYTHON} tf_cnn_benchmarks.py --xla=True --model vgg16 --batch_size=16 --forward_only=True > vgg.out 2>&1
ips=`grep "total" vgg.out | tail -1 | awk '{ print $3 }'`
echo "vgg16	$ips"

${PYTHON} tf_cnn_benchmarks.py --xla=True --model mobilenet --batch_size=64 --forward_only=True > mobilenet.out 2>&1
ips=`grep "total" mobilenet.out | tail -1 | awk '{ print $3 }'`
echo "mobilenet	$ips"

${PYTHON} tf_cnn_benchmarks.py --xla=True --model alexnet --batch_size=64 --forward_only=True > alexnet.out 2>&1
ips=`grep "total" alexnet.out | tail -1 | awk '{ print $3 }'`
echo "alexnet		$ips"

echo "No XLA"
${PYTHON} tf_cnn_benchmarks.py --xla=False --model resnet50 --batch_size=64 --forward_only=True > resnet50.out 2>&1
ips=`grep "total" resnet50.out | tail -1 | awk '{ print $3 }'`
echo "resnet50	$ips"

${PYTHON} tf_cnn_benchmarks.py --xla=False --model inception3 --batch_size=32 --forward_only=True > inception.out 2>&1
ips=`grep "total" inception.out | tail -1 | awk '{ print $3 }'`
echo "inception3	$ips"

${PYTHON} tf_cnn_benchmarks.py --xla=False --model vgg16 --batch_size=16 --forward_only=True > vgg.out 2>&1
ips=`grep "total" vgg.out | tail -1 | awk '{ print $3 }'`
echo "vgg16	$ips"

${PYTHON} tf_cnn_benchmarks.py --xla=False --model mobilenet --batch_size=64 --forward_only=True > mobilenet.out 2>&1
ips=`grep "total" mobilenet.out | tail -1 | awk '{ print $3 }'`
echo "mobilenet	$ips"

${PYTHON} tf_cnn_benchmarks.py --xla=False --model alexnet --batch_size=64 --forward_only=True > alexnet.out 2>&1
ips=`grep "total" alexnet.out | tail -1 | awk '{ print $3 }'`
echo "alexnet		$ips"

echo "XLA fp16"
cd ${TFCNNDIR}
${PYTHON} tf_cnn_benchmarks.py --use_fp16=True --xla=True --model resnet50 --batch_size=64 --forward_only=True > resnet50.out 2>&1
ips=`grep "total" resnet50.out | tail -1 | awk '{ print $3 }'`
echo "resnet50	$ips"

${PYTHON} tf_cnn_benchmarks.py --use_fp16=True --xla=True --model inception3 --batch_size=32 --forward_only=True > inception.out 2>&1
ips=`grep "total" inception.out | tail -1 | awk '{ print $3 }'`
echo "inception3	$ips"

${PYTHON} tf_cnn_benchmarks.py --use_fp16=True --xla=True --model vgg16 --batch_size=16 --forward_only=True > vgg.out 2>&1
ips=`grep "total" vgg.out | tail -1 | awk '{ print $3 }'`
echo "vgg16	$ips"

${PYTHON} tf_cnn_benchmarks.py --use_fp16=True --xla=True --model mobilenet --batch_size=64 --forward_only=True > mobilenet.out 2>&1
ips=`grep "total" mobilenet.out | tail -1 | awk '{ print $3 }'`
echo "mobilenet	$ips"

${PYTHON} tf_cnn_benchmarks.py --use_fp16=True --xla=True --model alexnet --batch_size=64 --forward_only=True > alexnet.out 2>&1
ips=`grep "total" alexnet.out | tail -1 | awk '{ print $3 }'`
echo "alexnet		$ips"

echo "No XLA fp16"
${PYTHON} tf_cnn_benchmarks.py --use_fp16=True --xla=False --model resnet50 --batch_size=64 --forward_only=True > resnet50.out 2>&1
ips=`grep "total" resnet50.out | tail -1 | awk '{ print $3 }'`
echo "resnet50	$ips"

${PYTHON} tf_cnn_benchmarks.py --use_fp16=True --xla=False --model inception3 --batch_size=32 --forward_only=True > inception.out 2>&1
ips=`grep "total" inception.out | tail -1 | awk '{ print $3 }'`
echo "inception3	$ips"

${PYTHON} tf_cnn_benchmarks.py --use_fp16=True --xla=False --model vgg16 --batch_size=16 --forward_only=True > vgg.out 2>&1
ips=`grep "total" vgg.out | tail -1 | awk '{ print $3 }'`
echo "vgg16	$ips"

${PYTHON} tf_cnn_benchmarks.py --use_fp16=True --xla=False --model mobilenet --batch_size=64 --forward_only=True > mobilenet.out 2>&1
ips=`grep "total" mobilenet.out | tail -1 | awk '{ print $3 }'`
echo "mobilenet	$ips"

${PYTHON} tf_cnn_benchmarks.py --use_fp16=True --xla=False --model alexnet --batch_size=64 --forward_only=True > alexnet.out 2>&1
ips=`grep "total" alexnet.out | tail -1 | awk '{ print $3 }'`
echo "alexnet		$ips"

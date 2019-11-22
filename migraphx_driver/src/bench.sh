#!/bin/bash
MIGX=${MIGX:="/home/mev/source/migraphx_sample/migraphx_driver/build/migx"}
ONNXDIR=${ONNXDIR:="/home/mev/source/migraphx_onnx"}
TFPBDIR=${TFPBDIR:="/home/mev/source/migraphx_sample/tfpb"}
OUTFILE=outfile

rocminfo=`/opt/rocm/bin/rocminfo | grep gfx | head -1 | awk '{ print $2 }'`
date=`date '+%F'`
echo
echo "Date,Status,Arch,Model,Frozen,Batch,Quant,IPS"

while read model quant batch tfpb extra
do
    if [ "${quant}" == "fp16" ]; then
	quant_option="--fp16"
    elif [ "${quant}" == "int8" ]; then
	quant_option="--int8"
    fi

    echo "DEBUG " ${MIGX} --tfpb ${TFPBDIR}/${tfpb} ${quant_option} --argname input ${extra} --perf_report
    ${MIGX} --tfpb ${TFPBDIR}/${tfpb} ${quant_option} --argname input ${extra} --perf_report > ${OUTFILE} 2>&1
    if grep "Rate: " ${OUTFILE} > lastresult; then
	rate=`awk -F'[ /]' '{ print $2 }' lastresult`
	echo "DEBUG Rate = " $rate
	imagepersec=`echo $rate \* $batch | bc`
	echo "DEBUG IPS  = " $imagepersec
	echo $date,PASS,$rocminfo,$model,"TF",$batch,$quant,$imagepersec 
    else
	imagepersec="0"
	echo $date,FAIL,$rocminfo,$model,"TF",$batch,$quant,$imagepersec 
    fi
done <<BENCHCONFIG
bert/bert_mrpc1 fp32 1 bert_mrpc1.pb --glue=MRPC --gluefile=/home/mev/source/migraphx_sample/migraphx_driver/glue/MRPC.tst
mobilenet_v2    fp32 1 mobilenet_v2i1.pb
resnet50_v1     fp32 1 resnet_v1_50i1.pb
resnet50_v2     fp32 1 resnet_v2_50i1.pb
bert/bert_mrpc1 fp16 1 bert_mrpc1.pb --glue=MRPC --gluefile=/home/mev/source/migraphx_sample/migraphx_driver/glue/MRPC.tst
mobilenet_v2    fp16 1 mobilenet_v2i1.pb
resnet50_v1     fp16 1 resnet_v1_50i1.pb
resnet50_v2     fp16 1 resnet_v2_50i1.pb
mobilenet_v2    fp32 64 mobilenet_v2i64.pb
resnet50_v1     fp32 64 resnet_v1_50i64.pb
resnet50_v2     fp32 64 resnet_v2_50i64.pb
mobilenet_v2    fp16 64 mobilenet_v2i64.pb
resnet50_v1     fp16 64 resnet_v1_50i64.pb
resnet50_v2     fp16 64 resnet_v2_50i64.pb
BENCHCONFIG

while read model quant batch onnx extra
do
    if [ "${quant}" == "fp16" ]; then
	quant_option="--fp16"
    elif [ "${quant}" == "int8" ]; then
	quant_option="--int8"
    fi

    echo "DEBUG " ${MIGX} --onnx ${ONNXDIR}/${onnx} ${quant_option} --perf_report
    ${MIGX} --onnx ${ONNXDIR}/${onnx} ${quant_option} ${extra} --perf_report > ${OUTFILE} 2>&1
    if grep "Rate: " ${OUTFILE} > lastresult; then
	rate=`awk -F'[ /]' '{ print $2 }' lastresult`
	echo "DEBUG Rate = " $rate
	imagepersec=`echo $rate \* $batch | bc`
	echo "DEBUG IPS  = " $imagepersec
	echo $date,PASS,$rocminfo,$model,"ONNX",$batch,$quant,$imagepersec 
    else
	imagepersec="0"
	echo $date,FAIL,$rocminfo,$model,"ONNX",$batch,$quant,$imagepersec 
    fi
done <<BENCHCONFIG
resnet50         fp32 1 torchvision/resnet50i1.onnx
alexnet          fp32 1 torchvision/alexneti1.onnx
densenet121      fp32 1 torchvision/densenet121i1.onnx
dpn92            fp32 1 cadene/dpn92i1.onnx
fbresnet152      fp32 1 cadene/fbresnet152i1.onnx
resnext101_64x4d fp32 1 cadene/resnext101_64x4di1.onnx
inceptionv3      fp32 1 torchvision/inceptioni1.onnx
vgg16            fp32 1 torchvision/vgg16i1.onnx
wlang/gru        fp32 1 wlang/wlang_gru.onnx --zero_input --argname=input.1
wlang/lstm       fp32 1 wlang/wlang_lstm.onnx --zero_input --argname=input.1
bert/bert_mrpc1  fp32 1 bert/bert_mrpc1.onnx --glue=MRPC --gluefile=/home/mev/source/migraphx_sample/migraphx_driver/glue/MRPC.tst
resnet50         fp16 1 torchvision/resnet50i1.onnx
alexnet          fp16 1 torchvision/alexneti1.onnx
densenet121      fp16 1 torchvision/densenet121i1.onnx
dpn92            fp16 1 cadene/dpn92i1.onnx
fbresnet152      fp16 1 cadene/fbresnet152i1.onnx
resnext101_64x4d fp16 1 cadene/resnext101_64x4di1.onnx
inceptionv3      fp16 1 torchvision/inceptioni1.onnx
vgg16            fp16 1 torchvision/vgg16i1.onnx
wlang/gru        fp16 1 wlang/wlang_gru.onnx  --zero_input --argname=input.1
wlang/lstm       fp16 1 wlang/wlang_lstm.onnx  --zero_input --argname=input.1
bert/bert_mrpc1  fp16 1 bert/bert_mrpc1.onnx  --glue=MRPC --gluefile=/home/mev/source/migraphx_sample/migraphx_driver/glue/MRPC.tst
resnet50         fp32 64 torchvision/resnet50i64.onnx
alexnet          fp32 64 torchvision/alexneti64.onnx
densenet121      fp32 32 torchvision/densenet121i32.onnx
dpn92            fp32 32 cadene/dpn92i32.onnx
fbresnet152      fp32 32 cadene/fbresnet152i32.onnx
resnext101_64x4d fp32 16 cadene/resnext101_64x4di16.onnx
inceptionv3      fp32 32 torchvision/inceptioni32.onnx
inceptionv4      fp32 16 cadene/inceptionv4i16.onnx
vgg16            fp32 16 torchvision/vgg16i16.onnx
bert/bert_mrpc8  fp32 8 bert/bert_mrpc8.onnx  --glue=MRPC --gluefile=/home/mev/source/migraphx_sample/migraphx_driver/glue/MRPC.tst
resnet50         fp16 64 torchvision/resnet50i64.onnx
alexnet          fp16 64 torchvision/alexneti64.onnx
densenet121      fp16 32 torchvision/densenet121i32.onnx
dpn92            fp16 32 cadene/dpn92i32.onnx
fbresnet152      fp16 32 cadene/fbresnet152i32.onnx
resnext101_64x4d fp16 16 cadene/resnext101_64x4di16.onnx
inceptionv3      fp16 32 torchvision/inceptioni32.onnx
inceptionv4      fp16 16 cadene/inceptionv4i16.onnx
vgg16            fp16 16 torchvision/vgg16i16.onnx
bert/bert_mrpc8  fp16 8 bert/bert_mrpc8.onnx  --glue=MRPC --gluefile=/home/mev/source/migraphx_sample/migraphx_driver/glue/MRPC.tst
BENCHCONFIG

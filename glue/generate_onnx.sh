#!/bin/bash
#
# Script to fine-tune glue tasks and export to ONNX
#
#    git clone https://github.com/huggingface/pytorch-transformers
#
# Then run this script in the pytorch-transformers/examples directory
GLUE_TASK=${GLUE_TASK:="MRPC"}
BERT_MODEL=${BERT_MODEL:="bert-base-cased"}
GLUE_DATADIR=${GLUE_DATADIR:="./glue/glue_data/${GLUE_TASK}"}
OUTPUT_DIR=${OUTPUT_DIR:="./checkpoint/${GLUE_TASK}"}

# run model to create checkpoints
python3 run_glue.py \
	--model_type bert \
	--model_name_or_path ${BERT_MODEL} \
	--task_name ${GLUE_TASK} \
	--do_eval \
	--do_train \
	--output_dir ${OUTPUT_DIR} \
	--data_dir ${GLUE_DATADIR}

python3 export_glue.py \
	--model_type bert \
	--model_name_or_path ${BERT_MODEL} \
	--task_name ${GLUE_TASK} \
	--data_dir ${GLUE_DATADIR}

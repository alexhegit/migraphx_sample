import torch
import numpy as np

from pytorch_transformers import (BertConfig, BertForSequenceClassification, BertTokenizer)

from utils_glue import (processors)

import argparse

ALL_MODELS = tuple(BertConfig.pretrained_config_archive_map.keys())

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", default='bert', type=str,
                    help="Path to pre-trained model")
parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                    help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
parser.add_argument("--data_dir", default=None, type=str, required=True,
                    help="Input data directory. Should contain .tsv files")
parser.add_argument("--task_name", default=None, type=str, required=True,
                    help="The name of the task to export selected in the list: " + ", ".join(processors.keys()))
parser.add_argument("--do_lower_case", action='store_true',
                    help="Set this flag if you are using an uncased model.")
parser.add_argument("--batch_size", default=1, type=int,
                    help="Batch size for exported model")
parser.add_argument("--max_seq_length", default=128, type=int,
                    help="Maximum sequence length after tokenization")
args = parser.parse_args()

args.task_name = args.task_name.lower()
processor = processors[args.task_name]()
label_list = processor.get_labels()
num_labels = len(label_list)

args.model_type = args.model_type.lower()
config_class = BertConfig
model_class = BertForSequenceClassification
tokenizer_class = BertTokenizer
config = config_class.from_pretrained(args.model_name_or_path, num_labels = num_labels, finetuning_task = args.task_name)
tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case = args.do_lower_case)

input_ids = torch.zeros([args.batch_size,args.max_seq_length],dtype=torch.long)
token_type_ids = torch.zeros([args.batch_size,args.max_seq_length],dtype=torch.long)

model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
torch.onnx.export(model,(input_ids,token_type_ids),'bert_'+'batch'+str(args.batch_size)+'_'+args.task_name+'.onnx')

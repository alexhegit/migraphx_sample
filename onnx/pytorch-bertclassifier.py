# Export BERT classifier model to ONNX
#
# NOTE: This script loads an (not fine-tuned) version of the BertForSequenceClassification
#       model and saves it to an ONNX file.  This particular script has not done any
#       fine-tuning work for a particular classification task.
#
#       To do this, insert the equivalent statements listed below into a script such as
#       as found in https://github.com/huggingface/pytorch-pretrained-bert/examples/run_classifier.py
#
#         input_ids=torch.zeros([args.eval_batch_size,args.max_seq_length],dtype=torch.long)
#         token_type_ids=torch.zeros([args.eval_batch_size,args.max_seq_length],dtype=torch.long)
#         onnx_filename='bertmodel_'+args.task_name.lower()
#         torch.onnx.export(model,(input_ids,token_type_ids),onnx_filename)
#
import torch
from pytorch_pretrained_bert import BertForSequenceClassification
batch_size=1
sequence_length=128

input_ids=torch.zeros([batch_size,sequence_length],dtype=torch.long)
token_type_ids=torch.zeros([batch_size,sequence_length],dtype=torch.long)
attention_mask=torch.zeros([batch_size,sequence_length],dtype=torch.long)
output_all_encoded_layers=True

model=BertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=2)

torch.onnx.export(model,(input_ids,token_type_ids),'bertmodel_classifier.onnx')

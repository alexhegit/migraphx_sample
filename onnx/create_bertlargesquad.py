import os

# Create a directory to store predict file
output_dir = "./pytorch_output"
cache_dir = "./pytorch_squad"
predict_file = os.path.join(cache_dir, "dev-v1.1.json")
# create cache dir
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

# Download the file
predict_file_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
if not os.path.exists(predict_file):
    import wget
    print("Start downloading predict file.")
    wget.download(predict_file_url, predict_file)
    print("Predict file downloaded.")

# Define some variables
model_type = "bert"
model_name_or_path = "bert-large-cased"
max_seq_length = 128
doc_stride = 128
max_query_length = 64
per_gpu_eval_batch_size = 1
eval_batch_size = 1
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The following code is adapted from HuggingFace transformers
# https://github.com/huggingface/transformers/blob/master/examples/run_squad.py#L290

from transformers import (WEIGHTS_NAME, BertConfig, BertForQuestionAnswering, BertTokenizer)
from torch.utils.data import (DataLoader, SequentialSampler)

# Load pretrained model and tokenizer
config_class, model_class, tokenizer_class = (BertConfig, BertForQuestionAnswering, BertTokenizer)
config = config_class.from_pretrained(model_name_or_path, cache_dir=cache_dir)
tokenizer = tokenizer_class.from_pretrained(model_name_or_path, do_lower_case=True, cache_dir=cache_dir)
model = model_class.from_pretrained(model_name_or_path,
                                    from_tf=False,
                                    config=config,
                                    cache_dir=cache_dir)
# load_and_cache_examples
from transformers.data.processors.squad import SquadV2Processor

processor = SquadV2Processor()
examples = processor.get_dev_examples(None, filename=predict_file)

from transformers import squad_convert_examples_to_features
features, dataset = squad_convert_examples_to_features(
            examples=examples,
	                tokenizer=tokenizer,
			            max_seq_length=max_seq_length,
				                doc_stride=doc_stride,
						            max_query_length=max_query_length,
							                is_training=False,
									            return_dataset='pt'
										            )

cached_features_file = os.path.join(cache_dir, 'cached_{}_{}_{}'.format(
        'dev',
	        list(filter(None, model_name_or_path.split('/'))).pop(),
		        str(384))
			    )

torch.save({"features": features, "dataset": dataset}, cached_features_file)
print("Saved features into cached file ", cached_features_file)

# create output dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

n_gpu = torch.cuda.device_count()
# eval_batch_size = 8 * max(1, n_gpu)

eval_sampler = SequentialSampler(dataset)
eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=eval_batch_size)

# multi-gpu evaluate
if n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
    model = torch.nn.DataParallel(model)

# Eval!
print("***** Running evaluation {} *****")
print("  Num examples = ", len(dataset))
print("  Batch size = ", eval_batch_size)

output_model_path = './pytorch_squad/bert-large-cased-squad.onnx'
inputs = {}
outputs= {}
# Get the first batch of data to run the model and export it to ONNX
batch = dataset[0]

# Set model to inference mode, which is required before exporting the model because some operators behave differently in
# inference and training mode.
model.eval()
batch = tuple(t.to(device) for t in batch)
inputs = {
    'input_ids':      batch[0].reshape(1, 128),                         # using batch size = 1 here. Adjust as needed.
    'attention_mask': batch[1].reshape(1, 128),
    'token_type_ids': batch[2].reshape(1, 128)
}

with torch.no_grad():
    symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
    torch.onnx.export(model,                                            # model being run
                      (inputs['input_ids'],                             # model input (or a tuple for multiple inputs)
                       inputs['attention_mask'],
                       inputs['token_type_ids']),
                      output_model_path,                                # where to save the model (can be a file or file-like object)
                      opset_version=11,                                 # the ONNX version to export the model to
                      do_constant_folding=True,                         # whether to execute constant folding for optimization
                      input_names=['input_ids',                         # the model's input names
                                   'input_mask',
                                   'segment_ids'],
                      output_names=['start', 'end'],                    # the model's output names
                      dynamic_axes={'input_ids': symbolic_names,        # variable length axes
                                    'input_mask' : symbolic_names,
                                    'segment_ids' : symbolic_names,
                                    'start' : symbolic_names,
                                    'end' : symbolic_names})
    print("Model exported at ", output_model_path)

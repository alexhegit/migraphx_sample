Scripts and data for running the GLUE benchmark as well as creating BERT ONNX files.

download_glue_data.py - This script downloads the GLUE benchmark files to ./glue_data
   Various options are possible including changing where the files are saved.

generate_onnx.sh - This script runs BERT model to fine-tune a task and then exports an ONNX file.
   run_glue.py - is the Python program that runs BERT for fine tuning.  Results are saved to a checkpoint file.
   export_glue.py - is the Python program that exports the saved checkpoint to an ONNX file
   utils_glue.py - is utility file copied from pytorch-transformers

glue_dump.sh - This script exports glue_data to show TSV file components for various subsets

bert_tokenize.py - example script that shows how the BERT tokenizer is called.

NOTE: run_glue, utils_glue.py are unchanged copies from github.com/pytorch-transformers/examples directory.

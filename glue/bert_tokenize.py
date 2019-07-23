from pytorch_transformers import BertTokenizer

# Note: A more complete example is the "convert_examples_to_features" in the utils_glue.py script.

tokenizer = BertTokenizer.from_pretrained("bert-base-cased",do_lower_case=False)
tokens = tokenizer.tokenize("The rain in Spain falls mainly on the plain.")
token_ids = list(map(tokenizer.convert_tokens_to_ids,tokens))
print(token_ids)

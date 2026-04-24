import data
import statistics
import pandas as pd

from util import *
from code_train import CrossAttentionTransformer

data_to_params_ratio = 20
max_samples_count_params = 1_000


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Create model instance
model = CrossAttentionTransformer()
model = model.to(device)

# Load the saved state dict
model.load_state_dict(torch.load(code_model_name))
model.eval()

param_count = count_parameters(model)
token_count = data_to_params_ratio * param_count
df = pd.read_pickle(dataset_name)
df = df.head(max_samples_count_params)

lex_merges, ast_merges = data.get_merges()
# Should use the "usable" tokens here, therefore it encodes, rather than simply the x_in and x_out,
# because it has the pad tokens, that effectively have 0 information.
lens1 = [len(data.encode(row['ast_text'], ast_merges)) / (len(row['ast_text'].split('\n')) - 1) for _, row in df.iterrows()]
avg_tokens_per_sentence = statistics.mean(lens1)

print(f'Total model params: {param_count}')
print(f'Should train on {token_count} tokens')
sentences = round(token_count/avg_tokens_per_sentence)
print(f'Should train on {sentences} sentences')
print(f'Should train for {round(sentences / batch_size)} iterations')
print(f'stats are: {avg_tokens_per_sentence=}, {statistics.stdev(lens1)=}, {max(lens1)=}, {min(lens1)=}')

# cut below which 99.7% of samples lie.
above_3_std_dev = avg_tokens_per_sentence + 3*statistics.stdev(lens1)
print(f'block_size should be: {min(above_3_std_dev, max(lens1))}')

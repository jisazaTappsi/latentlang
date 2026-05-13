import os
import json
import pandas as pd

from .util import *
from .tokens import *

lex_vocab_size = 256 + len(TOKENS)  # Source (Lexer) vocabulary size
print('lex_vocab_size:', lex_vocab_size)
ast_vocab_size = 256 + len(TOKENS)  # Target (AST) vocabulary size
max_merge_samples = 10_000


def get_max_pair(ids):
    counts = {}
    max_pair = None
    max_pair_count = 0
    for key in zip(ids[:-1], ids[1:]):
        count = counts.get(key, 0) + 1
        counts[key] = count
        if max_pair_count < count:
            max_pair_count = count
            max_pair = key
    return max_pair, max_pair_count

def get_stats(ids):
    counts = {}
    for pair in zip(ids[:-1], ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and pair[0] == ids[i] and pair[1] == ids[i + 1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


def get_merge_filename(input_type):
    return f'merges_{input_type}.json'


def save_merge_to_file(input_type, my_merges):
    # Convert tuple keys to strings for JSON
    with open(get_merge_filename(input_type), 'w') as outfile:
        json.dump({f'{k[0]},{k[1]}': v for k, v in my_merges.items()}, outfile)


def train_merges(toks, input_type, target_vocab_size=280):
    ids = toks.copy()
    idx = 256 + len(TOKEN_BYTES)
    my_merges = {}  # pair => idx
    for i in range(target_vocab_size - idx + 1):
        pair, pair_count = get_max_pair(ids)
        decoded_pair = decode(pair, my_merges)
        print(f'Merging "{decode([pair[0]], my_merges)}", "{decode([pair[1]], my_merges)}" into: "{decoded_pair}"')
        ids = merge(ids, pair, idx=idx)
        my_merges[pair] = idx
        idx += 1

    print(f'tokens len {len(toks)}')
    print(f'ids len: {len(ids)}')
    print(f'compression: {round(len(toks) / len(ids), 2)}')

    save_merge_to_file(input_type, my_merges)
    return my_merges

def decode(ids, my_merges):
    vocab = get_vocab(my_merges)
    toks = b''.join(vocab[i] for i in ids if i not in (TOKEN_IDS[SOF], TOKEN_IDS[EOF], TOKEN_IDS[PAD]))
    return toks.decode('utf-8', errors='replace')

def encode(string, my_merges):
    toks = get_compressed_tokens(string)
    while len(toks) > 1:
        stats = get_stats(toks)
        pair = min(stats, key=lambda p: my_merges.get(p, float('inf')))
        if pair in my_merges:
            toks = merge(toks, pair, idx=my_merges[pair])
        else:
            break
    return toks

def get_vocab(merges):
    vocab = {idx: bytes([idx]) for idx in range(256)}
    vocab |= TOKEN_BYTES
    for (p0, p1), idx  in merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]
    return vocab


def get_merge(input_type):
    return {}
    with open(get_merge_filename(input_type), 'r') as file:
        data = json.load(file)
        # Convert string keys back to tuples
        return {tuple(map(int, k.split(','))): v for k, v in data.items()}


def get_merges():
    try:
        return get_merge('lex'), get_merge('ast')
    except FileNotFoundError:
        return save_code_merges()


def get_compressed_tokens(text):
    tokens_text = ','.join(str(int(i)) for i in text.encode('utf-8'))
    for token_id, token  in TOKEN_BASIC_IDS.items():
        tokens_text = tokens_text.replace(token, str(token_id))
    return [int(i) for i in tokens_text.split(',')]


def load_code_tokens():
    df = pd.read_pickle(DATASET_NAME)
    train_df = get_train_data(df).head(max_merge_samples)

    lex_text = ' '.join(train_df['lex_text'].tolist())
    ast_text = ' '.join(train_df['ast_text'].tolist())

    lex_tokens = get_compressed_tokens(lex_text)
    ast_tokens = get_compressed_tokens(ast_text)
    print(f'Lex tokens length: {len(lex_tokens)}')
    print(f'AST tokens length: {len(ast_tokens)}')
    return lex_tokens, ast_tokens


def save_code_merges():
    try:
        # When recalculating merges needs to delete the model first, as it will lose the encoding :(
        os.remove(MODEL_NAME)
    except FileNotFoundError:
        pass

    lex_tokens, ast_tokens = load_code_tokens()
    lex_merges = train_merges(lex_tokens, 'lex', target_vocab_size=ast_vocab_size)
    ast_merges = train_merges(ast_tokens, 'ast', target_vocab_size=lex_vocab_size)
    return lex_merges, ast_merges


def save_code_merges_dummy():
    save_merge_to_file('lex', {})
    save_merge_to_file('ast', {})


def add_pad_tokens_and_trim(ids, truncate_size):
    ids = ids[:truncate_size]  # Truncate/pad to block_size
    return ids + [TOKEN_IDS[PAD]] * max(0, truncate_size - len(ids))


def df_to_tensors(df):
    """Convert DataFrame with x_in/x_out columns to stacked tensors (one-time conversion)."""
    x_out = torch.tensor([row[:BLOCK_SIZE] for statement in df['x_out'] for row in statement], dtype=torch.long)
    x_in = torch.tensor([row[:BLOCK_SIZE] for statement in df['x_in'] for row in statement], dtype=torch.long)
    y = torch.tensor([row[1:BLOCK_SIZE + 1] for statement in df['x_out'] for row in statement], dtype=torch.long)
    return {'x_out': x_out, 'x_in': x_in, 'y': y}


def get_train_data(iterable):
    n = int(TRAIN_SPLIT_RATIO * len(iterable))
    return iterable[:n]


def get_val_data(iterable):
    n = int(TRAIN_SPLIT_RATIO * len(iterable))
    return iterable[n:]


def get_code_data():
    """We translate from Lex to AST, ie our x_in=LEX, while x_out=AST"""
    df = pd.read_pickle(DATASET_NAME)
    lex_merges, ast_merges = get_merges()

    train_df = get_train_data(df)
    val_df = get_val_data(df)
    train_tensors = df_to_tensors(train_df)
    val_tensors = df_to_tensors(val_df)
    print(f'Training samples: {len(train_df)}, Validation samples: {len(val_df)}')
    return {
        'train': train_tensors,
        'val': val_tensors
    }, lex_merges, ast_merges


if __name__ == '__main__':
    save_code_merges()

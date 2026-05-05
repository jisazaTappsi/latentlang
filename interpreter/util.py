import torch

PRODUCTION_RUN = True
NUM_TRAINING_SAMPLES = 1_000_000 if PRODUCTION_RUN else 100_000
BATCH_SIZE = 64  # 32
BLOCK_SIZE = 64  # 256
MAX_ITERS = 35_001 if PRODUCTION_RUN else 5_001
EVAL_INTERVAL = 500
LR_PEAK = 8e-4  # peak learning rate
LR_MIN = 1e-4  # min learning rate
WARMUP_ITERS = 0.03 * MAX_ITERS   # linear warmup steps (e.g. ~3% of MAX_ITERS)
EVAL_ITERS = 200
DROPOUT = 0.2
N_HEAD = 3  # 4
N_EMBED = 64 * N_HEAD  # 32
TRAIN_SPLIT_RATIO = 0.8
MODEL_NAME = 'interpreter/model.pth'
DATASET_NAME = 'interpreter/dataset.pkl'


if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

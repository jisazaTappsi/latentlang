import math
import torch.nn as nn
from torch.nn import functional as F

from . import data
from .util import *
from .tokens import *

torch.manual_seed(42)


def get_batch(my_data):
    """my_data is a dict of tensors: x_out, x_in, y each (N, BLOCK_SIZE)."""
    size = my_data['x_out'].size(0)
    ix = torch.randint(size, (BATCH_SIZE,))
    return {
        'x_out': my_data['x_out'][ix].to(device),
        'x_in': my_data['x_in'][ix].to(device),
        'y': my_data['y'][ix].to(device)
    }


@torch.no_grad()
def estimate_loss(my_data, model):
    out = {}
    model.eval()
    for split, data_tensor in my_data.items():
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            batch_dict = get_batch(data_tensor)
            logits, loss = model(**batch_dict)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class HeadCross(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBED, head_size, bias=False)
        self.query = nn.Linear(N_EMBED, head_size, bias=False)
        self.value = nn.Linear(N_EMBED, head_size, bias=False)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x_out, x_in=None):
        B, T, C = x_out.shape
        if x_in is not None:
            queries = self.query(x_out)
            keys = self.key(x_in)
            values = self.value(x_in)
        else:  # self attention
            queries = self.query(x_out)
            keys = self.key(x_out)
            values = self.value(x_out)

        wei = queries @ keys.transpose(-2, -1) * C ** -0.5  # B, T, head_size @ B, head_size, T = B, T, T
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        return wei @ values  # (B, T, T) @ (B, T, head_size) = B, T, head_size


class MaskedHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBED, head_size, bias=False)
        self.query = nn.Linear(N_EMBED, head_size, bias=False)
        self.value = nn.Linear(N_EMBED, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x_out):
        B, T, C = x_out.shape
        keys = self.key(x_out)
        queries = self.query(x_out)

        wei = queries @ keys.transpose(-2, -1) * C ** -0.5  # B, T, head_size @ B, head_size, T = B, T, T
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # B, T, T
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        values = self.value(x_out)
        return wei @ values  # (B, T, T) @ (B, T, head_size) = B, T, head_size


class MultiAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([HeadCross(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(N_EMBED, N_EMBED)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x_out, x_in=None):
        out = torch.cat([h(x_out, x_in) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class MaskedMultiAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([MaskedHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(N_EMBED, N_EMBED)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x_out):
        out = torch.cat([h(x_out) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x_out):
        return self.net(x_out)


class BlockCross(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.masked_multi = MaskedMultiAttention(n_head, head_size=n_embed//n_head)
        self.ln2 = nn.LayerNorm(n_embed)
        self.multi = MultiAttention(n_head, head_size=n_embed//n_head)
        self.ln3 = nn.LayerNorm(n_embed)
        self.ffwd = FeedForward(n_embed)

        self.ln1_cross = nn.LayerNorm(n_embed)
        self.multi_cross = MultiAttention(n_head, head_size=n_embed//n_head)
        self.ln2_cross = nn.LayerNorm(n_embed)
        self.ffwd_cross = FeedForward(n_embed)

    def forward(self, x_out, x_in):
        # Small left branch in "Attention is all you need" paper
        x_in = x_in + self.multi_cross(self.ln1_cross(x_in))  # Operates on self-attention mode
        x_in = x_in + self.ffwd_cross(self.ln2_cross(x_in))

        # Big main branch in "Attention is all you need" paper
        x_out = x_out + self.masked_multi(self.ln1(x_out))  # Operates on self-attention mode
        x_out = x_out + self.multi(self.ln2(x_out), x_in)  # Operates on cross-attention mode
        x_out = x_out + self.ffwd(self.ln3(x_out))
        return x_out, x_in


class CrossAttentionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # Main branch embeddings
        self.emb_table = nn.Embedding(num_embeddings=data.ast_vocab_size, embedding_dim=N_EMBED)
        self.position_emb_table = nn.Embedding(num_embeddings=BLOCK_SIZE, embedding_dim=N_EMBED)

        # Cross branch embeddings
        self.emb_table_cross = nn.Embedding(num_embeddings=data.lex_vocab_size, embedding_dim=N_EMBED)
        self.position_emb_table_cross = nn.Embedding(num_embeddings=BLOCK_SIZE, embedding_dim=N_EMBED)

        self.blocks = nn.ModuleList([
            BlockCross(N_EMBED, n_head=N_HEAD),
            BlockCross(N_EMBED, n_head=N_HEAD),
            BlockCross(N_EMBED, n_head=N_HEAD),
        ])
        self.ln_final = nn.LayerNorm(N_EMBED)
        self.lm_head = nn.Linear(N_EMBED, data.ast_vocab_size)

    @staticmethod
    def get_embedding(x, token_table, position_table):
        B, T = x.shape
        tok_emb = token_table(x)  # (B, T, n_embed)
        pos_emb = position_table(torch.arange(T, device=device))
        return tok_emb + pos_emb

    def forward(self, x_out, x_in, y=None):
        emb = CrossAttentionTransformer.get_embedding(x_out, token_table=self.emb_table,
                                                      position_table=self.position_emb_table)
        emb_cross = CrossAttentionTransformer.get_embedding(x_in, token_table=self.emb_table_cross,
                                                            position_table=self.position_emb_table_cross)

        # Manually iterate through blocks to pass both inputs
        block_out = emb
        block_cross_out = emb_cross
        for block in self.blocks:
            block_out, block_cross_out = block(block_out, block_cross_out)

        logits = self.lm_head(self.ln_final(block_out))  # (B, T, vocab_size)

        if y is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            target = y.view(B*T)
            loss = F.cross_entropy(logits, target)

        return logits, loss

    def generate(self, x_out, x_in=None, max_new_tokens=100):
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                x_window = x_out[:, -BLOCK_SIZE:]
                logits, loss = self(x_out=x_window, x_in=x_in)
                logits = logits[:, -1, :]  # last element in T dim (B, C)
                probs = F.softmax(logits, dim=-1)
                x_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
                x_out = torch.cat((x_out, x_next), dim=1)  # (B, T+1)
                if x_next.item() == TOKEN_IDS[EOF]:
                    break
        self.train()
        return x_out

    @staticmethod
    def fix_unmatched_parenthesis(text):
        count_lparen = text.count('(')
        count_rparen = text.count(')')
        if count_lparen > count_rparen:  # balance on right
            text = text.strip() + ')' * (count_lparen - count_rparen)
        elif count_lparen < count_rparen:  # balance on left
            text = (count_rparen - count_lparen) * '(' + text.strip()

        return text.strip()

    def inference(self, x_in: list, ast_merges: dict):
        with torch.no_grad():
            generated = self.generate(x_out=torch.tensor([[TOKEN_IDS[SOF]]], dtype=torch.long, device=device),
                                      x_in=torch.tensor([x_in], dtype=torch.long, device=device),
                                      max_new_tokens=BLOCK_SIZE)[0].tolist()
            predicted_ast_text = data.decode(generated, ast_merges)
            return CrossAttentionTransformer.fix_unmatched_parenthesis(predicted_ast_text)


def get_lr(step, learning_rate_decay=True):
    if not learning_rate_decay:
        return LR_MIN
    if step < WARMUP_ITERS:
        return LR_PEAK * (step + 1) / WARMUP_ITERS  # avoid 0 on first step
    decay_iters = MAX_ITERS - WARMUP_ITERS
    progress = (step - WARMUP_ITERS) / decay_iters
    min_ratio = LR_MIN / LR_PEAK
    return LR_PEAK * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def train():
    dataset, lex_merges, ast_merges = data.get_code_data()
    model = CrossAttentionTransformer()
    model = model.to(device)

    try:
        model.load_state_dict(torch.load(MODEL_NAME))
        print('training an existing model')
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MIN)
        scheduler = None
        learning_rate_decay = False
    except FileNotFoundError:
        print('Creating model from scratch')
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR_PEAK)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: get_lr(step) / LR_PEAK)
        learning_rate_decay = True

    last_losses = None
    for iter in range(MAX_ITERS):
        if iter % EVAL_INTERVAL == 0:
            last_losses = estimate_loss(dataset, model)
            print(f"step {iter} train loss: {last_losses['train']:.4f}, val loss: {last_losses['val']:.4f}, "
                  f"lr: {get_lr(iter, learning_rate_decay):.2e}")
        batch_dict = get_batch(dataset['train'])
        logits, loss = model(**batch_dict)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

    # Save the model after training
    torch.save(model.state_dict(), MODEL_NAME)
    print(f"Model saved to {MODEL_NAME}")
    return last_losses['val'].item()


if __name__ == '__main__':
    train()

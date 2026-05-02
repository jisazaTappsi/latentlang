# LatentLang

**The language of latent intent.**

LatentLang features a trainable AI interpreter and a non-rigid, free-variable syntax that adapts to your mental model. No more fighting rigid compilers: Latent learns your methods and executes your logic with fluid precision. Define the variables, train the core, and watch your intent manifest as code.

Try these examples on the web REPL here: www.latentlang.com:
```
> var my_var = 12        # JS var
12
> my_var = 12            # Python-style, no `var`
12
> 3 plus 4               # natural language
7
> 3.times 4              # method-style
12
> mul(8 8)               # no commas
64
> sum(3,4)               # classic call
7
```

Every line above is a valid program to LatentLang. The AI compiler translates whatever shape your intent takes into one canonical AST, then executes it.

---

## Why LatentLang

Traditional interpreters and compilers reject anything that doesn't match their grammar to the character. LatentLang does the opposite: it treats the grammar as a *target*, not a *gate*. A Cross-Attention Transformer learns the mapping from **free-form source** (your mental model) to a **rigid AST** (the executable core).

- **Free-variable syntax** — write `3 plus 4`, `3 + 4`, `plus(3,4)`, `3.plus(4)`; all valid.
- **Trainable core** — the compiler is a Cross-Attention Transformer. Teach it new methods by adding templates and retraining.
- **Deterministic execution** — once the AST is produced, a classical tree-walking interpreter runs it. No hallucinated results.
- **Graceful fallback** — a hand-written parser tries first; the AI only kicks in when the input doesn't fit the formal grammar.

This flexibility enables LatentLang to potentially learn millions of functions and build high level languages that get closer to English. 


## LatentLang Potential

LatentLang is an MVP for a new way of building interpreters and compilers. The same architecture could ingest the entire PyPI ecosystem and train a single model that understands every known Python library as a first-class language feature — opening the door to a new breed of high-level languages that sit much closer to English than anything available today.


## Intuition behind LatentLang

LatentLang grew out of a simple observation: most bugs are not deep logical failures, they are small ambiguities — places where the programmer's intent is clear to a human reader but not quite aligned with what the compiler accepts. Today's toolchains treat every such ambiguity as an outright error. A neural compiler can instead *resolve* it — reading the surrounding intent the way another engineer would, and producing the canonical form the machine needs to execute.


---

## Architecture

```
     source text
         │
         ▼
 ┌───────┴───────┐        
 │ Lexer/Parser  │── syntax error ───┐ 
 └───────┬───────┘                   │
         │                           ▼
         │ AST ok            ┌───────┴───────┐
         │                   │ AI Transformer│
         │                   └───────┬───────┘   
         │◀──────── AST ok ──────────┘
 ┌───────┴──────────────┐
 │ Classical Interpreter│ 
 └───────┬──────────────┘
         │
         ▼
     result
```

The AI interpreter is a **Cross-Attention Transformer** (encoder-decoder) trained to translate a tokenized *Lex stream* (the user's free-form source) into a tokenized *AST stream* (a canonical, parseable expression tree). The encoder cross-attends to the raw lex input.

### Key components

| File                            | Role |
|---------------------------------|---|
| `interpreter/tokens.py`         | Token types, keyword list, and byte-ID reservations. |
| `interpreter/basic.py`          | Lexer, parser, AST nodes, interpreter, symbol table, error types, `run_ai()` entry point. |
| `interpreter/grammar.txt`       | Formal grammar the AST targets. |
| `interpreter/data.py`           | Byte-pair-encoding (BPE) merges over the custom token space; batching into tensors. |
| `interpreter/data_generator.py` | Synthesizes `(lex_text, ast_text)` pairs using `FUNC_TEMPLATES` — the dictionary of methods the AI learns. |
| `interpreter/train_module.py`   | `CrossAttentionTransformer` definition and training loop. |
| `interpreter/train_pipeline.py` | End-to-end: generate data → train → evaluate. |
| `interpreter/samples.py`        | Scores the trained model: per-sample AST match and symbol-table (computation) match. |
| `interpreter/shell.py`          | Interactive REPL for local use. |
| `interpreter/tests.py`          | Lexer/parser/interpreter tests plus multi-style arithmetic tests. |
| `frontend/`                     | Nuxt 3 + Vue 3 REPL served statically to S3 + CloudFront. |
| `backend/`                      | FastAPI service exposing `POST /interpret`. |

---

## The model

At present the model is a tiny 3M parameters LLM. It can be scaled at will and I invite the reader to do so, just **see hyperparameters in: (`interpreter/util.py`)**


`interpreter/train_module.py` defines `CrossAttentionTransformer` — an encoder-decoder built from scratch in the style of *Attention Is All You Need*, specialized to map a **lex-token stream** to an **AST-token stream**.

**Two vocabularies, two embeddings.** Input (lex) and output (AST) have independent BPE-merged vocabularies and their own token + positional embedding tables. The model never has to disambiguate surface syntax from canonical syntax — they live in different token spaces.

**`BlockCross` — the repeated unit.** Each layer updates both streams:

- *Encoder branch* (lex): self-attention → feed-forward, pre-LayerNorm + residuals.
- *Decoder branch* (AST): masked self-attention → cross-attention over the encoder output → feed-forward.

Three `BlockCross` layers are stacked; a final LayerNorm and linear head project the decoder stream onto the AST vocabulary.


Flip `PRODUCTION_RUN = True` in `util.py` to scale to 1M training samples and 35k iterations.

**Inference:** Start from a single `SOF` (Start of file) token, then predicts until `EOF` (End of file) or `BLOCK_SIZE` tokens are reached. A post-step balances any unmatched parentheses so the emitted AST text is always parseable by the classical interpreter.

**Checkpointing:** `train_module.py` loads `model.pth` when present and continues at a flat `LR_MIN`; otherwise it initializes from scratch with the full warmup-cosine schedule.

---

## Grammar and basic.py (the AST target)

A very small interpreter and grammar was built as the foundation upon which the model learns more complex representations: see `grammar.txt` and `basic.py`.

---

## The method dictionary

`data_generator.py` defines `FUNC_TEMPLATES` — a list of `(name, params, body)` tuples like:

```python
("sum",    ["a", "b"],      "a+b"),
("times",  ["a", "b"],      "a*b"),
("halve",  ["x"],           "x/2"),
("cube",   ["x"],           "x*x*x"),
("mid3",   ["a", "b", "c"], "(a+b+c)/3"),
...
```

These templates:
1. Seed data generation — every training pair uses them in varied surface forms.
2. Become live functions in the symbol table at runtime, so the classical interpreter can execute the AST the model emits.
3. Are learned as standard methods and also as **infix operators** (`7 times 8` → `56`).

Add a template, regenerate data, retrain, and the interpreter has learned new syntax.

---

## Quickstart

### 1. Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requires PyTorch (CUDA, MPS, or CPU — auto-detected in `util.py`), FastAPI, and pandas.

### 2. Train

```bash
python -m interpreter.train_pipeline
```

This will:
1. Generate `NUM_TRAINING_SAMPLES` synthetic `(lex, ast)` pairs using `FUNC_TEMPLATES`.
2. Train `CrossAttentionTransformer` for `MAX_ITERS` steps (see `util.py` for defaults).
3. Score the trained model on held-out validation samples.

Outputs:
- `model.pth` — model weights.
- `dataset.pkl` — training data.

Re-running loads the existing checkpoint and fine-tunes further.

### 3. Run the REPL

```bash
python -m interpreter.shell
```

```
basic > sum(3, 4)
7
basic > 3 plus 4
7
basic > var x = halve(20)
10
basic > x times x
100
```

### 4. Run the API + web REPL

Backend:

```bash
uvicorn backend.backend:app --port 9000 --reload
```

Frontend:

```bash
cd frontend
npm install
npm run dev -- --port 4000  # http://localhost:4000
```

The Nuxt app calls `POST /interpret` with `{ code, symbols }` and keeps the symbol table client-side across turns.

---

## Evaluation metrics

`samples.py` reports two numbers over held-out validation samples:

- **Tree accuracy** — fraction of programs whose *every* statement decodes to the exact target AST.
- **Computation accuracy** — fraction of programs whose post-execution symbol table matches the target's, even if the AST differs syntactically.

Computation accuracy is the more forgiving (and more useful) metric: we care that the model captures intent, not that it picks the same surface form.

`train_pipeline.py` additionally emits a *time-complexity score*:

```
time-complexity score            = 1 / (val_loss * minutes)
time-complexity computation score = computation_pct / minutes
```

A rough throughput number for comparing training configurations.

---

## Deployment

### Frontend → S3 + CloudFront (static site)

```bash
cd frontend
python scripts/deploy.py --bucket YOUR_BUCKET --distribution YOUR_DIST_ID
```

Point `NUXT_PUBLIC_API_BASE` at the backend URL when building.

### Backend → uvicorn behind Caddy

The backend is a FastAPI app served by `uvicorn` and fronted by [Caddy](https://caddyserver.com/) as a reverse proxy (Caddy handles TLS automatically in production).

Set `CORS_ORIGINS` to a comma-separated list of allowed origins (e.g. the CloudFront domain and `https://www.latentlang.com`).

**Run uvicorn** (bind to localhost; Caddy handles the public port):

```bash
CORS_ORIGINS="https://www.latentlang.com" \
uvicorn backend.backend:app --host 127.0.0.1 --port 9000
```

For production, run uvicorn under a process manager (`systemd`, `supervisord`, `pm2`, …) so it restarts on failure and at boot.

**Run Caddy** — two configs are provided:

- `backend/Caddyfile` — production. Serves `api.latentlang.com` and reverse-proxies to `127.0.0.1:9000`. Caddy provisions a Let's Encrypt cert automatically (ports 80/443 must be reachable and DNS for `api.latentlang.com` must point at the host).
- `backend/Caddyfiledev` — local. Listens on `:8080` with no TLS, useful for testing the proxy path before deploying.

```bash
# production
sudo caddy run --config backend/Caddyfile

# local dev
caddy run --config backend/Caddyfiledev
```

In production, run Caddy via its bundled systemd unit (`systemctl enable --now caddy`) pointed at `backend/Caddyfile` so it survives reboots.

---

## Tests

```bash
pytest interpreter/tests.py
```

Covers the lexer, parser, classical interpreter, function defs, and the multi-style arithmetic pass (`sum(3,4)`, `3 plus 4`, `3.times 4`, `mul(8 8)`, …) going through `run_ai()`.

---

## Extending the language

1. **Add a method**: append to `FUNC_TEMPLATES` in `data_generator.py` with a short arithmetic body.
2. **Add a syntactic style**: extend the surface-form generator in `data_generator.py` so the pair `(new_style, canonical_ast)` shows up in training.
3. **Retrain**: `python -m interpreter.train_pipeline`.
4. **Use it**: the REPL will now understand the new form.

No changes to the interpreter are needed: The grammar stays fixed; the model builds higher order methods.

---

## License

See repository root. Contributions welcome :)

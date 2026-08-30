# BUILDING AN LLM FROM SCRATCH — 2026

A compact, educational GPT-style language model built from first principles with PyTorch, including an interactive chat prototype.

This repository covers the complete pipeline:

1. raw text → byte tokens
2. token + positional embeddings
3. causal multi-head self-attention
4. feed-forward network
5. residual connections + LayerNorm
6. stacked Transformer blocks
7. next-token cross-entropy training
8. autoregressive text generation
9. checkpoint save/load
10. browser chat prototype + real local checkpoint chat
11. tests and a GitHub Pages learning site

> The code is intentionally small enough to study, modify, and train on a laptop/GPU. It is an educational small language model, not a production-scale ChatGPT replacement.

## Architecture

```text
text
  ↓
UTF-8 bytes (0..255)
  ↓
Token Embedding + Position Embedding
  ↓
[ LayerNorm → Masked Multi-Head Attention → Residual
  LayerNorm → MLP/GELU → Residual ] × N
  ↓
LayerNorm
  ↓
Linear vocabulary head
  ↓
next-token logits
```

## Quick start

```bash
git clone https://github.com/KrAzad0/BUILDING-AN-LLM-FROM-SCRATCH-2026.git
cd BUILDING-AN-LLM-FROM-SCRATCH-2026
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Train:

```bash
python train.py --data data/sample.txt --steps 1000
```

Generate:

```bash
python generate.py --checkpoint checkpoints/minigpt.pt --prompt "Quantum mechanics"
```

Run tests:

```bash
pytest -q
```

## Chat prototype

The project now includes a ChatGPT-style interface at:

```text
docs/chat.html
```

It has two operating modes.

### 1. GitHub Pages / browser mode

Open `chat.html` through the GitHub Pages site. The interface works entirely in the browser with:

- conversation history stored in `localStorage`
- streaming/typing animation
- byte-token counter
- temperature and top-k controls
- suggested prompts
- responsive desktop/mobile chat UI
- local educational fallback response engine

### 2. Real PyTorch checkpoint mode

After training a checkpoint, launch:

```bash
python chat_server.py
```

Then open:

```text
http://127.0.0.1:8000/
```

The same UI automatically detects the local API and switches from the browser fallback to the actual PyTorch `minigpt.pt` checkpoint.

You can select another checkpoint or device:

```bash
python chat_server.py \
  --checkpoint checkpoints/minigpt.pt \
  --device auto \
  --port 8000
```

The local API exposes:

```text
GET  /api/status
POST /api/chat
```

GitHub Pages cannot execute Python, so real checkpoint inference requires the local server for now. A future ONNX/WebGPU export can make genuine model inference run directly inside the browser.

## Train on your own corpus

Put any UTF-8 `.txt` file in `data/` and run:

```bash
python train.py --data data/my_corpus.txt --steps 5000 --batch-size 32 --block-size 128
```

For a larger experiment:

```bash
python train.py \
  --data data/my_corpus.txt \
  --steps 20000 \
  --batch-size 64 \
  --block-size 256 \
  --n-layer 8 \
  --n-head 8 \
  --n-embd 512 \
  --learning-rate 3e-4
```

## Project structure

```text
.
├── src/
│   ├── __init__.py
│   ├── tokenizer.py
│   └── minigpt.py
├── data/sample.txt
├── tests/test_model.py
├── docs/
│   ├── index.html
│   ├── style.css
│   ├── app.js
│   ├── chat.html
│   ├── chat.css
│   └── chat.js
├── .github/workflows/pages.yml
├── train.py
├── generate.py
├── chat_server.py
├── requirements.txt
└── README.md
```

## What is implemented here?

PyTorch supplies tensors and autograd, while the language-model architecture is implemented directly in this repository rather than imported from Hugging Face:

- reversible UTF-8 byte tokenizer
- causal attention mask
- scaled dot-product attention
- multi-head attention and projection
- Transformer blocks
- token/position embeddings
- language-modeling loss
- temperature + top-k sampling
- checkpointing and generation loop
- chat-style local inference server
- browser chat interface with local fallback mode

## The mathematics

For hidden states `X`, a self-attention head learns queries, keys and values:

```text
Q = XWq
K = XWk
V = XWv
Attention(Q,K,V) = softmax(QKᵀ / √d + causal_mask)V
```

The causal mask sets future positions to `-∞`, so token `t` can only use tokens at positions `≤ t`.

Training minimizes next-token cross entropy:

```text
L = - Σ log p(x[t+1] | x[0:t])
```

Generation repeatedly samples a new token from the model distribution and appends it to the context.

## Default model

- vocabulary: 256 bytes
- context length: 128
- layers: 4
- heads: 4
- embedding dimension: 256
- dropout: 0.1

Use CLI flags to scale it.

## GitHub Pages

A static educational website lives in `docs/`, with an Actions workflow that deploys it to GitHub Pages. The intended URL is:

**https://krazad0.github.io/BUILDING-AN-LLM-FROM-SCRATCH-2026/**

The chat prototype will be available at:

**https://krazad0.github.io/BUILDING-AN-LLM-FROM-SCRATCH-2026/chat.html**

GitHub Pages is static hosting, so actual PyTorch training and checkpoint inference run locally, in Codespaces, Colab, or another compute environment.

## Suggested upgrades

After the baseline works, add a BPE tokenizer, RoPE, RMSNorm, SwiGLU, grouped-query attention, KV caching, mixed precision, distributed training, instruction fine-tuning, preference optimization, and ONNX/WebGPU browser inference.

## License

MIT License.

import torch

from src import ByteTokenizer, GPT, GPTConfig


def test_tokenizer_round_trip():
    tokenizer = ByteTokenizer()
    text = "Hello, LLM! Quantum: ψ"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_forward_and_loss_are_valid():
    config = GPTConfig(block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    model = GPT(config)
    x = torch.randint(0, 256, (2, 16))
    y = torch.randint(0, 256, (2, 16))
    logits, loss = model(x, y)
    assert logits.shape == (2, 16, 256)
    assert loss is not None
    assert torch.isfinite(loss)


def test_generate_extends_sequence():
    torch.manual_seed(0)
    config = GPTConfig(block_size=8, n_layer=1, n_head=2, n_embd=16, dropout=0.0)
    model = GPT(config)
    prompt = torch.tensor([[72, 105]], dtype=torch.long)
    out = model.generate(prompt, max_new_tokens=5, temperature=1.0, top_k=10)
    assert out.shape == (1, 7)


def test_context_limit_is_enforced():
    config = GPTConfig(block_size=4, n_layer=1, n_head=1, n_embd=8, dropout=0.0)
    model = GPT(config)
    x = torch.randint(0, 256, (1, 5))
    try:
        model(x)
    except ValueError as exc:
        assert "block_size" in str(exc)
    else:
        raise AssertionError("Expected a ValueError for an oversized context")

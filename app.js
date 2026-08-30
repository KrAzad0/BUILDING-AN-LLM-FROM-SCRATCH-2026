const stages = {
  tokens: {
    title: "01 — Byte tokenization",
    text: "Text becomes UTF-8 bytes. Every byte is an integer token from 0 to 255, giving a fixed vocabulary that can represent any Unicode text.",
    code: "ids = list(text.encode('utf-8'))"
  },
  embed: {
    title: "02 — Token + position embeddings",
    text: "Discrete IDs are converted into learned vectors. A learned position vector is added so the Transformer can distinguish where each token occurs.",
    code: "x = token_embedding(idx) + position_embedding(pos)"
  },
  attention: {
    title: "03 — Causal multi-head attention",
    text: "Queries compare with keys to produce attention weights. The upper triangle is masked so each token can only read its past and present, never the future.",
    code: "weights = softmax((Q @ K.T) / sqrt(d) + causal_mask)"
  },
  blocks: {
    title: "04 — Transformer blocks",
    text: "Pre-normalized attention and an MLP are wrapped in residual connections. Stacking blocks repeatedly mixes contextual information and transforms each token representation.",
    code: "x = x + attention(LN(x))\nx = x + MLP(LN(x))"
  },
  logits: {
    title: "05 — Next-token logits",
    text: "The final hidden vectors are projected to one score per vocabulary token. Cross entropy trains those scores; sampling turns them into generated text.",
    code: "logits = lm_head(final_norm(x))"
  }
};

const detail = document.querySelector("#stage-detail");
const stageButtons = document.querySelectorAll(".stage");
function showStage(name) {
  const item = stages[name];
  detail.innerHTML = `<h3>${item.title}</h3><p>${item.text}</p><code>${item.code}</code>`;
  stageButtons.forEach(btn => btn.classList.toggle("active", btn.dataset.stage === name));
}
stageButtons.forEach(btn => btn.addEventListener("click", () => showStage(btn.dataset.stage)));
showStage("tokens");

const input = document.querySelector("#token-input");
const output = document.querySelector("#token-output");
const tokenCount = document.querySelector("#token-count");
const encoder = new TextEncoder();
function tokenize() {
  const ids = Array.from(encoder.encode(input.value));
  tokenCount.textContent = ids.length.toLocaleString();
  output.textContent = `[${ids.join(", ")}]`;
}
input.addEventListener("input", tokenize);
tokenize();

const controls = {layers: document.querySelector("#layers"), heads: document.querySelector("#heads"), embd: document.querySelector("#embd")};
const values = {layers: document.querySelector("#layers-v"), heads: document.querySelector("#heads-v"), embd: document.querySelector("#embd-v")};
const paramCount = document.querySelector("#param-count");
function formatParams(n) {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return n.toString();
}
function calculateParams() {
  const L = Number(controls.layers.value), H = Number(controls.heads.value), C = Number(controls.embd.value);
  values.layers.textContent = L; values.heads.textContent = H; values.embd.textContent = C;
  const approximate = 256 * C + 128 * C + L * (12 * C * C + 13 * C) + 2 * C;
  const valid = C % H === 0;
  paramCount.textContent = `${formatParams(approximate)}${valid ? "" : " *"}`;
  paramCount.title = valid ? "Valid head split" : "Choose a head count that divides the embedding dimension";
}
Object.values(controls).forEach(control => control.addEventListener("input", calculateParams));
calculateParams();

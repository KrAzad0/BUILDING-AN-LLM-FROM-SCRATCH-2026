const $ = (selector) => document.querySelector(selector);
const encoder = new TextEncoder();

const conversation = $("#conversation");
const welcome = $("#welcome");
const promptBox = $("#prompt");
const sendButton = $("#send");
const newChatButton = $("#new-chat");
const temperature = $("#temperature");
const topk = $("#topk");
const temperatureValue = $("#temperature-value");
const topkValue = $("#topk-value");
const tokenCount = $("#token-count");
const sidebar = $("#sidebar");
const menuButton = $("#menu");

let isGenerating = false;
let messages = loadHistory();

const knowledge = {
  attention: {
    title: "Causal self-attention",
    body: [
      "Self-attention lets each token decide which earlier tokens matter for its next representation.",
      "The model creates queries Q, keys K, and values V. Similarity is computed with `QKᵀ / √d`, then a causal mask blocks all future positions before softmax is applied.",
      "That is why GPT can predict the next token without cheating: position t can only use positions ≤ t."
    ]
  },
  tokenizer: {
    title: "Byte tokenization",
    body: [
      "This project starts with a deliberately simple tokenizer: UTF-8 bytes.",
      "Every byte is already an integer from 0 to 255, so the vocabulary is fixed at 256 tokens and can represent any text without an unknown-token problem.",
      "The tradeoff is efficiency: modern LLMs normally use BPE or a related subword tokenizer so common words need fewer tokens."
    ]
  },
  train: {
    title: "Training the repository model",
    body: [
      "Install dependencies, train on a UTF-8 text corpus, then load the saved checkpoint for generation.",
      "```bash\npip install -r requirements.txt\npython train.py --data data/sample.txt --steps 1000\npython generate.py --checkpoint checkpoints/minigpt.pt --prompt \"Quantum mechanics\"\n```",
      "During training, the model minimizes next-token cross-entropy and saves the best validation checkpoint."
    ]
  },
  architecture: {
    title: "GPT-Scratch-256 architecture",
    body: [
      "The flow is: byte tokens → token + position embeddings → repeated Transformer blocks → final LayerNorm → vocabulary logits.",
      "Each Transformer block uses pre-norm residual structure: `x = x + attention(LN(x))`, followed by `x = x + MLP(LN(x))`.",
      "The default configuration uses a 256-token vocabulary, context length 128, 4 layers, 4 heads, and embedding dimension 256."
    ]
  },
  upgrade: {
    title: "A strong version-2 roadmap",
    body: [
      "A useful next step is to evolve the educational GPT into a modern decoder stack.",
      "I would prioritize: BPE tokenizer → RoPE → RMSNorm → SwiGLU → grouped-query attention → KV cache → mixed precision → larger training corpus → instruction tuning.",
      "After that, export a small checkpoint to ONNX or WebGPU so the chat page can run real model inference in the browser instead of the current demo engine."
    ]
  },
  mask: {
    title: "Why the causal mask matters",
    body: [
      "Without the mask, a token could attend to tokens that occur later in the training sequence.",
      "That would leak the answer during next-token prediction. The triangular causal mask sets future attention scores to −∞ before softmax, making their probability exactly zero.",
      "This one constraint turns ordinary self-attention into autoregressive GPT-style attention."
    ]
  }
};

const fallbackOpeners = [
  "A useful way to think about that in this project is",
  "For this from-scratch GPT, the key idea is",
  "In the current MiniGPT implementation, I would break that into three parts:",
  "The shortest technical answer is"
];

function loadHistory() {
  try {
    return JSON.parse(localStorage.getItem("minigpt-chat-history")) || [];
  } catch {
    return [];
  }
}

function saveHistory() {
  localStorage.setItem("minigpt-chat-history", JSON.stringify(messages.slice(-30)));
}

function escapeHtml(text) {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function formatMessage(text) {
  const escaped = escapeHtml(text);
  const withCodeBlocks = escaped.replace(/```([\s\S]*?)```/g, (_, code) => `<pre><code>${code.trim()}</code></pre>`);
  const withInlineCode = withCodeBlocks.replace(/`([^`]+)`/g, "<code>$1</code>");
  return withInlineCode
    .split(/\n\n+/)
    .map(part => part.startsWith("<pre>") ? part : `<p>${part.replaceAll("\n", "<br>")}</p>`)
    .join("");
}

function addMessage(role, text, { streaming = false } = {}) {
  welcome?.remove();
  const article = document.createElement("article");
  article.className = `message ${role}`;
  article.innerHTML = `
    <div class="avatar">${role === "user" ? "U" : "G"}</div>
    <div class="bubble">
      <div class="content ${streaming ? "cursor" : ""}">${formatMessage(text)}</div>
      <div class="message-meta">${role === "user" ? encoder.encode(text).length + " byte tokens" : "MiniGPT browser prototype"}</div>
    </div>`;
  conversation.appendChild(article);
  conversation.scrollTop = conversation.scrollHeight;
  return article.querySelector(".content");
}

function addThinking() {
  welcome?.remove();
  const article = document.createElement("article");
  article.className = "message assistant";
  article.innerHTML = `<div class="avatar">G</div><div class="bubble"><div class="thinking"><i></i><i></i><i></i></div></div>`;
  conversation.appendChild(article);
  conversation.scrollTop = conversation.scrollHeight;
  return article;
}

function choose(items) {
  const t = Number(temperature.value);
  const spread = Math.max(1, Math.min(items.length, Math.ceil(t * items.length)));
  return items[Math.floor(Math.random() * spread)];
}

function classify(input) {
  const text = input.toLowerCase();
  if (/attention|q\s*,?\s*k|query|key|value/.test(text)) return "attention";
  if (/token|byte|bpe|vocab/.test(text)) return "tokenizer";
  if (/train|training|checkpoint|generate|run|command/.test(text)) return "train";
  if (/architecture|layer|block|embedding|model/.test(text)) return "architecture";
  if (/upgrade|version 2|v2|llama|rope|rmsnorm|swiglu|gqa/.test(text)) return "upgrade";
  if (/mask|future|causal/.test(text)) return "mask";
  return null;
}

function makeResponse(input) {
  const lower = input.toLowerCase().trim();
  if (/^(hi|hello|hey|namaste)\b/.test(lower)) {
    return "Hello. I’m the browser prototype for the LLM-from-scratch project. Ask me about the tokenizer, Transformer architecture, causal attention, training commands, or how to improve the model.";
  }

  const topic = classify(input);
  if (topic) {
    const item = knowledge[topic];
    return `${item.title}\n\n${item.body.join("\n\n")}`;
  }

  const bytes = encoder.encode(input).length;
  const k = Number(topk.value);
  return `${choose(fallbackOpeners)} your prompt first becomes ${bytes} byte tokens, then those IDs are embedded and passed through causal Transformer blocks.\n\nFor a real learned answer to “${input.slice(0, 90)}${input.length > 90 ? "…" : ""}”, this repository needs a trained checkpoint containing enough examples related to that topic. The current GitHub Pages chat is intentionally a lightweight local prototype because static hosting cannot directly run the PyTorch checkpoint.\n\nWith temperature ${Number(temperature.value).toFixed(1)} and top-k ${k}, the production path would sample each next token from the model logits. A good next step is exporting the trained model to ONNX/WebGPU so this same interface can perform genuine browser inference.`;
}

async function streamResponse(text) {
  const target = addMessage("assistant", "", { streaming: true });
  let visible = "";
  const chunks = text.split(/(\s+)/);
  for (const chunk of chunks) {
    if (!isGenerating) break;
    visible += chunk;
    target.innerHTML = formatMessage(visible);
    conversation.scrollTop = conversation.scrollHeight;
    await new Promise(resolve => setTimeout(resolve, 12 + Math.random() * 24));
  }
  target.classList.remove("cursor");
}

async function sendPrompt(text = promptBox.value) {
  const clean = text.trim();
  if (!clean || isGenerating) return;

  isGenerating = true;
  sendButton.disabled = true;
  promptBox.value = "";
  resizePrompt();
  updateTokenCount();

  addMessage("user", clean);
  messages.push({ role: "user", content: clean });
  saveHistory();

  const thinking = addThinking();
  await new Promise(resolve => setTimeout(resolve, 280 + Math.random() * 320));
  thinking.remove();

  const response = makeResponse(clean);
  await streamResponse(response);
  messages.push({ role: "assistant", content: response });
  saveHistory();

  isGenerating = false;
  sendButton.disabled = false;
  promptBox.focus();
}

function resizePrompt() {
  promptBox.style.height = "auto";
  promptBox.style.height = `${Math.min(promptBox.scrollHeight, 190)}px`;
}

function updateTokenCount() {
  tokenCount.textContent = encoder.encode(promptBox.value).length.toLocaleString();
}

function resetChat() {
  messages = [];
  localStorage.removeItem("minigpt-chat-history");
  conversation.innerHTML = `
    <div class="welcome" id="welcome">
      <div class="logo-orb">G</div>
      <h2>Talk to the model prototype.</h2>
      <p>This interface runs entirely in your browser. It demonstrates the product experience around the trainable PyTorch model in this repository.</p>
      <div class="welcome-grid">
        <button data-prompt="Explain the equation Attention(Q,K,V) = softmax(QKᵀ/√d + mask)V."><b>Understand the math</b><span>Walk through attention step by step</span></button>
        <button data-prompt="Give me the exact commands to train and then generate text."><b>Run the model</b><span>Training and inference commands</span></button>
        <button data-prompt="Why is the causal mask necessary in a GPT model?"><b>Explore causality</b><span>Why future tokens stay hidden</span></button>
        <button data-prompt="What should version 2 of this repository implement?"><b>Plan v2</b><span>Roadmap toward a modern LLM</span></button>
      </div>
    </div>`;
  wirePromptButtons();
  promptBox.focus();
}

function wirePromptButtons() {
  document.querySelectorAll("[data-prompt]").forEach(button => {
    button.onclick = () => {
      sendPrompt(button.dataset.prompt);
      sidebar.classList.remove("open");
    };
  });
}

promptBox.addEventListener("input", () => {
  resizePrompt();
  updateTokenCount();
});
promptBox.addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendPrompt();
  }
});
sendButton.addEventListener("click", () => sendPrompt());
newChatButton.addEventListener("click", resetChat);
menuButton.addEventListener("click", () => sidebar.classList.toggle("open"));
temperature.addEventListener("input", () => temperatureValue.textContent = Number(temperature.value).toFixed(1));
topk.addEventListener("input", () => topkValue.textContent = topk.value);

wirePromptButtons();
updateTokenCount();
resizePrompt();

if (messages.length) {
  welcome?.remove();
  messages.forEach(message => addMessage(message.role, message.content));
}

from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

import torch

from src import ByteTokenizer, GPT, GPTConfig


ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class MiniGPTService:
    def __init__(self, checkpoint: Path, device: str):
        self.tokenizer = ByteTokenizer()
        self.device = choose_device(device)
        self.checkpoint_path = checkpoint
        self.model: GPT | None = None
        self.error: str | None = None
        self._load()

    def _load(self) -> None:
        if not self.checkpoint_path.exists():
            self.error = (
                f"Checkpoint not found: {self.checkpoint_path}. "
                "Train one first with python train.py --data data/sample.txt --steps 1000"
            )
            return

        try:
            payload = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            config = GPTConfig(**payload["config"])
            model = GPT(config).to(self.device)
            model.load_state_dict(payload["model_state"])
            model.eval()
            self.model = model
            self.error = None
            print(
                f"Loaded {self.checkpoint_path} | device={self.device} | "
                f"parameters={model.num_parameters():,}"
            )
        except Exception as exc:  # pragma: no cover - runtime diagnostics
            self.error = f"Could not load checkpoint: {exc}"

    def generate(
        self,
        messages: list[dict],
        temperature: float = 0.8,
        top_k: int = 40,
        max_new_tokens: int = 160,
    ) -> str:
        if self.model is None:
            raise RuntimeError(self.error or "Model is not loaded")

        turns = []
        for message in messages[-8:]:
            role = "User" if message.get("role") == "user" else "Assistant"
            content = str(message.get("content", "")).strip()
            if content:
                turns.append(f"{role}: {content}")
        transcript = "\n".join(turns) + "\nAssistant:"

        ids = self.tokenizer.encode(transcript)
        prompt = torch.tensor([ids], dtype=torch.long, device=self.device)
        generated = self.model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=max(0.05, float(temperature)),
            top_k=max(1, int(top_k)),
        )
        new_ids = generated[0, len(ids) :].tolist()
        text = self.tokenizer.decode(new_ids)

        # Stop if the small model begins a new synthetic turn.
        for marker in ("\nUser:", "\nAssistant:"):
            if marker in text:
                text = text.split(marker, 1)[0]
        return text.strip() or "[The checkpoint produced no printable text for this prompt.]"


class ChatHandler(SimpleHTTPRequestHandler):
    service: MiniGPTService

    def _json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/chat":
            self._json(404, {"error": "Not found"})
            return

        try:
            size = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(size) or b"{}")
            messages = body.get("messages", [])
            if not isinstance(messages, list) or not messages:
                self._json(400, {"error": "messages must be a non-empty array"})
                return

            reply = self.service.generate(
                messages=messages,
                temperature=body.get("temperature", 0.8),
                top_k=body.get("top_k", 40),
                max_new_tokens=min(512, max(1, int(body.get("max_new_tokens", 160)))),
            )
            self._json(
                200,
                {
                    "reply": reply,
                    "engine": "pytorch-checkpoint",
                    "device": self.service.device,
                },
            )
        except RuntimeError as exc:
            self._json(503, {"error": str(exc), "engine": "browser-fallback"})
        except Exception as exc:  # pragma: no cover - runtime diagnostics
            self._json(500, {"error": str(exc)})

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/status":
            self._json(
                200,
                {
                    "ready": self.service.model is not None,
                    "device": self.service.device,
                    "checkpoint": str(self.service.checkpoint_path),
                    "error": self.service.error,
                },
            )
            return

        path = unquote(self.path.split("?", 1)[0])
        if path in ("/", ""):
            path = "/chat.html"
        requested = (DOCS / path.lstrip("/")).resolve()

        try:
            requested.relative_to(DOCS.resolve())
        except ValueError:
            self.send_error(403)
            return

        if not requested.exists() or not requested.is_file():
            self.send_error(404)
            return

        data = requested.read_bytes()
        content_type = mimetypes.guess_type(str(requested))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args) -> None:
        print(f"[chat] {self.address_string()} - {fmt % args}")


def parse_args():
    parser = argparse.ArgumentParser(description="Serve the MiniGPT chat UI with real checkpoint inference")
    parser.add_argument("--checkpoint", default="checkpoints/minigpt.pt")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, mps, cuda:0, ...")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = MiniGPTService(Path(args.checkpoint), args.device)
    ChatHandler.service = service
    server = ThreadingHTTPServer((args.host, args.port), ChatHandler)
    print(f"MiniGPT chat: http://{args.host}:{args.port}/")
    if service.error:
        print(f"Model status: {service.error}")
        print("The UI still works with its browser fallback engine.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping MiniGPT chat server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("EXA_API_KEY", "test-key")


class _OpenAIHandler(BaseHTTPRequestHandler):
    requests = []
    model_ids = ["Qwen/Qwen3.6-35B-A3B"]

    def log_message(self, *_args):
        return

    def _write_json(self, payload):
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):
        self.__class__.requests.append(("GET", self.path, None))
        self._write_json(
            {
                "object": "list",
                "data": [
                    {
                        "id": model_id,
                        "object": "model",
                        "created": 0,
                        "owned_by": "local-test",
                    }
                    for model_id in self.__class__.model_ids
                ],
            }
        )

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        self.__class__.requests.append(("POST", self.path, body))
        self._write_json(
            {
                "id": "chatcmpl-local-test",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "local endpoint reasoning",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 4,
                    "total_tokens": 7,
                },
            }
        )


def test_synth_client_calls_local_openai_compatible_http_endpoint(monkeypatch):
    import config.config as config
    from data.synth_client import get_generate_fn, is_available

    _OpenAIHandler.requests = []
    _OpenAIHandler.model_ids = ["Qwen/Qwen3.6-35B-A3B"]
    server = ThreadingHTTPServer(("127.0.0.1", 0), _OpenAIHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}/v1"
    monkeypatch.setattr(config, "SYNTH_ENDPOINT", endpoint)
    monkeypatch.setattr(config, "SYNTH_MODEL", "Qwen/Qwen3.6-35B-A3B")
    monkeypatch.setattr(config, "SYNTH_API_KEY", "EMPTY")

    try:
        assert is_available(timeout=2.0, log=lambda _message: None) is True
        generate = get_generate_fn(
            log=lambda _message: None,
            request_timeout=2.0,
        )
        assert generate is not None
        result = generate("Explain 2+2", temperature=0.2, max_tokens=17)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)

    assert result == "local endpoint reasoning"
    assert _OpenAIHandler.requests[0][:2] == ("GET", "/v1/models")
    _, path, request = _OpenAIHandler.requests[1]
    assert path == "/v1/chat/completions"
    assert request["model"] == "Qwen/Qwen3.6-35B-A3B"
    assert request["messages"] == [
        {"role": "user", "content": "Explain 2+2"}
    ]
    assert request["max_tokens"] == 17
    assert request["temperature"] == 0.2
    assert request["top_k"] == 20
    assert request["chat_template_kwargs"] == {"enable_thinking": False}


def test_synth_preflight_rejects_endpoint_serving_different_model(monkeypatch):
    import config.config as config
    from data.synth_client import is_available

    _OpenAIHandler.requests = []
    _OpenAIHandler.model_ids = ["other/local-model"]
    server = ThreadingHTTPServer(("127.0.0.1", 0), _OpenAIHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}/v1"
    monkeypatch.setattr(config, "SYNTH_ENDPOINT", endpoint)
    monkeypatch.setattr(config, "SYNTH_MODEL", "Qwen/Qwen3.6-35B-A3B")
    monkeypatch.setattr(config, "SYNTH_API_KEY", "EMPTY")
    logs = []

    try:
        available = is_available(timeout=2.0, log=logs.append)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)

    assert available is False
    assert _OpenAIHandler.requests == [("GET", "/v1/models", None)]
    rendered = "\n".join(logs)
    assert "Qwen/Qwen3.6-35B-A3B" in rendered
    assert "other/local-model" in rendered

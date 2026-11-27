"""Call Interfaze's OCR task and cache the raw responses.

Works without an API key (Interfaze allows 50 keyless requests per IP per day).
Set INTERFAZE_API_KEY to use a key instead. Raw responses are cached so the
analysis never needs to call the API twice for the same image.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

API_URL = "https://api.interfaze.ai/v1/chat/completions"


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi  # type: ignore
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        pass
    for path in ("/etc/ssl/cert.pem", "/etc/ssl/certs/ca-certificates.crt"):
        if os.path.exists(path):
            return ssl.create_default_context(cafile=path)
    return ssl.create_default_context()


def ocr_request(image: Path, model: str) -> dict:
    mime = "image/png" if image.suffix.lower() == ".png" else "image/jpeg"
    data_url = f"data:{mime};base64," + base64.b64encode(image.read_bytes()).decode()
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": "<task>ocr</task>"},
            {"role": "user", "content": [
                {"type": "text", "text": "Extract all text from this document"},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]},
        ],
    }


class QuotaExhausted(Exception):
    pass


def call(image: Path, model: str, timeout: int = 300) -> tuple[dict, dict]:
    headers = {"Content-Type": "application/json", "User-Agent": "ocr-trust/0.1"}
    key = os.environ.get("INTERFAZE_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(API_URL, data=json.dumps(ocr_request(image, model)).encode(), headers=headers)
    try:
        with urllib.request.urlopen(req, context=_ssl_context(), timeout=timeout) as r:
            limits = {k.lower(): v for k, v in r.headers.items() if "ratelimit" in k.lower()}
            return json.loads(r.read()), limits
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:500]
        if e.code == 429:
            raise QuotaExhausted(body) from e
        raise RuntimeError(f"HTTP {e.code}: {body}") from e


def collect(images: list[Path], out_dir: Path, model: str, run: int, workers: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [p for p in images if not (out_dir / f"{p.stem}__r{run}.json").exists()]
    print(f"{len(images) - len(todo)} cached, {len(todo)} to request (run {run}, {model})")
    stop = False

    def one(p: Path):
        nonlocal stop
        if stop:
            return
        t = time.time()
        try:
            resp, limits = call(p, model)
        except QuotaExhausted as e:
            stop = True
            print(f"  quota reached at {p.stem}: {e}", file=sys.stderr)
            return
        except Exception as e:  # one failure should not lose the rest; it is retried next time
            print(f"  {p.stem}: failed ({e})", file=sys.stderr)
            return
        resp["_meta"] = {"seconds": round(time.time() - t, 2), "run": run, "image": p.name}
        (out_dir / f"{p.stem}__r{run}.json").write_text(json.dumps(resp))
        print(f"  {p.stem}: {resp['_meta']['seconds']}s, remaining={limits.get('x-ratelimit-remaining')}")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, todo))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("images", type=Path, help="folder of document images")
    ap.add_argument("--out", type=Path, required=True, help="where raw responses are cached")
    ap.add_argument("--model", default="interfaze-1")
    ap.add_argument("--run", type=int, default=1, help="repeat index, for consistency runs")
    ap.add_argument("--only", nargs="*", help="image stems to include")
    ap.add_argument("--limit", type=int, help="at most this many images")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args(argv)
    images = sorted(a.images.glob("*.png")) + sorted(a.images.glob("*.jpg"))
    if a.only:
        images = [p for p in images if p.stem in set(a.only)]
    if a.limit:
        images = images[: a.limit]
    collect(images, a.out, a.model, a.run, a.workers)


if __name__ == "__main__":
    main()

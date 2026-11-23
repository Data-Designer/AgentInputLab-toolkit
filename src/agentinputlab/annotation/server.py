"""Loopback-only annotation web server with automatic versioned persistence."""

from __future__ import annotations

import argparse
import json
import mimetypes
import socket
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .store import ANNOTATION_KINDS, AnnotationStore, AnnotationStoreError

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
MAX_REQUEST_BYTES = 32_768
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_ROUTES = {"/": "index.html", "/app.js": "app.js", "/styles.css": "styles.css"}


class AnnotationHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], *, store: AnnotationStore) -> None:
        self.store = store
        self.address_family = socket.AF_INET6 if ":" in address[0] else socket.AF_INET
        super().__init__(address, AnnotationHandler)


class AnnotationHandler(BaseHTTPRequestHandler):
    server: AnnotationHTTPServer
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        if not self._accept_host():
            return
        path = urlsplit(self.path).path
        try:
            if path == "/api/status":
                self._json(self.server.store.status())
                return
            if path.startswith("/api/versions/"):
                version_id = path.removeprefix("/api/versions/")
                self._json(self.server.store.payload(version_id))
                return
            if path in STATIC_ROUTES:
                self._static(STATIC_ROUTES[path])
                return
            self._error(HTTPStatus.NOT_FOUND, "not_found", "route does not exist")
        except AnnotationStoreError as exc:
            self._error(HTTPStatus.UNPROCESSABLE_ENTITY, "annotation_invalid", str(exc))

    def do_POST(self) -> None:
        if not self._accept_host():
            return
        path = urlsplit(self.path).path
        try:
            payload = self._read_json()
            if path == "/api/versions":
                if payload.get("action") != "new":
                    raise AnnotationStoreError(
                        "choose an existing completed version in the UI or request action=new"
                    )
                self._json(
                    self.server.store.create_version(
                        annotator_id=str(payload.get("annotator_id", ""))
                    ),
                    status=HTTPStatus.CREATED,
                )
                return
            if path == "/api/selection":
                self._json(self.server.store.select(str(payload.get("version_id", ""))))
                return
            if path.startswith("/api/versions/") and path.endswith("/annotations"):
                version_id = path.removeprefix("/api/versions/").removesuffix("/annotations")
                self._json(self.server.store.save(version_id, payload))
                return
            if path.startswith("/api/versions/") and path.endswith("/complete"):
                version_id = path.removeprefix("/api/versions/").removesuffix("/complete")
                self._json(self.server.store.complete(version_id))
                return
            self._error(HTTPStatus.NOT_FOUND, "not_found", "route does not exist")
        except _HTTPInputError as exc:
            self._error(exc.status, exc.code, exc.message)
        except AnnotationStoreError as exc:
            self._error(HTTPStatus.UNPROCESSABLE_ENTITY, "annotation_invalid", str(exc))

    def _read_json(self) -> dict[str, Any]:
        if self.headers.get("Content-Type", "").partition(";")[0].strip() != "application/json":
            raise _HTTPInputError(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                "content_type",
                "Content-Type must be application/json",
            )
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise _HTTPInputError(
                HTTPStatus.BAD_REQUEST, "content_length", "invalid length"
            ) from exc
        if length < 1 or length > MAX_REQUEST_BYTES:
            raise _HTTPInputError(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                "body_size",
                "JSON request size is invalid",
            )
        try:
            value = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise _HTTPInputError(HTTPStatus.BAD_REQUEST, "invalid_json", "invalid JSON") from exc
        if not isinstance(value, dict):
            raise _HTTPInputError(HTTPStatus.BAD_REQUEST, "json_type", "JSON must be an object")
        return value

    def _accept_host(self) -> bool:
        raw = self.headers.get("Host", "")
        try:
            parsed = urlsplit(f"//{raw}")
            _ = parsed.port
        except ValueError:
            parsed = None
        if parsed is None or parsed.hostname not in LOOPBACK_HOSTS:
            self._error(HTTPStatus.MISDIRECTED_REQUEST, "host_rejected", "loopback Host required")
            return False
        return True

    def _security_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; "
            "img-src 'self' data:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
        )

    def _json(self, value: Any, *, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self._security_headers()
        self.end_headers()
        self.wfile.write(data)

    def _error(self, status: HTTPStatus, code: str, message: str) -> None:
        self._json({"error": {"code": code, "message": message}}, status=status)

    def _static(self, filename: str) -> None:
        data = (STATIC_DIR / filename).read_bytes()
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type == "application/javascript":
            content_type += "; charset=utf-8"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self._security_headers()
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write(
            f"annotation {self.client_address[0]} {self.command} "
            f"{urlsplit(self.path).path} — {format % args}\n"
        )


class _HTTPInputError(Exception):
    def __init__(self, status: HTTPStatus, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def create_server(
    host: str,
    port: int,
    *,
    items_path: str | Path,
    output_root: str | Path,
    kind: str,
) -> AnnotationHTTPServer:
    if host not in LOOPBACK_HOSTS:
        raise ValueError("annotation UI is loopback-only; use SSH port forwarding on L20")
    return AnnotationHTTPServer(
        (host, port),
        store=AnnotationStore(items_path=items_path, root=output_root, kind=kind),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--kind", choices=sorted(ANNOTATION_KINDS), required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7862)
    args = parser.parse_args(argv)
    if args.host not in LOOPBACK_HOSTS:
        print("annotation UI is loopback-only", file=sys.stderr)
        return 2
    server = create_server(
        args.host,
        args.port,
        items_path=args.items,
        output_root=args.output_root,
        kind=args.kind,
    )
    host, port = server.server_address[:2]
    print(f"AgentInputLab annotation UI listening on http://{host}:{port}")
    if server.store.status()["completed_version_exists"]:
        print("Completed annotation found: choose reuse or create a named new version in the UI.")
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\nStopping annotation UI")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["AnnotationHTTPServer", "create_server", "main"]

"""Dependency-free HTTP server for the packaged local AgentInputLab Studio."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import secrets
import socket
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from agentinputlab.contracts.errors import ArtifactError, ValidationError

from .campaigns import (
    CampaignManager,
    CampaignQueueFullError,
    CampaignSecurityError,
    CampaignValidationError,
    disabled_campaign_catalog_payload,
)
from .engine import (
    RunManager,
    StudioQueueFullError,
    StudioSecurityError,
    StudioValidationError,
    validate_config,
)
from .model_backends import (
    STUDIO_MODEL_BACKEND_CONFIG_ENV,
    STUDIO_MODEL_BACKEND_CONFIG_SHA256_ENV,
    StudioModelBackendGate,
)
from .workflows import WorkflowBusyError, WorkflowManager, generate_workflow, workflow_catalog

MAX_REQUEST_BYTES = 65_536
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
RUN_ID = re.compile(r"^[0-9a-f]{32}$")
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_ROUTES = {
    "/": "index.html",
    "/index.html": "index.html",
    "/result": "result.html",
    "/result.html": "result.html",
    "/campaigns": "campaigns.html",
    "/campaigns.html": "campaigns.html",
    "/styles.css": "styles.css",
    "/app.js": "app.js",
    "/result.js": "result.js",
    "/campaigns.js": "campaigns.js",
    "/workflows": "workflows.html",
    "/workflows.html": "workflows.html",
    "/workflows.js": "workflows.js",
    "/workflows.css": "workflows.css",
}


class StudioHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        handler: type[BaseHTTPRequestHandler],
        *,
        manager: RunManager,
        campaign_manager: CampaignManager | None = None,
        workflow_manager: WorkflowManager | None = None,
    ) -> None:
        self.manager = manager
        self.catalog = manager.catalog
        self.campaign_manager = campaign_manager
        self.workflow_manager = workflow_manager
        self.workflow_session = secrets.token_urlsafe(32)
        self.address_family = socket.AF_INET6 if ":" in server_address[0] else socket.AF_INET
        super().__init__(server_address, handler)

    def server_close(self) -> None:
        try:
            super().server_close()
        finally:
            self.manager.shutdown(wait=True)
            if self.campaign_manager is not None:
                self.campaign_manager.shutdown(wait=True)
            if self.workflow_manager is not None:
                self.workflow_manager.shutdown()


class StudioHandler(BaseHTTPRequestHandler):
    server: StudioHTTPServer
    protocol_version = "HTTP/1.1"

    def _accept_host(self) -> bool:
        raw_host = self.headers.get("Host", "")
        try:
            parsed = urlsplit(f"//{raw_host}")
            _ = parsed.port
        except ValueError:
            parsed = None
        if (
            parsed is None
            or parsed.hostname not in LOOPBACK_HOSTS
            or parsed.username is not None
            or parsed.password is not None
        ):
            self._send_error(
                HTTPStatus.MISDIRECTED_REQUEST,
                "host_rejected",
                "Host must be localhost, 127.0.0.1, or [::1]",
            )
            return False
        return True

    def _security_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; connect-src 'self'; object-src 'none'; "
            "base-uri 'none'; frame-ancestors 'none'; form-action 'self'",
        )
        self.send_header(
            "Permissions-Policy",
            "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
        )

    def _send_json(
        self,
        payload: Any,
        *,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        data = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self._security_headers()
        self.end_headers()
        self.wfile.write(data)

    def _send_error(self, status: HTTPStatus, code: str, message: str) -> None:
        self._send_json(
            {"error": {"code": code, "message": message}},
            status=status,
        )

    def _read_json(self, *, strict=False) -> Any:
        content_type = self.headers.get("Content-Type", "").partition(";")[0].strip().lower()
        if content_type != "application/json":
            raise _HTTPInputError(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                "content_type",
                "Content-Type must be application/json",
            )
        raw_length = self.headers.get("Content-Length")
        try:
            length = int(raw_length or "0")
        except ValueError as exc:
            raise _HTTPInputError(
                HTTPStatus.BAD_REQUEST,
                "content_length",
                "invalid Content-Length",
            ) from exc
        if length < 1:
            raise _HTTPInputError(HTTPStatus.BAD_REQUEST, "empty_body", "JSON body is required")
        if length > MAX_REQUEST_BYTES:
            raise _HTTPInputError(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                "body_too_large",
                f"request exceeds {MAX_REQUEST_BYTES} bytes",
            )
        data = self.rfile.read(length)
        try:
            if strict:
                from agentinputlab.configuration import _unique
                def reject_constant(_value):
                    raise ValueError("non-finite JSON number")
                return json.loads(data.decode("utf-8"), object_pairs_hook=_unique,
                                  parse_constant=reject_constant)
            return json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, ValueError, ValidationError) as exc:
            raise _HTTPInputError(
                HTTPStatus.BAD_REQUEST,
                "invalid_json",
                "body must be valid UTF-8 JSON",
            ) from exc

    def do_GET(self) -> None:
        if not self._accept_host():
            return
        parsed = urlsplit(self.path)
        path = parsed.path
        if path.startswith("/api/workflows/"):
            self._handle_workflow(path, post=False)
            return
        if path == "/api/health":
            self._send_json(
                {
                    "ok": True,
                    "service": "agentinputlab-studio",
                    "version": 1,
                    "execution": "local-only",
                    "network_runs": bool(
                        self.server.catalog["execution_policy"]["network_enabled"]
                    ),
                    "campaign_mode_configured": self.server.campaign_manager is not None,
                }
            )
            return
        if path == "/api/catalog":
            self._send_json(self.server.catalog)
            return
        if path == "/api/campaigns/profiles":
            campaigns = self.server.campaign_manager
            self._send_json(
                campaigns.catalog_payload()
                if campaigns is not None
                else disabled_campaign_catalog_payload()
            )
            return
        if path == "/api/campaigns/executions":
            campaigns = self.server.campaign_manager
            self._send_json(
                {
                    "configured": campaigns is not None,
                    "executions": (campaigns.list_executions() if campaigns is not None else []),
                }
            )
            return
        if path.startswith("/api/campaigns/executions/"):
            execution_id = path.removeprefix("/api/campaigns/executions/")
            campaigns = self.server.campaign_manager
            record = (
                campaigns.get_execution(execution_id)
                if campaigns is not None and RUN_ID.fullmatch(execution_id)
                else None
            )
            if record is None:
                self._send_error(
                    HTTPStatus.NOT_FOUND,
                    "campaign_execution_not_found",
                    "campaign execution does not exist",
                )
                return
            self._send_json(record)
            return
        if path.startswith("/api/campaigns/preflights/"):
            preparation_id = path.removeprefix("/api/campaigns/preflights/")
            campaigns = self.server.campaign_manager
            record = (
                campaigns.get_preparation(preparation_id)
                if campaigns is not None and RUN_ID.fullmatch(preparation_id)
                else None
            )
            if record is None:
                self._send_error(
                    HTTPStatus.NOT_FOUND,
                    "campaign_preflight_not_found",
                    "campaign preflight does not exist",
                )
                return
            self._send_json(record)
            return
        if path == "/api/runs":
            self._send_json({"runs": self.server.manager.list_runs()})
            return
        if path.startswith("/api/runs/"):
            run_id = path.removeprefix("/api/runs/")
            if not RUN_ID.fullmatch(run_id):
                self._send_error(HTTPStatus.NOT_FOUND, "run_not_found", "run does not exist")
                return
            record = self.server.manager.get(run_id)
            if record is None:
                self._send_error(HTTPStatus.NOT_FOUND, "run_not_found", "run does not exist")
                return
            self._send_json(record)
            return
        if path in STATIC_ROUTES:
            self._send_static(STATIC_ROUTES[path])
            return
        self._send_error(HTTPStatus.NOT_FOUND, "not_found", "route does not exist")

    def do_POST(self) -> None:
        if not self._accept_host():
            return
        path = urlsplit(self.path).path
        if path.startswith("/api/workflows/"):
            self._handle_workflow(path, post=True)
            return
        if path not in {
            "/api/validate",
            "/api/runs",
            "/api/campaigns/preflight",
            "/api/campaigns/executions",
        }:
            self._send_error(HTTPStatus.NOT_FOUND, "not_found", "route does not exist")
            return
        if path.startswith("/api/campaigns/"):
            self._handle_campaign_post(path)
            return
        try:
            payload = self._read_json()
            report = validate_config(payload, catalog=self.server.catalog)
        except _HTTPInputError as exc:
            self._send_error(exc.status, exc.code, exc.message)
            return
        except StudioSecurityError as exc:
            self._send_error(HTTPStatus.FORBIDDEN, "credentials_rejected", str(exc))
            return
        if path == "/api/validate":
            self._send_json(report)
            return
        if not report["valid"]:
            self._send_json(report, status=HTTPStatus.UNPROCESSABLE_ENTITY)
            return
        try:
            run = self.server.manager.submit(payload)
        except StudioQueueFullError as exc:
            self._send_error(HTTPStatus.TOO_MANY_REQUESTS, "queue_full", str(exc))
            return
        except StudioValidationError as exc:
            self._send_error(HTTPStatus.UNPROCESSABLE_ENTITY, "invalid_config", str(exc))
            return
        self._send_json(run, status=HTTPStatus.ACCEPTED)

    def _handle_workflow(self, path: str, *, post: bool) -> None:
        origin = self.headers.get("Origin")
        host = self.headers.get("Host", "")
        if (self.headers.get("Sec-Fetch-Site") == "cross-site"
                or (origin is not None and origin != f"http://{host}")):
            self._send_error(HTTPStatus.FORBIDDEN, "origin_rejected", "Use the Studio page on this server.")
            return
        if post and not secrets.compare_digest(
                self.headers.get("X-Studio-Session", ""), self.server.workflow_session):
            self._send_error(HTTPStatus.FORBIDDEN, "session_required", "Reload Studio before submitting a workflow.")
            return
        manager = self.server.workflow_manager
        try:
            if not post and path == "/api/workflows/catalog":
                self._send_json({**workflow_catalog(), "session": self.server.workflow_session,
                                 "execution_enabled": manager is not None,
                                 "workspace": str(manager.root) if manager else None})
                return
            payload = self._read_json(strict=True) if post else None
            if post:
                from agentinputlab.experiments.secrets import require_secret_free
                require_secret_free(payload, label="workflow request")
                if not isinstance(payload, dict):
                    raise ValidationError("workflow request must be an object")
            if post and path == "/api/workflows/template":
                self._send_json(generate_workflow(payload))
                return
            if manager is None:
                self._send_error(HTTPStatus.CONFLICT, "workflow_workspace_required",
                                 "Start Studio with --workflow-root ./studio-workflows to inspect and execute local workflows.")
                return
            if not post and path == "/api/workflows/runs":
                self._send_json({"runs": manager.list_runs()})
                return
            if not post and path.startswith("/api/workflows/runs/"):
                self._send_json(manager.get(path.removeprefix("/api/workflows/runs/")))
                return
            if post and path == "/api/workflows/inspect":
                if set(payload) - {"workflow", "probe_endpoint"}:
                    raise ValidationError("unknown inspection fields")
                self._send_json(manager.inspect(payload.get("workflow"),
                                                probe_endpoint=payload.get("probe_endpoint", False)))
                return
            if post and path == "/api/workflows/launch":
                if set(payload) - {"workflow", "allow_network"} or "workflow" not in payload:
                    raise ValidationError("launch requires workflow and optional allow_network")
                self._send_json(manager.submit(payload["workflow"], allow_network=payload.get("allow_network", False)),
                                status=HTTPStatus.ACCEPTED)
                return
            if post and path == "/api/workflows/resume":
                if set(payload) - {"id", "allow_network"} or "id" not in payload:
                    raise ValidationError("resume requires an existing run id")
                self._send_json(manager.submit(resume_id=payload["id"], allow_network=payload.get("allow_network", False)),
                                status=HTTPStatus.ACCEPTED)
                return
            self._send_error(HTTPStatus.NOT_FOUND, "not_found", "workflow route does not exist")
        except _HTTPInputError as exc:
            self._send_error(exc.status, exc.code, exc.message)
        except WorkflowBusyError as exc:
            self._send_error(HTTPStatus.CONFLICT, "workflow_busy", str(exc))
        except (ArtifactError, ValidationError, TypeError, ValueError, KeyError) as exc:
            from agentinputlab.experiments.secrets import redact_secret_text
            self._send_error(HTTPStatus.UNPROCESSABLE_ENTITY, "workflow_invalid", redact_secret_text(str(exc)))
        except Exception:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "workflow_error",
                             "Workflow operation failed. Check local dependencies and workspace access.")

    def _handle_campaign_post(self, path: str) -> None:
        campaigns = self.server.campaign_manager
        if campaigns is None:
            self._send_error(
                HTTPStatus.CONFLICT,
                "campaigns_not_configured",
                "Studio was not started with a campaign profile catalog",
            )
            return
        try:
            payload = self._read_json()
            if path == "/api/campaigns/preflight":
                self._send_json(campaigns.prepare(payload))
                return
            self._send_json(
                campaigns.submit_execution(payload),
                status=HTTPStatus.ACCEPTED,
            )
        except _HTTPInputError as exc:
            self._send_error(exc.status, exc.code, exc.message)
        except CampaignSecurityError as exc:
            self._send_error(
                HTTPStatus.FORBIDDEN,
                "campaign_input_rejected",
                str(exc),
            )
        except CampaignQueueFullError as exc:
            self._send_error(HTTPStatus.TOO_MANY_REQUESTS, "queue_full", str(exc))
        except CampaignValidationError as exc:
            self._send_error(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "campaign_invalid",
                str(exc),
            )
        except (ArtifactError, ValidationError):
            self._send_error(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "campaign_preflight_failed",
                "Campaign profile attestation or preflight failed",
            )

    def do_OPTIONS(self) -> None:
        if not self._accept_host():
            return
        self.send_response(HTTPStatus.METHOD_NOT_ALLOWED)
        self.send_header("Allow", "GET, POST")
        self.send_header("Content-Length", "0")
        self._security_headers()
        self.end_headers()

    def _send_static(self, filename: str) -> None:
        path = STATIC_DIR / filename
        try:
            data = path.read_bytes()
        except OSError:
            self._send_error(HTTPStatus.NOT_FOUND, "asset_not_found", "asset does not exist")
            return
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type == "application/javascript":
            content_type = f"{content_type}; charset=utf-8"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self._security_headers()
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:
        # The handler never logs request bodies or headers, which keeps secrets
        # out of terminal output even when a client sends a rejected payload.
        sys.stderr.write(
            f"studio {self.client_address[0]} {self.command} "
            f"{urlsplit(self.path).path} — {format % args}\n"
        )


class _HTTPInputError(Exception):
    def __init__(self, status: HTTPStatus, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def create_server(
    host: str = "127.0.0.1",
    port: int = 7861,
    *,
    max_workers: int = 1,
    manager: RunManager | None = None,
    prepared_data_root: str | Path | None = None,
    campaign_manager: CampaignManager | None = None,
    campaign_catalog_path: str | Path | None = None,
    campaign_output_root: str | Path | None = None,
    campaign_workers: int = 1,
    model_backend_gate: StudioModelBackendGate | None = None,
    model_backend_config_path: str | Path | None = None,
    model_backend_config_sha256: str | None = None,
    workflow_root: str | Path | None = None,
    allow_remote: bool = False,
) -> StudioHTTPServer:
    if host not in LOOPBACK_HOSTS and not allow_remote:
        raise ValueError(
            "refusing non-loopback bind without allow_remote=True; Studio has no authentication"
        )
    if workflow_root is not None and host not in LOOPBACK_HOSTS:
        raise ValueError("editable workflows require a loopback server; use SSH port forwarding for remote work")
    if manager is not None and prepared_data_root is not None:
        raise ValueError("prepared_data_root must be configured on the supplied RunManager")
    if manager is not None and (
        model_backend_gate is not None
        or model_backend_config_path is not None
        or model_backend_config_sha256 is not None
    ):
        raise ValueError("model backend gate must be configured on the supplied RunManager")
    if model_backend_gate is not None and (
        model_backend_config_path is not None or model_backend_config_sha256 is not None
    ):
        raise ValueError("supply either model_backend_gate or operator config path/SHA, not both")
    if campaign_manager is not None and (
        campaign_catalog_path is not None or campaign_output_root is not None
    ):
        raise ValueError(
            "campaign catalog/output must be configured on the supplied CampaignManager"
        )
    if campaign_catalog_path is None and campaign_output_root is not None:
        raise ValueError("campaign_output_root requires campaign_catalog_path")
    operator_backend_gate = model_backend_gate or StudioModelBackendGate.from_operator_config(
        model_backend_config_path,
        model_backend_config_sha256,
    )
    run_manager = manager or RunManager(
        max_workers=max_workers,
        prepared_data_root=prepared_data_root,
        model_backend_gate=operator_backend_gate,
    )
    campaigns = campaign_manager
    owns_campaigns = campaigns is None and campaign_catalog_path is not None
    workflows = None
    try:
        if workflow_root is not None:
            workflows = WorkflowManager(workflow_root)
        if owns_campaigns:
            campaigns = CampaignManager(
                catalog_path=campaign_catalog_path,
                output_root=campaign_output_root,
                max_workers=campaign_workers,
            )
        return StudioHTTPServer(
            (host, port),
            StudioHandler,
            manager=run_manager,
            campaign_manager=campaigns,
            workflow_manager=workflows,
        )
    except Exception:
        if manager is None:
            run_manager.shutdown(wait=True)
        if owns_campaigns and campaigns is not None:
            campaigns.shutdown(wait=True)
        if workflows is not None:
            workflows.shutdown()
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local AgentInputLab experiment Studio.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (loopback by default).")
    parser.add_argument("--port", type=int, default=7861, help="Bind port.")
    parser.add_argument("--workers", type=int, default=1, choices=range(1, 5))
    parser.add_argument("--workflow-root", type=Path,
                        help="Enable trusted local workflow editing/execution; store runs in this directory. Model keys stay in environment variables.")
    parser.add_argument(
        "--prepared-data-root",
        type=Path,
        default=(
            Path(os.environ["AGENTINPUTLAB_PREPARED_DATA_ROOT"])
            if os.environ.get("AGENTINPUTLAB_PREPARED_DATA_ROOT")
            else None
        ),
        help=(
            "Server-controlled prepared benchmark cache root. "
            "May also be set with AGENTINPUTLAB_PREPARED_DATA_ROOT."
        ),
    )
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="Acknowledge that remote binding has no authentication and is unsafe on public networks.",
    )
    parser.add_argument(
        "--campaign-catalog",
        type=Path,
        default=(
            Path(os.environ["AGENTINPUTLAB_CAMPAIGN_CATALOG"])
            if os.environ.get("AGENTINPUTLAB_CAMPAIGN_CATALOG")
            else None
        ),
        help=(
            "Server-controlled, checksum-locked campaign profile catalog. "
            "Never accepted from a browser request."
        ),
    )
    parser.add_argument(
        "--campaign-output-root",
        type=Path,
        default=(
            Path(os.environ["AGENTINPUTLAB_CAMPAIGN_OUTPUT_ROOT"])
            if os.environ.get("AGENTINPUTLAB_CAMPAIGN_OUTPUT_ROOT")
            else None
        ),
        help=(
            "Server-controlled campaign artifact root. Without it, campaign "
            "profiles remain preflight/plan only."
        ),
    )
    parser.add_argument(
        "--campaign-workers",
        type=int,
        default=1,
        choices=range(1, 5),
        help="Bounded campaign workers, separate from tiny Studio runs.",
    )
    parser.add_argument(
        "--model-backend-config",
        type=Path,
        default=(
            Path(os.environ[STUDIO_MODEL_BACKEND_CONFIG_ENV])
            if os.environ.get(STUDIO_MODEL_BACKEND_CONFIG_ENV)
            else None
        ),
        help=(
            "Server-controlled ProductionBackendBuilder parameter file. "
            f"May also be set with {STUDIO_MODEL_BACKEND_CONFIG_ENV}; never "
            "accepted from a browser request."
        ),
    )
    parser.add_argument(
        "--model-backend-config-sha256",
        default=os.environ.get(STUDIO_MODEL_BACKEND_CONFIG_SHA256_ENV),
        help=(
            "Exact SHA-256 of --model-backend-config. "
            f"May also be set with {STUDIO_MODEL_BACKEND_CONFIG_SHA256_ENV}."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.host not in LOOPBACK_HOSTS and not args.allow_remote:
        print(
            "refusing non-loopback bind without --allow-remote; Studio has no authentication",
            file=sys.stderr,
        )
        return 2
    server = create_server(
        args.host,
        args.port,
        max_workers=args.workers,
        prepared_data_root=args.prepared_data_root,
        campaign_catalog_path=args.campaign_catalog,
        campaign_output_root=args.campaign_output_root,
        campaign_workers=args.campaign_workers,
        model_backend_config_path=args.model_backend_config,
        model_backend_config_sha256=args.model_backend_config_sha256,
        workflow_root=args.workflow_root,
        allow_remote=args.allow_remote,
    )
    host, port = server.server_address[:2]
    print(f"AgentInputLab Studio listening on http://{host}:{port}")
    print(f"Workflow builder: http://{host}:{port}/workflows")
    if args.workflow_root is not None:
        print(f"Editable local workflows enabled: {server.workflow_manager.root}")
    model_backend_id = server.catalog["execution_policy"]["operator_model_backend_id"]
    if model_backend_id is None:
        print("Local execution only · credentials rejected · operator model backend locked")
    else:
        print(
            "Local execution only · credentials rejected · "
            f"operator model backend ready: {model_backend_id}"
        )
    verified_cells = server.catalog["execution_policy"]["verified_prepared_cells"]
    if verified_cells:
        print(f"Verified prepared benchmark cells: {', '.join(verified_cells)}")
    if server.campaign_manager is None:
        print("Research campaigns: disabled (no server-side profile catalog)")
    else:
        profiles = server.campaign_manager.catalog_payload()["profiles"]
        mode = (
            "preflight + confirmed execution"
            if args.campaign_output_root is not None
            else "preflight/plan only"
        )
        print(f"Research campaigns: {len(profiles)} allowlisted profiles · {mode}")
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\nStopping Studio")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["MAX_REQUEST_BYTES", "StudioHTTPServer", "create_server", "main"]

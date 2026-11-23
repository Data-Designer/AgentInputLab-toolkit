"""Independent, blinded two-person input review with third-person adjudication."""

# ruff: noqa: RUF001 -- Chinese UI text intentionally uses native full-width punctuation.
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from agentinputlab.contracts.serialization import content_hash
from agentinputlab.experiments.measurement import MeasurementItem, analyze_measurement_annotations

FIELDS = {
    "contract_anchor": {
        "entity_anchor_present": (
            "实体锚点",
            "输入中是否明确给出必须保留的人名、地点、产品、对象或标识符？",
        ),
        "numeric_anchor_present": (
            "数值锚点",
            "是否给出必须保留的金额、数量、时间、日期、范围或阈值？",
        ),
        "constraint_anchor_present": (
            "约束锚点",
            "是否明确规定必须／禁止、范围、格式、先后次序或资源限制？",
        ),
    },
    "defect_eligibility": {
        "unclear_or_misaligned_request": (
            "歧义／意图不清",
            "在可见上下文中，请求是否仍有阻碍正确执行的歧义或不一致？",
        ),
        "decomposition_need": ("需任务分解", "是否有多个依赖步骤，显式规划或分解能减少执行遗漏？"),
        "missing_external_evidence": (
            "缺外部证据",
            "执行是否缺少输入和已有上下文之外的事实或检索材料？",
        ),
        "clarification_need": ("需澄清", "是否缺少只能由用户补充、且会改变正确行动的关键条件？"),
        "context_overload_or_noise": (
            "上下文噪声",
            "可见上下文是否有明显冗余、冲突或与任务无关的材料？",
        ),
        "low_defect_state": ("低缺陷输入", "是否没有上述明显缺陷，直接执行较为合理？"),
        "reformulate_eligible": ("允许改写", "保持实体、数值和约束不变时，改写是否适用？"),
        "decompose_eligible": ("允许分解", "不增加目标或更改约束时，分解是否适用？"),
        "acquire_eligible": ("允许补充证据", "是否可通过已有工具／公开材料补充所需信息？"),
        "clarify_eligible": ("允许询问", "向用户询问关键缺失条件是否适用？"),
        "curate_eligible": ("允许整理上下文", "不删除任务必需信息时，整理或去噪是否适用？"),
    },
}
OPTIONS = {"yes", "no", "uncertain"}


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def create_json(path, value):
    with Path(path).open("x", encoding="utf8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.chmod(path, 0o600)


class ReviewStudy:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.manifest = json.loads((self.root / "assignment_manifest.json").read_text())
        unhashed = {k: v for k, v in self.manifest.items() if k != "assignment_hash"}
        if self.manifest["assignment_hash"] != content_hash(unhashed):
            raise ValueError("Assignment manifest changed")
        self.secret = json.loads((self.root / "private/reviewer_access.json").read_text())
        self.items = {}
        self.assignment = {}
        self.short = {}
        self.lock = threading.RLock()
        for reviewer, expected in self.manifest["packet_hashes"].items():
            packet = json.loads((self.root / "private" / f"{reviewer}.json").read_text())
            if content_hash(packet) != expected:
                raise ValueError("Reviewer packet changed")
            for entry in packet:
                blind = entry["blind_id"]
                self.items[blind] = entry
        for i, row in enumerate(self.manifest["assignments"], 1):
            blind = row["blind_id"]
            self.assignment[blind] = row["assigned_annotators"]
            self.short[blind] = f"S{i:03d}"
        if set(self.items) != set(self.assignment):
            raise ValueError("Item assignment differs")
        for entry in self.items.values():
            MeasurementItem("validation", entry["group_hash"], entry["kind"], entry["payload"])
        self.db = self.root / "private/reviews.sqlite3"
        with self.connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, reviewer TEXT NOT NULL, blind TEXT NOT NULL, stage TEXT NOT NULL, action TEXT NOT NULL, labels TEXT NOT NULL, notes TEXT NOT NULL, utc TEXT NOT NULL)"
            )
            db.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS final_vote ON events(reviewer,blind,stage) WHERE action='submit'"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS sessions (token_hash TEXT PRIMARY KEY, reviewer TEXT NOT NULL, created TEXT NOT NULL)"
            )
        os.chmod(self.db, 0o600)

    def connect(self):
        db = sqlite3.connect(self.db, timeout=15)
        db.row_factory = sqlite3.Row
        return db

    def login(self, reviewer, code, independent):
        if independent is not True:
            raise ValueError("请确认由你本人独立评审")
        expected = self.secret.get(reviewer, {}).get("code_sha256", "")
        if (
            not isinstance(code, str)
            or not expected
            or not hmac.compare_digest(sha(code.encode()), expected)
        ):
            raise ValueError("评审 ID 或访问码不正确")
        token = secrets.token_urlsafe(32)
        with self.connect() as db:
            db.execute("INSERT INTO sessions VALUES(?,?,?)", (sha(token.encode()), reviewer, utc()))
        return token

    def reviewer(self, token):
        if not isinstance(token, str):
            raise ValueError("请先登录")
        with self.connect() as db:
            row = db.execute(
                "SELECT reviewer FROM sessions WHERE token_hash=?", (sha(token.encode()),)
            ).fetchone()
        if not row:
            raise ValueError("请先登录")
        return row["reviewer"]

    def latest(self, reviewer, blind, stage):
        with self.connect() as db:
            return db.execute(
                "SELECT * FROM events WHERE reviewer=? AND blind=? AND stage=? ORDER BY id DESC LIMIT 1",
                (reviewer, blind, stage),
            ).fetchone()

    def votes(self, blind):
        with self.connect() as db:
            return db.execute(
                "SELECT * FROM events WHERE blind=? AND stage='first_pass' AND action='submit'",
                (blind,),
            ).fetchall()

    def stage(self, reviewer, blind):
        if blind not in self.assignment:
            raise ValueError("样本不存在")
        if reviewer in self.assignment[blind]:
            return "first_pass"
        votes = self.votes(blind)
        if len(votes) == 2 and votes[0]["labels"] != votes[1]["labels"]:
            return "adjudication"
        raise ValueError("这个样本未分配给你")

    def status(self, reviewer):
        items = []
        for blind in sorted(self.items, key=lambda b: sha((reviewer + b).encode())):
            try:
                stage = self.stage(reviewer, blind)
            except ValueError:
                continue
            latest = self.latest(reviewer, blind, stage)
            items.append(
                {
                    "sample_id": self.short[blind],
                    "stage": stage,
                    "submitted": bool(latest and latest["action"] == "submit"),
                    "draft": bool(latest and latest["action"] == "draft"),
                }
            )
        return {
            "reviewer_id": reviewer,
            "items": items,
            "submitted": sum(r["submitted"] for r in items),
            "total": len(items),
        }

    def item(self, reviewer, sample_id):
        blind = next((b for b, s in self.short.items() if s == sample_id), None)
        stage = self.stage(reviewer, blind)
        entry = self.items[blind]
        latest = self.latest(reviewer, blind, stage)
        return {
            "sample_id": sample_id,
            "stage": stage,
            "kind": entry["kind"],
            "payload": entry["payload"],
            "fields": [
                {"id": k, "title": v[0], "help": v[1]} for k, v in FIELDS[entry["kind"]].items()
            ],
            "labels": json.loads(latest["labels"]) if latest else {},
            "notes": latest["notes"] if latest else "",
            "submitted": bool(latest and latest["action"] == "submit"),
        }

    def record(self, reviewer, sample_id, labels, notes, submit):
        blind = next((b for b, s in self.short.items() if s == sample_id), None)
        stage = self.stage(reviewer, blind)
        allowed = FIELDS[self.items[blind]["kind"]]
        if (
            not isinstance(labels, dict)
            or set(labels) - set(allowed)
            or any(v not in OPTIONS for v in labels.values())
        ):
            raise ValueError("标注字段或选项不合法")
        if submit and set(labels) != set(allowed):
            raise ValueError("每项都需选择；无法判断请选择“不确定”")
        if not isinstance(notes, str) or len(notes) > 4000:
            raise ValueError("备注最长 4000 字符")
        if submit and not notes.strip():
            raise ValueError("请提供依据、原文摘录或无法判断的原因")
        with self.lock, self.connect() as db:
            if db.execute(
                "SELECT 1 FROM events WHERE reviewer=? AND blind=? AND stage=? AND action='submit'",
                (reviewer, blind, stage),
            ).fetchone():
                raise ValueError("已提交标注不可覆盖")
            db.execute(
                "INSERT INTO events(reviewer,blind,stage,action,labels,notes,utc) VALUES(?,?,?,?,?,?,?)",
                (
                    reviewer,
                    blind,
                    stage,
                    "submit" if submit else "draft",
                    json.dumps(labels, sort_keys=True),
                    notes,
                    utc(),
                ),
            )
        return {"saved": True, "submitted": bool(submit), "saved_at": utc()}

    def export(self, path):
        with self.connect() as db:
            rows = db.execute("SELECT * FROM events WHERE action='submit' ORDER BY id").fetchall()
        annotations = [
            {
                "blind_id": r["blind"],
                "annotator_id": r["reviewer"],
                "annotation_stage": r["stage"],
                "labels": json.loads(r["labels"]),
            }
            for r in rows
        ]
        output = {
            "schema": "independent-human-input-review-export.v1",
            "assignment_hash": self.manifest["assignment_hash"],
            "annotations": annotations,
            "evidence_notes": [
                {
                    "blind_id": r["blind"],
                    "annotator_id": r["reviewer"],
                    "notes": r["notes"],
                    "utc": r["utc"],
                }
                for r in rows
            ],
            "scope": "input defects, eligibility and original-input anchor inventories only; not trajectory preservation or execution success",
        }
        try:
            analysis = analyze_measurement_annotations(self.manifest, annotations)
            # The legacy analyzer maps an all-one-category pair to 1. Report the
            # mathematically undefined chance-corrected coefficients explicitly here.
            kinds = {b: self.items[b]["kind"] for b in self.items}
            for field in analysis["agreement"]:
                values = [
                    r["labels"][field["field"]]
                    for r in annotations
                    if r["annotation_stage"] == "first_pass"
                    and kinds[r["blind_id"]] == field["kind"]
                ]
                field["category_counts"] = {v: values.count(v) for v in sorted(set(values))}
                field["uncertain_first_pass_count"] = values.count("uncertain")
                field["coefficient_status"] = "defined"
                if len(set(values)) == 1:
                    field["cohen_kappa"] = None
                    field["krippendorff_alpha_nominal"] = None
                    field["coefficient_status"] = "undefined_single_category"
            analysis.pop("analysis_hash", None)
            analysis["analysis_hash"] = content_hash(analysis)
            output["analysis"] = analysis
        except ValueError as exc:
            output["analysis_complete"] = False
            output["incomplete_reason"] = str(exc)
        create_json(path, output)
        return {
            "submitted_votes": len(rows),
            "analysis_complete": output.get("analysis", {}).get("analysis_complete", False),
        }


def handler(study):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, status, body, mime="application/json"):
            raw = (
                json.dumps(body, ensure_ascii=False).encode()
                if mime == "application/json"
                else body
            )
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(raw)

        def valid_origin(self):
            host = self.headers.get("Host", "")
            expected = "127.0.0.1:" + str(self.server.server_port)
            if host != expected:
                raise ValueError("Host not allowed")
            origin = self.headers.get("Origin")
            if origin is not None and origin != "http://" + expected:
                raise ValueError("Origin not allowed")

        def identity(self):
            return study.reviewer(self.headers.get("Authorization", "").removeprefix("Bearer "))

        def do_GET(self):
            try:
                self.valid_origin()
                path = urlsplit(self.path).path
                if path == "/api/status":
                    return self.send(200, study.status(self.identity()))
                if path.startswith("/api/items/"):
                    return self.send(200, study.item(self.identity(), path.rsplit("/", 1)[1]))
                name = {
                    "/": "review.html",
                    "/review.js": "review.js",
                    "/review.css": "review.css",
                }.get(path)
                if not name:
                    return self.send(404, {"error": "Not found"})
                mime = {
                    "review.html": "text/html; charset=utf-8",
                    "review.js": "application/javascript; charset=utf-8",
                    "review.css": "text/css; charset=utf-8",
                }[name]
                return self.send(200, (Path(__file__).parent / "static" / name).read_bytes(), mime)
            except (ValueError, KeyError, TypeError):
                return self.send(403, {"error": "无权限访问；请检查登录或样本 ID"})

        def do_POST(self):
            try:
                self.valid_origin()
                if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                    raise ValueError("JSON required")
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 32768:
                    raise ValueError("Invalid request size")
                value = json.loads(self.rfile.read(size))
                path = urlsplit(self.path).path
                if path == "/api/login":
                    return self.send(
                        200,
                        {
                            "token": study.login(
                                value["reviewer_id"], value["code"], value.get("independent")
                            )
                        },
                    )
                if path == "/api/review":
                    return self.send(
                        200,
                        study.record(
                            self.identity(),
                            value["sample_id"],
                            value["labels"],
                            value.get("notes", ""),
                            value.get("submit") is True,
                        ),
                    )
                return self.send(404, {"error": "Not found"})
            except (ValueError, KeyError, TypeError, sqlite3.IntegrityError) as exc:
                return self.send(400, {"error": str(exc)})

    return Handler


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--port", type=int, default=7864)
    p.add_argument("--export", type=Path)
    a = p.parse_args()
    study = ReviewStudy(a.study)
    if a.export:
        print(json.dumps(study.export(a.export)))
        return
    server = ThreadingHTTPServer(("127.0.0.1", a.port), handler(study))
    print(f"Independent input review: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

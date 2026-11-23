from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.client import HTTPConnection
from pathlib import Path

from agentinputlab.annotation import (
    AnnotationStore,
    AnnotationStoreError,
    validate_annotation_selection,
)
from agentinputlab.annotation.server import create_server


def _items(path: Path) -> None:
    path.write_text(
        "".join(
            json.dumps({"item_id": item_id, "payload": {"query": item_id}}) + "\n"
            for item_id in ("item-1", "item-2")
        ),
        encoding="utf-8",
    )


def _complete(store: AnnotationStore, version_id: str) -> None:
    for item_id in ("item-1", "item-2"):
        store.save(version_id, {"item_id": item_id, "labels": {"valid": "yes"}})
    store.complete(version_id)


class AnnotationStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._temporary_directory.name)

    def tearDown(self) -> None:
        self._temporary_directory.cleanup()

    def test_completed_version_requires_explicit_selection_and_reannotation_is_create_only(
        self,
    ) -> None:
        items = self.tmp_path / "items.jsonl"
        root = self.tmp_path / "annotations"
        _items(items)
        store = AnnotationStore(items_path=items, root=root, kind="input_audit")
        first = store.create_version(annotator_id="auditor-a")
        _complete(store, first["version_id"])

        with self.assertRaisesRegex(AnnotationStoreError, "no version is selected"):
            validate_annotation_selection(root, kind="input_audit")
        selection = store.select(first["version_id"])
        self.assertEqual(
            validate_annotation_selection(root, kind="input_audit"),
            selection,
        )
        with self.assertRaisesRegex(AnnotationStoreError, "immutable"):
            store.save(
                first["version_id"],
                {"item_id": "item-1", "labels": {"valid": "no"}},
            )

        second = store.create_version(annotator_id="auditor-b")
        self.assertEqual(second["version_id"], "input_audit-v002")
        self.assertIs(second["supersedes_completed_version"], True)
        self.assertIs(store.status()["requires_reuse_or_new_version_choice"], True)

    def test_absent_or_incomplete_annotations_do_not_block_execution(self) -> None:
        root = self.tmp_path / "annotations"
        self.assertIsNone(validate_annotation_selection(root, kind="trajectory_audit"))

    def test_selected_completed_annotation_detects_data_tampering(self) -> None:
        items = self.tmp_path / "items.jsonl"
        root = self.tmp_path / "annotations"
        _items(items)
        store = AnnotationStore(items_path=items, root=root, kind="input_audit")
        version = store.create_version(annotator_id="auditor")
        _complete(store, version["version_id"])
        store.select(version["version_id"])
        annotations = root / version["version_id"] / "annotations.jsonl"
        annotations.write_text(
            annotations.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(AnnotationStoreError, "hash does not match"):
            validate_annotation_selection(root, kind="input_audit")
        with self.assertRaisesRegex(AnnotationStoreError, "unavailable or modified"):
            store.status()
        trajectory_store = AnnotationStore(
            items_path=items,
            root=root,
            kind="trajectory_audit",
        )
        trajectory_store.create_version(annotator_id="auditor")
        self.assertIsNone(validate_annotation_selection(root, kind="trajectory_audit"))

    def test_annotation_web_server_serves_ui_and_persists_new_version(self) -> None:
        items = self.tmp_path / "items.jsonl"
        _items(items)
        server = create_server(
            "127.0.0.1",
            0,
            items_path=items,
            output_root=self.tmp_path / "annotations",
            kind="input_audit",
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            connection = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
            connection.request("GET", "/")
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            self.assertIn(b"AgentInputLab", response.read())
            body = json.dumps({"action": "new", "annotator_id": "auditor"})
            connection.request(
                "POST",
                "/api/versions",
                body=body,
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            payload = json.loads(response.read())
            self.assertEqual(response.status, 201)
            self.assertEqual(payload["version_id"], "input_audit-v001")
            self.assertTrue(
                (self.tmp_path / "annotations/input_audit-v001/manifest.json").is_file()
            )
            connection.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

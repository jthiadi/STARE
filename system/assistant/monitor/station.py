"""PIC-reviewed station setup. References are optional; approval is not evidence."""
from __future__ import annotations

import hashlib
import io
import json
import re
import secrets
from pathlib import Path

from PIL import Image, ImageOps

from .workflow import validate_sop


EXTERNAL = re.compile(r"\b(torque|n[· .]?m|newton|pressure|voltage|continuity|leak[- ]?test|temperature|resistance|internal|electrical test)\b", re.I)
PART_CODE = re.compile(r"\b[A-Z]{1,6}[-_]\d{2,}[A-Za-z0-9-]*\b")


class Station:
    def __init__(self, sop, folder: Path | None = None):
        self.sop = validate_sop(sop)
        self.folder = folder
        self.approved = {}
        self.references = {}
        self.images = {}
        self.import_notes = []
        self.demo_profile = False

    def blockers(self, step):
        reasons = []
        if not self.approved.get(step["id"]):
            reasons.append("PIC review required")
        if not step["completion_check"]:
            reasons.append("Define what successful completion looks like")
        if step['verification'] == 'unspecified':
            reasons.append('Choose the verification method')
        elif step["verification"] != "visual" or EXTERNAL.search(step["instruction"] + " " + step["completion_check"]):
            reasons.append("Needs external measurement or human verification")
        refs = self.assigned(step['id'])
        if PART_CODE.search(step["instruction"]) and not refs:
            reasons.append("Unfamiliar part code: attach and label a visual reference")
        if refs and not refs[0]['label'].strip():
            reasons.append('Label the reference so its meaning is explicit')
        return reasons

    def assigned(self, step_id):
        return [r for r in self.references.values() if r.get("step_id") == step_id]

    def snapshot(self):
        rows = [{"step_id": s["id"], "approved": bool(self.approved.get(s["id"])),
                 "blockers": self.blockers(s), "references": self.assigned(s["id"]),
                 "missing_recovery": not bool(s["recovery"])} for s in self.sop["steps"]]
        return {"steps": rows, "ready_count": sum(not x["blockers"] for x in rows),
                "demo_profile": self.demo_profile,
                "references": list(self.references.values()), "import_notes": self.import_notes,
                "approval_is_not_validation": True}

    def review(self, step_id, fields, approved):
        if not isinstance(fields, dict):
            raise ValueError('Review fields must be an object.')
        if not isinstance(approved, bool):
            raise ValueError("Approval must be true or false.")
        steps = [dict(s) for s in self.sop["steps"]]
        step = next((s for s in steps if s["id"] == step_id), None)
        if step is None:
            raise ValueError("Unknown station step.")
        for key in ("parts", "tools", "completion_check", "verification", "recovery"):
            if key in fields:
                step[key] = fields[key]
        candidate = validate_sop({**self.sop, "steps": steps})
        self.sop = candidate
        self.approved[step_id] = approved
        self.save()

    def add_reference(self, content, name, step_id=None, label="", source="upload"):
        if len(self.references) >= 30 or len(content) > 8_000_000:
            raise ValueError("Maximum 30 reference images, each below 8 MB.")
        if step_id and step_id not in {s["id"] for s in self.sop["steps"]}:
            raise ValueError("Reference has an unknown SOP step.")
        with Image.open(io.BytesIO(content)) as image:
            if image.width * image.height > 16_000_000:
                raise ValueError("Reference image is too large.")
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((1024, 1024))
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=90)
        content = output.getvalue()
        rid = hashlib.sha256(content).hexdigest()[:20]
        # Equal images can still be associated with different steps.
        if rid in self.references:
            rid += secrets.token_hex(3)
        self.images[rid] = content
        self.references[rid] = {"id": rid, "name": str(name)[:160], "step_id": step_id or None,
                                "label": str(label)[:600], "source": source, "url": f"/api/reference/{rid}"}
        if step_id:
            self.approved.pop(step_id, None)
        self.save()
        return rid

    def assign_reference(self, rid, step_id, label):
        if rid not in self.references:
            raise ValueError("Unknown reference image.")
        if step_id and step_id not in {s["id"] for s in self.sop["steps"]}:
            raise ValueError("Unknown SOP step.")
        old = self.references[rid]["step_id"]
        self.references[rid].update(step_id=step_id or None, label=str(label)[:600])
        for changed in (old, step_id):
            self.approved.pop(changed, None)
        self.save()

    def image_bytes(self, rid):
        if rid not in self.references:
            raise ValueError("Unknown reference image.")
        if rid in self.images:
            return self.images[rid]
        if self.folder:
            return (self.folder / "references" / f"{rid}.jpg").read_bytes()
        raise ValueError("Reference image unavailable.")

    def save(self):
        if not self.folder:
            return
        self.folder.mkdir(parents=True, exist_ok=True)
        ref_dir = self.folder / "references"
        ref_dir.mkdir(exist_ok=True)
        for rid, content in self.images.items():
            path = ref_dir / f"{rid}.jpg"
            if not path.exists():
                path.write_bytes(content)
        data = {"schema": "sop-station-v1", "sop": self.sop, "approved": self.approved,
                "references": self.references, "import_notes": self.import_notes}
        temporary = self.folder / "station.tmp"
        temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(self.folder / "station.json")

    @classmethod
    def load(cls, sop, folder):
        if folder and (folder / "station.json").is_file():
            data = json.loads((folder / "station.json").read_text(encoding="utf-8"))
            if data.get("schema") != "sop-station-v1":
                raise ValueError("Unsupported saved station format.")
            instance = cls(data["sop"], folder)
            instance.approved = {k: v is True for k, v in data.get("approved", {}).items()}
            refs = data.get("references", {})
            if any(not re.fullmatch(r"[a-f0-9]{20,26}", k) for k in refs):
                raise ValueError("Invalid saved reference ID.")
            instance.references = refs
            instance.import_notes = data.get("import_notes", [])
            return instance
        return cls(sop, folder)

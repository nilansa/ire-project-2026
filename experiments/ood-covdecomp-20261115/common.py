from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"
RETRIEVAL = ROOT / "retrieval"
MUSIQUE_DEV = Path("/Users/nilan/Documents/ire-p5-minimal-evidence/stage1_musique/data/musique_ans_v1.0_dev.jsonl")
MUSIQUE_CORPUS = Path("/Users/nilan/Documents/ire-p5-minimal-evidence/stage1_musique/corpus.jsonl")
STAGE1_SAMPLE = Path("/Users/nilan/Documents/ire-p5-minimal-evidence/stage1_musique/sample_questions.json")
COLBERT_PATH = Path("/Users/nilan/Documents/ire-p5-minimal-evidence/stage2_musique/DISCo/checkpoints/colbertv2.0")
MODEL_NAME = "sentence-transformers/msmarco-distilbert-base-tas-b"
MODEL_REVISION = "b12d9352e776979147078a8975a4885042984fd1"
SALT = "p5-ood-20261115:"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def salted(qid: str) -> str:
    return hashlib.sha256((SALT + qid).encode("utf-8")).hexdigest()


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl_new(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_json_new(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")


def read_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


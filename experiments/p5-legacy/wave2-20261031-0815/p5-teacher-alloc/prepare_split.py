"""Rebuild and verify the frozen HotpotQA validation split without scoring it."""

import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


OUT = Path(__file__).resolve().parent
PILOT = Path("/Users/nilan/Documents/ire-p5-minimal-evidence/pilot")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def norm(value: str) -> str:
    return " ".join(str(value).split()).casefold()


def eligible(row: dict) -> bool:
    try:
        if row["type"] not in {"bridge", "comparison"}:
            return False
        titles = row["context"]["title"]
        sentences = row["context"]["sentences"]
        gold = {norm(title) for title in row["supporting_facts"]["title"]}
        return (
            len(titles) == 10
            and len(sentences) == 10
            and len(gold) == 2
            and sum(norm(title) in gold for title in titles) == 2
        )
    except (KeyError, TypeError):
        return False


def main() -> None:
    pilot = json.loads((PILOT / "results.json").read_text())
    source = PILOT / "data" / "hotpot_validation.parquet"
    source_bytes = source.read_bytes()
    if sha256(source_bytes) != pilot["data"]["sha256"]:
        raise ValueError("Pilot validation source hash changed")
    rows = pq.read_table(source).to_pylist()
    eligible_rows = [row for row in rows if eligible(row)]
    if len(eligible_rows) != 7342:
        raise ValueError(f"Expected 7342 eligible rows, got {len(eligible_rows)}")

    pilot_ids = sorted({str(row["qid"]) for row in pilot["query_info"]})
    if len(pilot_ids) != 600:
        raise ValueError("Pilot exclusion set must contain exactly 600 unique IDs")
    ordered = sorted(
        {str(row["id"]) for row in eligible_rows} - set(pilot_ids),
        key=lambda qid: (hashlib.sha256(qid.encode("utf-8")).hexdigest(), qid),
    )
    if len(ordered) != 6742:
        raise ValueError(f"Expected 6742 remaining IDs, got {len(ordered)}")
    holdout, dev2 = ordered[:2000], ordered[2000:]
    pilot_bytes = ("\n".join(pilot_ids) + "\n").encode()
    holdout_bytes = ("\n".join(holdout) + "\n").encode()

    manifest = json.loads((OUT / "split_manifest.json").read_text())
    checks = {
        "pilot_excluded_ids.txt": sha256(pilot_bytes),
        "seq_holdout_ids.txt": sha256(holdout_bytes),
    }
    for name, expected in checks.items():
        path = OUT / name
        if path.read_bytes() != (pilot_bytes if name.startswith("pilot_") else holdout_bytes):
            raise ValueError(f"{name} differs from the recomputed ID set")
        if sha256(path.read_bytes()) != expected:
            raise ValueError(f"{name} hash mismatch")
    if manifest["seq_holdout_ids_file_sha256"] != checks["seq_holdout_ids.txt"]:
        raise ValueError("Manifest holdout hash mismatch")
    if manifest["pilot_excluded_ids_sha256"] != checks["pilot_excluded_ids.txt"]:
        raise ValueError("Manifest pilot exclusion hash mismatch")
    if sha256(("\n".join(dev2) + "\n").encode()) != manifest["dev2_ids_sha256"]:
        raise ValueError("DEV2 ID hash mismatch")
    if set(pilot_ids) & set(holdout) or len(dev2) != 4742:
        raise ValueError("Split overlap or DEV2 size error")
    print(json.dumps({"status": "verified", "eligible": 7342, "pilot_excluded": 600,
                      "seq_holdout": len(holdout), "dev2": len(dev2),
                      "seq_holdout_sha256": checks["seq_holdout_ids.txt"],
                      "dev2_ids_sha256": manifest["dev2_ids_sha256"]}, indent=2))


if __name__ == "__main__":
    main()

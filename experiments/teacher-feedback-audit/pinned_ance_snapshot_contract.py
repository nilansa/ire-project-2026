#!/usr/bin/env python3
"""CPU-only tokenizer/module metadata pin for the historical ANCE snapshot.

A state_dict SHA-256 attests parameter tensors, not tokenization, pooling,
truncation or module configuration. This guards those *additional* bytes on
both future untouched and trained exact-evaluation paths. No model import,
forward pass, teacher call or optimizer execution occurs here.

Hashes were recorded from the original local snapshot of revision
ff271583209552b908399e7059ac79a2d88dccc3, independently of the new
training adapter. Do not substitute a later revision without a new protocol.
"""
import argparse
import hashlib
import json
from pathlib import Path

MODEL_ID = "sentence-transformers/msmarco-roberta-base-ance-firstp"
PINNED_REVISION = "ff271583209552b908399e7059ac79a2d88dccc3"

# Non-weight files that can affect model structure, pooling, text input or
# token IDs. Model weights are separately guarded by the original state hash.
METADATA_SHA256 = {
    "1_Pooling/config.json": "c9bef85e8bbf4b2eab4941b3fb62bd33f88686748b478f2e264d256472d9643b",
    "2_Dense/config.json": "f40a3730d1f27b5fc5389d4bfd24c320832a94d51f598386ce123a4879812001",
    "3_LayerNorm/config.json": "614f3a8c484c82841f606fbbd0f011658908892b249a89e7d503f765c219378d",
    "config.json": "7cc42679c3729174102fcce22edffbc01059b6c6c5b0192e8b09d0ab5e85250c",
    "config_sentence_transformers.json": "b8c64b5cece00d8424b4896ea75b512b6008576088497609dfeb6bd63e6d36b8",
    "merges.txt": "fe36cab26d4f4421ed725e10a2e9ddb7f799449c603a96e7f29b5a3c82a95862",
    "modules.json": "525ad50fbac17d55949b13c9281c584b44540d83f2de5f7bd7ce71cdc92a173a",
    "sentence_bert_config.json": "84e39fda68ccbff05bfa723ae9c0e70e23e2ec373b76e0f8c6e71af72a693cbf",
    "special_tokens_map.json": "378eb3bf733eb16e65792d7e3fda5b8a4631387ca04d2015199c4d4f22ae554d",
    "tokenizer.json": "33465117406b9007673e8ba283f7f1383d9b5094df947481af60eec94ed7d7bd",
    "tokenizer_config.json": "19abf3bb202a2530a61979f8589a82d3e2e06c5f32a8daafa4872844a2dc1b18",
    "vocab.json": "ed19656ea1707df69134c4af35c8ceda2cc9860bf2c3495026153a133670ab5e",
}


def verify_ance_snapshot_metadata(snapshot_dir):
    """Reject missing or altered pinned non-weight files before model loading.

    The state_dict must still independently match the original parameter hash
    after model loading. This does not establish numerical evaluator parity.
    """
    root = Path(snapshot_dir).expanduser()
    if not root.is_dir():
        raise FileNotFoundError("Pinned local ANCE snapshot directory unavailable")
    for name, expected in sorted(METADATA_SHA256.items()):
        candidate = root / name
        if not candidate.is_file():
            raise FileNotFoundError("Pinned ANCE metadata missing: " + name)
        actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError("Pinned ANCE tokenizer/module metadata mismatch: " + name)
    return {
        "student_model_id": MODEL_ID,
        "student_revision": PINNED_REVISION,
        "verified_metadata_files": len(METADATA_SHA256),
        "metadata_manifest_sha256": hashlib.sha256(
            json.dumps(METADATA_SHA256, sort_keys=True,
                       separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "weights_verification": "separate state_dict hash required after model load",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", required=True)
    parser.add_argument("--snapshot", required=True)
    args = parser.parse_args()
    result = verify_ance_snapshot_metadata(args.snapshot)
    print("PINNED_ANCE_SNAPSHOT_METADATA_PASS")
    print(json.dumps(result, sort_keys=True))

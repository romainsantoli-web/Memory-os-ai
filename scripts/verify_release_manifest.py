"""Fail closed before upload; SHA-256 is not itself a release approval."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("release_ready") is not True:
        print("REFUS : release_ready != true ; corriger/revoir/requalifier avant PyPI.")
        return 1
    if manifest.get("version") != "4.0.1" or not re.fullmatch(r"[0-9a-f]{40}", manifest.get("source_commit", "")):
        print("REFUS : identité de version/source absente ou invalide.")
        return 1
    expected = {"memory_os_ai-4.0.1-py3-none-any.whl", "memory_os_ai-4.0.1.tar.gz"}
    artifacts = manifest.get("artifacts", [])
    if {a["name"] for a in artifacts} != expected or len(artifacts) != 2:
        print("REFUS : inventaire wheel/sdist incorrect.")
        return 1
    for artifact in artifacts:
        path = args.manifest.parent / "artifacts" / artifact["name"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != artifact["sha256"]:
            print(f"REFUS : artefact absent/modifié : {artifact['name']}")
            return 1
    print("Manifeste approuvé et empreintes concordantes ; ne pas reconstruire avant upload.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

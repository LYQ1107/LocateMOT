#!/usr/bin/env python3
"""Freeze the complete LVIS v1 names and original base/novel frequencies."""
import ast
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
SOURCE = "https://raw.githubusercontent.com/facebookresearch/detectron2/main/detectron2/data/datasets/lvis_v1_categories.py"
SOURCE_SHA = "2fd8d25a76e465710bb87af0f9d7591ff3292e2eeb14bf921a6277c4fe67fc6a"


def main():
    directory = ROOT / "outputs/unified/u2/protocol"
    path = directory / "sources/lvis_v1_categories.py"
    if not path.exists():
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(opener.open(SOURCE, timeout=30).read())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SOURCE_SHA:
        raise RuntimeError("LVIS category metadata differs from the registered source")
    categories = next(ast.literal_eval(node.value) for node in ast.parse(path.read_text()).body
                      if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "LVIS_CATEGORIES" for target in node.targets))
    assert len(categories) == 1203
    result = {"source": SOURCE.replace("raw.githubusercontent.com/", "github.com/").replace("/main/", "/blob/main/"),
              "source_sha256": digest, "categories": categories,
              "base_rule": "LVIS frequency f or c", "novel_rule": "LVIS frequency r"}
    output = directory / "lvis_v1_vocabulary.json"
    text = json.dumps(result, indent=2) + "\n"
    if output.exists() and output.read_text() != text:
        raise RuntimeError("An existing frozen vocabulary differs")
    output.write_text(text)
    print(json.dumps({"categories": 1203, "base": sum(c["frequency"] in ("f", "c") for c in categories),
                      "novel": sum(c["frequency"] == "r" for c in categories), "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()

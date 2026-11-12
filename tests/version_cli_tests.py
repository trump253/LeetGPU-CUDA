"""Check all exercise version interfaces without a visible GPU or completed kernels."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
binary_dir = Path(sys.argv[1])
environment = dict(os.environ, CUDA_VISIBLE_DEVICES="")


def invoke(binary, *args, expected=0):
    result = subprocess.run([str(binary), *args], env=environment, capture_output=True,
                            text=True, timeout=20)
    if result.returncode != expected:
        raise RuntimeError(f"{binary.name} {args}: exit {result.returncode}\n{result.stdout}\n{result.stderr}")
    return result


for problem in json.loads((root / "scripts/problems.json").read_text()):
    binary = binary_dir / problem["target"]
    listing = invoke(binary, "--list-versions").stdout
    versions = []
    for line in listing.splitlines():
        match = re.fullmatch(r"\s+(\d+)\s+(.+?)( \(default\))?", line)
        if match:
            versions.append((int(match[1]), match[2], bool(match[3])))
    if not versions or sum(v[2] for v in versions) != 1:
        raise RuntimeError(f"{binary.name}: missing versions or ambiguous default\n{listing}")
    cases = invoke(binary, "--list-cases", "--version", "all").stdout
    if len(cases.splitlines()) != problem["cases"]:
        raise RuntimeError(f"{binary.name}: case listing duplicated across versions")
    for version_id, name, is_default in versions:
        for selector in (str(version_id), name):
            if invoke(binary, "--list-cases", "--version", selector).stdout != cases:
                raise RuntimeError(f"{binary.name}: selector {selector} changed test cases")
    unknown = str(max(v[0] for v in versions) + 1)
    rejected = invoke(binary, "--version", unknown, "--check-only", expected=2)
    if "Unknown solution version" not in rejected.stderr:
        raise RuntimeError(f"{binary.name}: invalid version reached CUDA initialization")
    if "--list-versions" not in invoke(binary, "--help").stdout:
        raise RuntimeError(f"{binary.name}: missing version help")

print("PASS: all 26 exercise version lists, ID/name selectors and errors without a visible GPU")

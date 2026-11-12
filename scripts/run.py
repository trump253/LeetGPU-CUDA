#!/usr/bin/env python3
"""Configure/build/run one exercise without external Python packages."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PROBLEMS = json.loads((ROOT / "scripts/problems.json").read_text())


def problem_for(selector):
    if selector.isdigit():
        matches = [p for p in PROBLEMS if p["id"] == int(selector)]
    else:
        path = Path(selector).resolve()
        matches = [p for p in PROBLEMS
                   if selector == p["target"] or selector == Path(p["directory"]).name
                   or path == ROOT / p["directory"] or ROOT / p["directory"] in path.parents]
    if len(matches) != 1:
        raise ValueError(f"Unknown exercise: {selector}; use --list")
    return matches[0]


def run(args):
    print("+", " ".join(map(str, args)), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def main():
    # Everything following -- is forwarded verbatim to the exercise binary.
    argv = sys.argv[1:]
    split = argv.index("--") if "--" in argv else len(argv)
    forwarded = argv[split + 1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exercise", nargs="?", help="01..26, target, or file/directory path")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--preset", choices=["release", "debug"], default="release")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("-j", "--jobs", type=int, default=4)
    args = parser.parse_args(argv[:split])
    if args.list:
        for p in PROBLEMS:
            print(f'{p["id"]:02d}  {p["target"]:<30} {p["directory"]}')
        return 0
    if not args.exercise:
        parser.error("exercise is required unless --list is used")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    p = problem_for(args.exercise)
    if not args.no_build:
        run(["cmake", "--preset", args.preset])
        run(["cmake", "--build", "--preset", args.preset, "--target", p["target"], "-j", str(args.jobs)])
    if not args.build_only:
        run([str(ROOT / "build" / args.preset / "bin" / p["target"]), *forwarded])
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

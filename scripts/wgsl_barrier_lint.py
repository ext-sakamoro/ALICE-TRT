#!/usr/bin/env python3
"""No compute entry point returns before one of its workgroup barriers.

Every invocation of a workgroup must reach each `workgroupBarrier()` /
`storageBarrier()` / `workgroupUniformLoad()` of the entry point. A `return`
(or `discard`) that runs before such a barrier is undefined behaviour even when
the condition is uniform across the workgroup, and it differs by backend: the
tiled ternary matvec kernel, written that way, ran on Metal and Vulkan
lavapipe but crashed the process on the DX12 WARP adapter. The fix is to keep
every invocation in the barrier sequence and guard only the writes.

The check reads WGSL in `*.wgsl` files and in Rust string literals under `src/`
(this crate keeps its kernels as `&str` constants), finds each `@compute` entry
point, and fails when a `return` / `discard` token appears before the last
barrier of that entry point's body. Comments are ignored. It also fails when it
found no compute entry point at all (a scan that compares nothing).

Usage: `python3 scripts/wgsl_barrier_lint.py [--root DIR]`
"""

from __future__ import annotations

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.dirname(HERE)

ENTRY_RE = re.compile(r"@compute[^{]*?\bfn\s+(\w+)\s*\([^)]*\)[^{]*\{", re.S)
BARRIER_RE = re.compile(r"\b(?:workgroupBarrier|storageBarrier|workgroupUniformLoad)\b")
EXIT_RE = re.compile(r"\b(?:return|discard)\b")
LINE_COMMENT_RE = re.compile(r"//[^\n]*")
BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.S)


def strip_comments(text: str) -> str:
    """Blank comments out, keeping newlines so line numbers stay right."""
    def blank(m: re.Match[str]) -> str:
        return re.sub(r"[^\n]", " ", m.group(0))
    return LINE_COMMENT_RE.sub(blank, BLOCK_COMMENT_RE.sub(blank, text))


def entry_points(text: str):
    """Yield (name, offset of the entry, body text) for each @compute fn."""
    for m in ENTRY_RE.finditer(text):
        i, depth = m.end(), 1
        while i < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        yield m.group(1), m.start(), text[m.end():i]


def violations(text: str, rel: str) -> tuple[list[str], int]:
    found: list[str] = []
    count = 0
    clean = strip_comments(text)
    for name, start, body in entry_points(clean):
        count += 1
        barriers = [m.start() for m in BARRIER_RE.finditer(body)]
        if not barriers:
            continue
        last = barriers[-1]
        body_start = clean.index(body, start)
        for m in EXIT_RE.finditer(body):
            if m.start() < last:
                line = clean.count("\n", 0, body_start + m.start()) + 1
                found.append(f"{rel}:{line}: `{m.group(0)}` in entry point `{name}` before its last workgroup barrier")
    return found, count


def scan(root: str) -> tuple[list[str], int, int]:
    errors: list[str] = []
    files = entries = 0
    for base, exts in ((os.path.join(root, "src"), (".rs", ".wgsl")), (os.path.join(root, "shaders"), (".wgsl",))):
        if not os.path.isdir(base):
            continue
        for dirpath, _dirs, names in os.walk(base):
            for name in sorted(names):
                if not name.endswith(exts):
                    continue
                path = os.path.join(dirpath, name)
                with open(path, encoding="utf-8") as f:
                    text = f.read()
                if "@compute" not in text:
                    continue
                files += 1
                rel = os.path.relpath(path, root).replace(os.sep, "/")
                found, n = violations(text, rel)
                entries += n
                errors.extend(found)
    return errors, files, entries


def main(argv: list[str]) -> int:
    root = DEFAULT_ROOT
    if "--root" in argv:
        root = argv[argv.index("--root") + 1]
    errors, files, entries = scan(root)
    for e in errors:
        print(f"error: {e}", file=sys.stderr)
    print(f"compared: files {files}, compute entry points {entries}, violations {len(errors)}")
    if entries == 0:
        print("error: no compute entry point found (the scan compared nothing)", file=sys.stderr)
        return 1
    if errors:
        return 1
    print("wgsl-barrier-lint: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

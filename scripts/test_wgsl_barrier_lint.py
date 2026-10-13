#!/usr/bin/env python3
"""Tests for wgsl_barrier_lint.py. Usage: python3 scripts/test_wgsl_barrier_lint.py"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wgsl_barrier_lint as lint  # noqa: E402

GOOD = """
var<workgroup> s: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) {
    let in_bounds = wg.x < params.n;
    s[lid.x] = select(0.0, 1.0, in_bounds);
    workgroupBarrier();
    if (lid.x == 0u && in_bounds) { out[wg.x] = s[0]; }
}
"""

EARLY_RETURN = GOOD.replace(
    "    let in_bounds = wg.x < params.n;\n",
    "    if (wg.x >= params.n) {\n        return;\n    }\n    let in_bounds = true;\n",
)


def run(files: dict[str, str]) -> tuple[list[str], int, int]:
    with tempfile.TemporaryDirectory() as d:
        for rel, text in files.items():
            path = os.path.join(d, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        return lint.scan(d)


class BarrierLint(unittest.TestCase):
    def test_clean_kernel_passes(self):
        errors, files, entries = run({"src/k.wgsl": GOOD})
        self.assertEqual((errors, files, entries), ([], 1, 1))

    def test_early_return_before_barrier_fails(self):
        errors, _, _ = run({"src/k.wgsl": EARLY_RETURN})
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("`return` in entry point `main`", errors[0])

    def test_kernel_in_a_rust_string_constant_is_scanned(self):
        rust = 'pub const K: &str = r"\n' + EARLY_RETURN + '\n";\n'
        errors, files, entries = run({"src/kernel.rs": rust})
        self.assertEqual((len(errors), files, entries), (1, 1, 1), errors)

    def test_return_after_the_last_barrier_is_allowed(self):
        late = GOOD.replace("    if (lid.x == 0u && in_bounds)", "    if (!in_bounds) { return; }\n    if (lid.x == 0u)")
        errors, _, _ = run({"src/k.wgsl": late})
        self.assertEqual(errors, [])

    def test_return_between_two_barriers_fails(self):
        two = GOOD.replace("    workgroupBarrier();\n", "    workgroupBarrier();\n    if (lid.x > 7u) { return; }\n    workgroupBarrier();\n")
        errors, _, _ = run({"src/k.wgsl": two})
        self.assertEqual(len(errors), 1, errors)

    def test_discard_counts_like_return(self):
        disc = EARLY_RETURN.replace("return;", "discard;")
        errors, _, _ = run({"src/k.wgsl": disc})
        self.assertEqual(len(errors), 1, errors)

    def test_a_return_inside_a_comment_is_ignored(self):
        commented = GOOD.replace("    workgroupBarrier();", "    // return early here? no\n    workgroupBarrier();")
        errors, _, _ = run({"src/k.wgsl": commented})
        self.assertEqual(errors, [])

    def test_entry_point_without_barrier_may_return_early(self):
        nobar = "@compute @workgroup_size(1)\nfn f(@builtin(global_invocation_id) g: vec3<u32>) {\n  if (g.x > 3u) { return; }\n  out[g.x] = 1.0;\n}\n"
        errors, files, entries = run({"src/k.wgsl": nobar})
        self.assertEqual((errors, entries), ([], 1))

    def test_nothing_scanned_is_reported(self):
        errors, files, entries = run({"src/lib.rs": "pub fn f() {}\n"})
        self.assertEqual((files, entries), (0, 0))
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "src"))
            self.assertEqual(lint.main(["--root", d]), 1)

    def test_the_real_tree_is_clean_and_not_empty(self):
        errors, files, entries = lint.scan(lint.DEFAULT_ROOT)
        self.assertEqual(errors, [])
        self.assertGreater(entries, 0)


if __name__ == "__main__":
    sys.exit(unittest.main())

#!/usr/bin/env python3
"""Tests for scripts/cargo_test_gpu.sh: every GPU test run it starts is serialized (`--test-threads=1`), whatever the
caller passes, and a caller that asks for parallel threads is refused before cargo runs.

The GPU tests hang when they create wgpu devices from parallel threads on NVIDIA's proprietary Vulkan driver
(reproduced on an A6000 and a P5000; Apple Metal does not hang), so the wrapper, not each caller, holds the rule.
A fake `cargo` on PATH records its arguments, so no build or GPU is needed."""
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts" / "cargo_test_gpu.sh"

FAKE_CARGO = """#!/bin/sh
printf '%s\\n' "$@" > "$CARGO_ARGS_FILE"
echo "[gpu] test device acquired"
echo "test result: ok. 1 passed; 0 failed"
"""

# the self-test run must fail with the requirement's panic, as a real hidden adapter makes it
FAKE_CARGO_NO_ADAPTER = """#!/bin/sh
printf '%s\\n' "$@" > "$CARGO_ARGS_FILE"
echo "TRT_REQUIRE_GPU is set but no GPU adapter is available: hidden"
exit 101
"""


class Wrapper(unittest.TestCase):
    def run_wrapper(self, *args: str, cargo: str = FAKE_CARGO) -> tuple[subprocess.CompletedProcess, list[str] | None]:
        with tempfile.TemporaryDirectory() as d:
            bin_ = Path(d) / "bin"
            bin_.mkdir()
            (bin_ / "cargo").write_text(cargo)
            (bin_ / "cargo").chmod(0o755)
            record = Path(d) / "args"
            env = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "CARGO_ARGS_FILE": str(record)}
            p = subprocess.run(["bash", str(WRAPPER), *args], capture_output=True, text=True, env=env, timeout=60)
            got = record.read_text().splitlines() if record.exists() else None
            return p, got

    def assert_serialized(self, got: list[str] | None) -> None:
        self.assertIsNotNone(got, "cargo was not run")
        self.assertIn("--", got)
        after = got[got.index("--") + 1:]
        self.assertEqual([a for a in after if a.startswith("--test-threads")], ["--test-threads=1"], got)

    def test_a_run_without_libtest_arguments_is_serialized(self):
        p, got = self.run_wrapper("--lib", "--features", "fix128-arithmetic", "fix128")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assert_serialized(got)
        self.assertEqual(got[:4], ["test", "--lib", "--features", "fix128-arithmetic"])

    def test_a_run_with_libtest_arguments_keeps_them_and_is_serialized(self):
        p, got = self.run_wrapper("--lib", "--", "--nocapture")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assert_serialized(got)
        self.assertIn("--nocapture", got[got.index("--") + 1:])

    def test_a_caller_that_already_serializes_is_not_doubled(self):
        p, got = self.run_wrapper("--lib", "--", "--nocapture", "--test-threads=1")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assert_serialized(got)

    def test_parallel_threads_are_refused_before_cargo_runs(self):
        for extra in (["--test-threads=4"], ["--test-threads", "2"], ["--test-threads=0"]):
            p, got = self.run_wrapper("--lib", "--", *extra)
            self.assertNotEqual(p.returncode, 0, extra)
            self.assertIn("--test-threads=1", p.stderr)
            self.assertIsNone(got, f"cargo ran with {extra}")

    def test_the_self_test_run_is_serialized_too(self):
        p, got = self.run_wrapper("--self-test", "--lib", "fix128::tests::x", cargo=FAKE_CARGO_NO_ADAPTER)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assert_serialized(got)


if __name__ == "__main__":
    unittest.main()

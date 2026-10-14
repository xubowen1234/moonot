#!/usr/bin/env python3
"""Compile and execute the actual MoonBit code blocks from README.md.

The temporary example imports this checkout's public package. No published
package or existing example is substituted for the documentation snippets.
"""
# SPDX-License-Identifier: Apache-2.0
import argparse
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


def verify(root, target):
    blocks = re.findall(r"```moonbit\s*\n(.*?)```", (root / "README.md").read_text(encoding="utf-8"), re.S)
    if len(blocks) != 3 or '"xubowen1234/moonot"' not in blocks[0]:
        raise ValueError("Expected one public import block and two runnable MoonBit snippets")
    examples = (root / "examples").resolve()
    temporary = Path(tempfile.mkdtemp(prefix="readme-verify-", dir=str(examples))).resolve()
    if temporary.parent != examples or not temporary.name.startswith("readme-verify-"):
        raise ValueError("Unexpected temporary documentation package location")
    try:
        (temporary / "moon.pkg").write_text(blocks[0] + '\npkgtype(kind: "executable")\n', encoding="utf-8")
        functions = []
        calls = []
        for index, block in enumerate(blocks[1:]):
            name = "readme_example_{}".format(index)
            functions.append("///|\nfn {}() -> Unit {{\n{}\n}}\n".format(name, block))
            calls.append("  {}()".format(name))
        functions.append("///|\nfn main {{\n{}\n}}\n".format("\n".join(calls)))
        (temporary / "main.mbt").write_text("\n".join(functions), encoding="utf-8")
        subprocess.run(["moon", "check", "--deny-warn", "--target", target], cwd=str(root), check=True)
        subprocess.run(["moon", "run", "--target", target, "examples/" + temporary.name], cwd=str(root), check=True)
        print("README public API snippets compiled and ran ({})".format(target))
    finally:
        # Only remove the unique direct child created above, never a caller path.
        if temporary.parent == examples and temporary.name.startswith("readme-verify-"):
            shutil.rmtree(str(temporary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["wasm-gc", "wasm", "js", "native"], default="wasm")
    args = parser.parse_args()
    verify(Path(__file__).resolve().parent.parent, args.target)

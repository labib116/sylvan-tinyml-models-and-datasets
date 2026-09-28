"""Convert a .tflite file into a C++ array during the ESP-IDF build."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model = args.input.read_bytes()
    lines = []
    for offset in range(0, len(model), 12):
        chunk = model[offset : offset + 12]
        lines.append("    " + ", ".join(f"0x{byte:02x}" for byte in chunk) + ",")

    source = (
        '#include "model_data.h"\n\n'
        "alignas(16) const unsigned char g_model_data[] = {\n"
        + "\n".join(lines)
        + "\n};\n\n"
        + f"const unsigned int g_model_data_len = {len(model)}u;\n"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(source, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

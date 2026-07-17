from __future__ import annotations

import sys
from collections.abc import Callable, Sequence

from policy_integrity_bench import builder, calibrate, materialize, validator

Command = Callable[[Sequence[str] | None], int]

COMMANDS: dict[str, tuple[Command, str]] = {
    "build": (builder.main, "deterministically rebuild the benchmark"),
    "validate": (validator.main, "run schemas, solvers, leakage checks, and runtime replays"),
    "materialize": (materialize.main, "create separated inference inputs and a scoring key"),
    "calibrate": (calibrate.main, "run the optional local Track E model calibration"),
}


def _help() -> str:
    commands = "\n".join(
        f"  {name:<12}{description}" for name, (_, description) in COMMANDS.items()
    )
    return f"""PolicyIntegrityBench command line interface

Usage:
  pib <command> [options]

Commands:
{commands}

Run `pib <command> --help` for command-specific options.
"""


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        print(_help())
        return 0
    command = arguments.pop(0)
    entry = COMMANDS.get(command)
    if entry is None:
        print(f"Unknown command: {command}\n\n{_help()}", file=sys.stderr)
        return 2
    handler, _ = entry
    return handler(arguments)

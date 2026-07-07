from __future__ import annotations

from pathlib import Path

from steering_research.reports.dashboard import write_static_dashboard


def main() -> int:
    dashboard = write_static_dashboard(Path("runs"))
    print(dashboard)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

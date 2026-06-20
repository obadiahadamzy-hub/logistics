from __future__ import annotations

from pathlib import Path

from generate_mock_data import generate_all


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DIRECTORIES = [
    PROJECT_ROOT / "data" / "raw",
    PROJECT_ROOT / "data" / "processed",
    PROJECT_ROOT / "data" / "sample_docs",
    PROJECT_ROOT / "outputs" / "reports",
    PROJECT_ROOT / "outputs" / "charts",
    PROJECT_ROOT / "outputs" / "logs",
]


def init_directories() -> None:
    for directory in DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)


def main() -> None:
    init_directories()
    generate_all()
    print("Project directories and mock data are ready.")


if __name__ == "__main__":
    main()

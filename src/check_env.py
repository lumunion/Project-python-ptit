"""Check the Python environment before running the project."""
from __future__ import annotations

import platform
import sys
from importlib.metadata import PackageNotFoundError, version

PACKAGES = ["numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "joblib", "streamlit", "pytest"]


def main() -> None:
    print("=" * 60)
    print("FOOTBALL MATCH PREDICTION - ENVIRONMENT CHECK")
    print("=" * 60)
    print("Python:", sys.version.split()[0])
    print("Executable:", sys.executable)
    print("Platform:", platform.platform())
    missing = []
    for package in PACKAGES:
        try:
            print(f"[OK] {package}: {version(package)}")
        except PackageNotFoundError:
            print(f"[MISSING] {package}")
            missing.append(package)
    if missing:
        print("\nCài thư viện bằng: python -m pip install -r requirements.txt")
        raise SystemExit(1)
    print("\nPASS: các thư viện cần thiết đã được cài.")


if __name__ == "__main__":
    main()

import sys
import platform
from importlib.metadata import PackageNotFoundError, version

# Các thư viện cần dùng
PACKAGES = [
    "numpy",
    "pandas",
    "matplotlib",
    "scikit-learn",
    "streamlit",
    "joblib",
    "pytest",
]

def main():
    print("=" * 55)
    print("FOOTBALL OUTCOME PREDICTION - ENVIRONMENT CHECK")
    print("=" * 55)

    print("Python version :", sys.version.split()[0])
    print("Python path    :", sys.executable)
    print("Operating system:", platform.platform())

    print("\nInstalled packages:")

    missing = []

    # Kiểm tra từng môi trường đã cài chưa
    for package in PACKAGES:
        try:
            package_version = version(package)
            print(f"[OK] {package}: {package_version}")
        except PackageNotFoundError:
            print(f"[MISSING] {package}")
            missing.append(package)

    if missing:
        print("\nThieu thu vien. Hay chay:")
        print("python -m pip install -r requirements.txt")
        return
    
    print("\nPASS: Moi truong co ban hoat dong.")

if __name__ == "__main__":
    main()
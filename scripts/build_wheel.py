#!/usr/bin/env python3
"""Create a wheel that installs stablehlo-opt directly to bin/."""

import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def find_binary():
    """Auto-detect stablehlo-opt binary from default build location"""
    script_dir = Path(__file__).parent.resolve()
    stablehlo_root = script_dir.parent

    # Default search paths
    search_paths = [
        stablehlo_root / "build/bin/stablehlo-opt",
        stablehlo_root / "build/tools/stablehlo-opt",
    ]

    for path in search_paths:
        if path.exists() and path.is_file():
            return path

    return None


def create_wheel(binary_path: Path, version: str = "1.0.1"):
    """Create a wheel that directly installs the binary to bin/."""

    if not binary_path.exists():
        raise FileNotFoundError(f"Binary not found: {binary_path}")

    # Detect platform
    arch = platform.machine().lower()

    if arch in ["x86_64", "amd64"]:
        platform_tag = "manylinux2014_x86_64"
    elif arch in ["aarch64", "arm64"]:
        platform_tag = "manylinux2014_aarch64"
    else:
        platform_tag = f"linux_{arch}"

    print(f"Creating wheel for platform: {platform_tag}")
    print("Binary will be installed directly to system bin directory")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        package_dir = temp_path / "stablehlo_opt"

        # Create directories
        package_dir.mkdir()

        # Create minimal __init__.py for Python API
        init_content = f'''"""StableHLO optimizer tools package."""
__version__ = "{version}"
__arch__ = "{arch}"

import os
import subprocess
from pathlib import Path

def get_binary_path():
    """Get the path to stablehlo-opt binary."""
    # Try to find it in common locations
    for prefix in [sys.prefix, Path.home() / ".local", "/usr/local"]:
        binary = prefix / "bin" / "stablehlo-opt"
        if binary.exists():
            return str(binary)

    # Use shutil.which to find in PATH
    binary_path = shutil.which("stablehlo-opt")
    if binary_path:
        return binary_path

    raise FileNotFoundError(
        "stablehlo-opt not found in PATH or common locations. "
        "Make sure the package is properly installed to a bin directory."
    )

def run_stablehlo_opt(args=None):
    """Run stablehlo-opt with the given arguments."""
    binary_path = get_binary_path()

    if args is None:
        args = []

    try:
        result = subprocess.run([binary_path] + args, check=True)
        return result.returncode
    except subprocess.CalledProcessError as err:
        print(f"Error running {{binary_path}}: {{err}}", file=sys.stderr)
        return err.returncode
'''
        (package_dir / "__init__.py").write_text(init_content)

        # Create setup.cfg with data_files
        setup_cfg = f'''[metadata]
name = stablehlo-opt
version = {version}
description = StableHLO optimizer tools
author = StableHLO Contributors
license = Apache-2.0
long_description = file: README.md
long_description_content_type = text/markdown
url = https://github.com/openxla/stablehlo
classifiers =
    Programming Language :: Python :: 3
    Programming Language :: Python :: 3.8
    Programming Language :: Python :: 3.9
    Programming Language :: Python :: 3.10
    Programming Language :: Python :: 3.11
    Programming Language :: Python :: 3.12

[options]
python_requires = >=3.8
packages = find:
include_package_data = True
zip_safe = False

[options.packages.find]
where = .

[options.data_files]
bin = stablehlo-opt

[bdist_wheel]
universal = 0
'''
        (temp_path / "setup.cfg").write_text(setup_cfg)

        # Copy binary to temp directory root (as referenced in setup.cfg)
        dest_binary = temp_path / "stablehlo-opt"
        shutil.copy2(binary_path, dest_binary)
        os.chmod(dest_binary, 0o755)

        # Create setup.py minimal
        setup_py = '''from setuptools import setup
setup()
'''
        (temp_path / "setup.py").write_text(setup_py)

        # Create README
        readme_content = """# StableHLO Optimizer Tools

Python package containing the StableHLO optimizer tool.

## Installation

```bash
pip install stablehlo-opt
```

## Usage

After installation, the `stablehlo-opt` binary is directly available:

```bash
stablehlo-opt --version
stablehlo-opt --help
```

The binary is installed to your Python environment's bin directory and works like any native command-line tool.

Or use in Python:

```python
from stablehlo_opt import run_stablehlo_opt
run_stablehlo_opt(["--version"])
```
"""
        (temp_path / "README.md").write_text(readme_content)

        # Build wheel
        orig_dir = os.getcwd()
        os.chdir(temp_path)

        # Build the wheel
        subprocess.run(
            [sys.executable, "setup.py", "bdist_wheel"],
            check=True,
        )

        # Find the wheel
        wheels = list(Path("dist").glob("*.whl")) if Path("dist").exists() else []
        if not wheels:
            wheels = list(Path(".").glob("*.whl"))

        if not wheels:
            raise RuntimeError("No wheel built!")

        # Use the first wheel found
        wheel_path = wheels[0]
        print(f"Built wheel: {wheel_path}")

        # Output to scripts/dist/ directory
        script_dir = Path(__file__).parent.resolve()
        output_dir = script_dir / "dist"
        output_dir.mkdir(parents=True, exist_ok=True)

        dest_wheel = output_dir / f"stablehlo_opt-{version}-py3-none-{platform_tag}.whl"
        shutil.copy2(wheel_path, dest_wheel)

        # Clean up
        os.chdir(orig_dir)

        return dest_wheel


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create wheel for stablehlo-opt")
    parser.add_argument("binary", help="Path to stablehlo-opt binary", nargs="?")
    parser.add_argument("--version", "-v", default="1.0.1", help="Package version")

    args = parser.parse_args()

    # Auto-detect binary if not provided
    binary_path = args.binary
    if not binary_path:
        print("Binary path not provided, auto-detecting...")
        detected = find_binary()
        if detected:
            binary_path = str(detected)
            print(f"Found: {binary_path}")
        else:
            print("Error: Cannot find stablehlo-opt binary", file=sys.stderr)
            print("Please specify binary path explicitly:", file=sys.stderr)
            print("  python3 build_wheel.py /path/to/stablehlo-opt", file=sys.stderr)
            sys.exit(1)

    try:
        wheel_path = create_wheel(Path(binary_path), args.version)
        print(f"\nSuccessfully created: {wheel_path}")
        print(f"\nTo install: pip install {wheel_path}")
        print("\nAfter installation, the stablehlo-opt binary will be")
        print("directly available in your PATH. No wrapper script!")
    except Exception as ex:
        import traceback
        print(f"Error: {ex}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
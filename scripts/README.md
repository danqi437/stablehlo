# StableHLO Build Scripts

This directory contains scripts for building and packaging a minimal `stablehlo-opt` binary.

## Scripts

| Script | Description |
|--------|-------------|
| `build_stablehlo_opt.sh` | Main build script - compiles LLVM/MLIR and StableHLO to produce `stablehlo-opt` |
| `build_wheel.py` | Package the binary as a Python wheel (`.whl`) |

**Test models:**
- `scripts/jax_model.mlir` - Real JAX-compiled model with dynamic_reduce_window, sort, gather
- `scripts/model.mlir` - Simple test model with dynamic_reduce_window + shape_assertion

## Quick Start

### 1. Build stablehlo-opt

```bash
cd /path/to/stablehlo
./scripts/build_stablehlo_opt.sh
```

This will:
- Detect/fetch the correct LLVM version from `build_tools/llvm_version.txt`
- Build LLVM/MLIR with ThinLTO and GC sections for minimal size
- Build `stablehlo-opt` binary
- Output location: `build/bin/stablehlo-opt`

### 2. Test the binary

**Real JAX model (jax_model.mlir):**
```bash
./build/bin/stablehlo-opt scripts/jax_model.mlir --inline \
  --stablehlo-refine-arguments='types=tensor<8xi64>,tensor<8xf64>' \
  --stablehlo-refine-shapes \
  --stablehlo-canonicalize-dynamism \
  --stablehlo-check-shape-assertions
```

**Simple test model (model.mlir):**
```bash
./build/bin/stablehlo-opt scripts/model.mlir --inline \
  --stablehlo-refine-arguments='types=tensor<4x6xf32>' \
  --stablehlo-refine-shapes \
  --stablehlo-canonicalize-dynamism \
  --stablehlo-check-shape-assertions
```

### 3. Package as wheel (optional)

```bash
# Build wheel (auto-detects binary)
./scripts/build_wheel.py

# Or with explicit binary and version
./scripts/build_wheel.py ./build/bin/stablehlo-opt --version 0.1.0
```

Output: `stablehlo_opt-*-py3-none-manylinux2014_*.whl` (platform-specific)

Supported platforms:
- `manylinux2014_x86_64` - x86_64
- `manylinux2014_aarch64` - ARM64

#### Install and Use

```bash
# Install locally
pip install scripts/dist/stablehlo_opt-*-py3-none-manylinux2014_x86_64.whl

# Run after install (binary directly in PATH)
stablehlo-opt --help
stablehlo-opt model.mlir --inline --stablehlo-refine-shapes
```

#### Push to PyPI (optional)

```bash
# Using twine
pip install twine

# Test PyPI (recommended first)
twine upload --repository testpypi scripts/dist/stablehlo_opt-*.whl

# Production PyPI
twine upload scripts/dist/stablehlo_opt-*.whl
```

## Pass Pipeline Reference

### Full Pass Pipeline

```bash
stablehlo-opt input.mlir \
  --inline \
  --stablehlo-refine-arguments='types=type1,type2,...' \
  --stablehlo-refine-shapes \
  --stablehlo-canonicalize-dynamism \
  --stablehlo-check-shape-assertions
```

### Pass Options

| Pass | Option Syntax | Description |
|------|---------------|--------------|
| `--inline` | No options | Inline function calls |
| `--stablehlo-refine-arguments` | `--stablehlo-refine-arguments='types=TYPE1,TYPE2,...'` | Refine function argument types from dynamic to static. Multiple types separated by commas, order matches function parameters. |
| `--stablehlo-refine-shapes` | No options | Propagate static shapes throughout the program |
| `--stablehlo-canonicalize-dynamism` | No options | Canonicalize dynamic shapes and operations |
| `--stablehlo-check-shape-assertions` | No options | Verify and remove shape_assertion custom calls |

### Type Syntax

Use standard MLIR type syntax:
- Static: `tensor<8xf64>`, `tensor<4x6xf32>`, `tensor<i32>`
- Dynamic: `tensor<?xf64>`, `tensor<?x?xf32>`

Examples:
```bash
# Single argument
--stablehlo-refine-arguments='types=tensor<4x6xf32>'

# Multiple arguments
--stablehlo-refine-arguments='types=tensor<8xi64>,tensor<8xf64>'

# Mix static and dynamic
--stablehlo-refine-arguments='types=tensor<?xi64>,tensor<8xf64>'
```

## Test Models

### jax_model.mlir

Derived from a JAX-compiled program. Includes:
- `stablehlo.custom_call @stablehlo.dynamic_reduce_window` - Dynamic window reduction
- `stablehlo.reduce` - Reductions (add, min, max)
- `stablehlo.sort` - Sorting with custom comparator
- `stablehlo.gather` / `stablehlo.concatenate` - Indexing and concatenation
- Function signature: `@main(%arg0: tensor<8xi64>, %arg1: tensor<8xf64>) -> tensor<f64>`

### model.mlir

Simple test model. Includes:
- `stablehlo.custom_call @stablehlo.dynamic_reduce_window` - Max pooling with window [1,3]
- `stablehlo.custom_call @shape_assertion` - Runtime shape validation
- Function signature: `@main(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32>`

## Binary Size

| Mode | Size | Supported passes |
|------|------|-----------------|
| Full (default) | ~129 MB | All MLIR + StableHLO passes |
| `--minimal` | ~17 MB | `--inline`, `--stablehlo-refine-arguments`, `--stablehlo-refine-shapes`, `--stablehlo-canonicalize-dynamism`, `--stablehlo-check-shape-assertions` |

### build_stablehlo_opt.sh

```bash
./scripts/build_stablehlo_opt.sh [--clean] [--minimal] [llvm-project-dir] [build-dir] [stablehlo-build-dir]
```

- `--clean`: Force full rebuild (rm -rf build dirs and reconfigure)
- `--minimal`: Build with `-DSTABLEHLO_OPT_MINIMAL`, producing a much smaller binary (~17MB vs ~129MB). Only registers the passes needed for the shape refinement pipeline. Requires `--clean` when switching modes.
- `llvm-project-dir`: Path to llvm-project (default: auto-detect)
- `build-dir`: LLVM build directory (default: `llvm-build`)
- `stablehlo-build-dir`: StableHLO build directory (default: `build`)

### build_wheel.py

```bash
./scripts/build_wheel.py [binary_path]
```

- `binary_path`: Path to stablehlo-opt binary (default: auto-detect from `build/bin/`)
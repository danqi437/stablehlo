#!/bin/bash
# Build script for stablehlo-opt
#
# Usage:
#   ./build_stablehlo_opt.sh [--clean] [--minimal] [llvm-project-dir] [build-dir]
#
# Options:
#   --clean     Force full rebuild (rm -rf build dirs and reconfigure)
#   --minimal   Build with -DSTABLEHLO_OPT_MINIMAL, producing a smaller binary.
#               Only supports: --inline, --stablehlo-refine-arguments,
#               --stablehlo-refine-shapes, --stablehlo-canonicalize-dynamism,
#               --stablehlo-check-shape-assertions.
#               Requires --clean if switching from full mode (or vice versa).
#
# Incremental mode (default): skips LLVM/MLIR configure+build if already done,
# only recompiles changed stablehlo sources and re-links.
#
# Example:
#   ./build_stablehlo_opt.sh                  # incremental, full binary
#   ./build_stablehlo_opt.sh --clean --minimal # clean + minimal binary

set -e

# Parse flags
CLEAN=false
MINIMAL=false
args=()
for arg in "$@"; do
  case "$arg" in
    --clean)   CLEAN=true ;;
    --minimal) MINIMAL=true ;;
    *)         args+=("$arg") ;;
  esac
done
set -- "${args[@]}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STABLEHLO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default paths
LLVM_PROJECT_DIR="${1:-}"
BUILD_DIR="${2:-$STABLEHLO_ROOT/llvm-build}"
STABLEHLO_BUILD_DIR="${3:-$STABLEHLO_ROOT/build}"

if [[ "$CLEAN" == "true" ]]; then
  echo ">>> Clean build requested, removing build directories <<<"
  rm -rf "$BUILD_DIR" "$STABLEHLO_BUILD_DIR"
fi

# Detect OS
OS="$(uname)"
IS_DARWIN=false
if [[ "$OS" == "Darwin" ]]; then
  IS_DARWIN=true
fi

# Function to find llvm-project
find_llvm_project() {
  local search_paths=(
    "$STABLEHLO_ROOT/llvm-project"
    "$(dirname "$STABLEHLO_ROOT")/llvm-project"
    "$HOME/llvm-project"
    "/root/llvm-project"
  )

  for path in "${search_paths[@]}"; do
    if [[ -f "$path/llvm/CMakeLists.txt" ]]; then
      echo "$path"
      return 0
    fi
  done
  return 1
}

# Check LLVM source directory
if [[ -z "$LLVM_PROJECT_DIR" ]]; then
  echo "LLVM project directory not provided, searching..."
  LLVM_PROJECT_DIR=$(find_llvm_project) || true
fi

if [[ -z "$LLVM_PROJECT_DIR" ]]; then
  echo "Error: Cannot find llvm-project"
  echo ""
  echo "Please clone llvm-project first:"
  echo "  git clone --depth=1 --single-branch https://github.com/llvm/llvm-project.git"
  echo ""
  echo "Then run this script with the path:"
  echo "  $0 /path/to/llvm-project"
  exit 1
fi

LLVM_PROJECT_DIR="$(cd "$LLVM_PROJECT_DIR" && pwd)"

if [[ ! -f "$LLVM_PROJECT_DIR/llvm/CMakeLists.txt" ]]; then
  echo "Error: Invalid LLVM project path: $LLVM_PROJECT_DIR"
  echo "Cannot find llvm/CMakeLists.txt"
  exit 1
fi

# Read required LLVM version from stablehlo version file
LLVM_VERSION_FILE="$STABLEHLO_ROOT/build_tools/llvm_version.txt"
if [[ -f "$LLVM_VERSION_FILE" ]]; then
  REQUIRED_LLVM_COMMIT=$(cat "$LLVM_VERSION_FILE" | tr -d '[:space:]')
  if [[ -n "$REQUIRED_LLVM_COMMIT" && ${#REQUIRED_LLVM_COMMIT} -ge 40 ]]; then
    echo ""
    echo ">>> Checking out required LLVM version <<<"
    echo "Required commit: $REQUIRED_LLVM_COMMIT"
    cd "$LLVM_PROJECT_DIR"
    
    # Check if we need to checkout
    CURRENT_COMMIT=$(git -c safe.directory="$LLVM_PROJECT_DIR" rev-parse HEAD 2>/dev/null || echo "")
    if [[ "$CURRENT_COMMIT" != "$REQUIRED_LLVM_COMMIT" ]]; then
      echo "Current:  $CURRENT_COMMIT"
      echo "Required: $REQUIRED_LLVM_COMMIT"
      echo ""
      echo "Fetching LLVM commit (this may take a while)..."
      git -c safe.directory="$LLVM_PROJECT_DIR" fetch --progress origin "$REQUIRED_LLVM_COMMIT" || {
        echo "Error: Failed to fetch LLVM commit $REQUIRED_LLVM_COMMIT"
        exit 1
      }
      echo ""
      echo "Checking out LLVM commit..."
      git -c safe.directory="$LLVM_PROJECT_DIR" checkout --progress "$REQUIRED_LLVM_COMMIT" || {
        echo "Error: Failed to checkout LLVM commit $REQUIRED_LLVM_COMMIT"
        exit 1
      }
    else
      echo "LLVM already at required commit"
    fi
    cd - > /dev/null
  fi
fi

echo "=============================================="
echo "  Building stablehlo-opt"
echo "=============================================="
echo "LLVM source:     $LLVM_PROJECT_DIR"
echo "LLVM build:      $BUILD_DIR"
echo "StableHLO build: $STABLEHLO_BUILD_DIR"
echo "OS:              $OS"
if [[ "$MINIMAL" == "true" ]]; then
  echo "Mode:            minimal (shape refinement pipeline only)"
  LLVM_TARGETS=""
else
  echo "Mode:            full"
  LLVM_TARGETS="host"
fi
echo ""

# ============================================
# Step 1: Configure and build LLVM/MLIR
# ============================================

# Build configuration
LINKER_FLAGS=""
LTO_TYPE="NO"
LLVM_ENABLE_LLD="OFF"
CMAKE_CXX_FLAGS="-ffunction-sections -fdata-sections -O3"

if [[ "$MINIMAL" == "true" ]]; then
  CMAKE_CXX_FLAGS="$CMAKE_CXX_FLAGS -DSTABLEHLO_OPT_MINIMAL"
fi

if [[ "$IS_DARWIN" == "false" ]]; then
  LINKER_FLAGS="-s -Wl,--gc-sections"
  LTO_TYPE="Thin"
  LLVM_ENABLE_LLD="ON"
  CMAKE_CXX_FLAGS="$CMAKE_CXX_FLAGS -fno-stack-protector"
fi

# Skip LLVM build if already configured
if [[ -f "$BUILD_DIR/build.ninja" ]]; then
  echo ">>> Step 1: LLVM/MLIR already configured, skipping <<<"
  echo "    (use --clean to force full rebuild)"
else
  echo ">>> Step 1: Configuring LLVM/MLIR <<<"
  echo "----------------------------------------------"

  mkdir -p "$BUILD_DIR"
  cd "$BUILD_DIR"

  cmake "$LLVM_PROJECT_DIR/llvm" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_TARGETS_TO_BUILD="$LLVM_TARGETS" \
    \
    -DMLIR_ENABLE_BINDINGS_PYTHON=OFF \
    \
    -DLLVM_INCLUDE_BENCHMARKS=OFF \
    -DLLVM_INCLUDE_TESTS=OFF \
    -DLLVM_INCLUDE_EXAMPLES=OFF \
    -DLLVM_INCLUDE_DOCS=OFF \
    -DLLVM_BUILD_TOOLS=OFF \
    -DLLVM_INSTALL_UTILS=ON \
    \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DLLVM_ENABLE_LLD="$LLVM_ENABLE_LLD" \
    -DLLVM_ENABLE_LTO="$LTO_TYPE" \
    -DCMAKE_CXX_FLAGS="$CMAKE_CXX_FLAGS" \
    -DCMAKE_EXE_LINKER_FLAGS="$LINKER_FLAGS"

  echo ""
  echo ">>> Step 2: Building LLVM/MLIR <<<"
  echo "----------------------------------------------"

  cmake --build "$BUILD_DIR" --target mlir-libraries mlir-pdll -- -j$(nproc)
fi

# ============================================
# Step 3: Configure and build StableHLO
# ============================================

# Skip StableHLO configure if already done
if [[ -f "$STABLEHLO_BUILD_DIR/build.ninja" ]]; then
  echo ">>> Step 3: StableHLO already configured, skipping configure <<<"
else
  echo ""
  echo ">>> Step 3: Configuring StableHLO <<<"
  echo "----------------------------------------------"

  mkdir -p "$STABLEHLO_BUILD_DIR"
  cd "$STABLEHLO_BUILD_DIR"

  cmake "$STABLEHLO_ROOT" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DSTABLEHLO_ENABLE_BINDINGS_PYTHON=OFF \
    -DMLIR_DIR="$BUILD_DIR/lib/cmake/mlir" \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DLLVM_ENABLE_LLD="$LLVM_ENABLE_LLD" \
    -DLLVM_ENABLE_LTO="$LTO_TYPE" \
    -DCMAKE_CXX_FLAGS="$CMAKE_CXX_FLAGS" \
    -DCMAKE_EXE_LINKER_FLAGS="$LINKER_FLAGS"
fi

echo ""
echo ">>> Step 4: Building stablehlo-opt <<<"
echo "----------------------------------------------"

cd "$STABLEHLO_BUILD_DIR"
cmake --build . --target stablehlo-opt -- -j$(nproc)

# ============================================
# Step 5: Verify and report
# ============================================
echo ""
echo "=============================================="
echo "  Build Complete!"
echo "=============================================="

STABLEHLO_OPT="$STABLEHLO_BUILD_DIR/bin/stablehlo-opt"

if [[ -f "$STABLEHLO_OPT" ]]; then
  echo ""
  echo "Binary: $(realpath $STABLEHLO_OPT)"
  echo "Size:   $(du -h "$STABLEHLO_OPT" | cut -f1)"
  echo ""

  if [[ "$IS_DARWIN" == "false" ]]; then
    echo "Dependencies:"
    if ldd "$STABLEHLO_OPT" 2>/dev/null | grep -q "not found"; then
      echo "  (some dependencies not found - may be OK for static binary)"
    else
      ldd "$STABLEHLO_OPT" 2>/dev/null || echo "  (statically linked)"
    fi
  fi

  echo ""
  echo "Usage example:"
  echo "  $STABLEHLO_OPT $STABLEHLO_ROOT/scripts/jax_model.mlir --inline \\"
  echo "    --stablehlo-refine-arguments='types=tensor<8xi64>,tensor<8xf64>' \\"
  echo "    --stablehlo-refine-shapes --stablehlo-canonicalize-dynamism \\"
  echo "    --stablehlo-check-shape-assertions"
else
  echo "Error: stablehlo-opt not found at $STABLEHLO_OPT"
  exit 1
fi
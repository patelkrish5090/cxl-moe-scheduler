#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY="${ASTERA_THIRD_PARTY:-$REPO_ROOT/third_party}"
GEM5_DIR="$THIRD_PARTY/gem5"
DRAMSIM3_DIR="$GEM5_DIR/ext/dramsim3/DRAMsim3"
GEM5_TAG="${ASTERA_GEM5_TAG:-v24.0.0.0}"
GEM5_ARCH="${ASTERA_GEM5_ARCH:-X86}"
GEM5_BINARY="$GEM5_DIR/build/$GEM5_ARCH/gem5.opt"
JOBS="${ASTERA_BUILD_JOBS:-$(nproc 2>/dev/null || echo 8)}"

info()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
warn()  { printf '\033[33mWARNING: %s\033[0m\n' "$*"; }
die()   { printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

probe_zlib() {
  local extra_flags="$1"
  local tmp
  tmp="$(mktemp -d)"
  cat > "$tmp/z.cc" <<'ZTEST'
#include <zlib.h>
int main() { zlibVersion(); return 0; }
ZTEST
  if g++ "$tmp/z.cc" -o "$tmp/z" $extra_flags -lz > "$tmp/err" 2>&1; then
    rm -rf "$tmp"
    return 0
  fi
  ZLIB_PROBE_ERROR="$(cat "$tmp/err")"
  rm -rf "$tmp"
  return 1
}

resolve_python_config() {
  if [ -n "${ASTERA_PYTHON_CONFIG:-}" ]; then
    PYTHON_CONFIG_CHOICE="$ASTERA_PYTHON_CONFIG"
    PYTHON_CONFIG_REASON="ASTERA_PYTHON_CONFIG was set"
    return
  fi

  local active_cfg="" system_cfg=""
  active_cfg="$(command -v python3-config || true)"
  for candidate in /usr/bin/python3-config /usr/bin/python3.12-config \
                   /usr/bin/python3.11-config /usr/bin/python3.10-config; do
    [ -x "$candidate" ] && { system_cfg="$candidate"; break; }
  done

  local active_ldflags=""
  [ -n "$active_cfg" ] && active_ldflags="$("$active_cfg" --ldflags 2>/dev/null || true)"

  if [ -n "$active_cfg" ] && probe_zlib "$active_ldflags"; then
    PYTHON_CONFIG_CHOICE="$active_cfg"
    PYTHON_CONFIG_REASON="its link environment finds zlib"
    return
  fi

  if [ -n "$system_cfg" ] && [ "$system_cfg" != "$active_cfg" ]; then
    local system_ldflags
    system_ldflags="$("$system_cfg" --ldflags 2>/dev/null || true)"
    if probe_zlib "$system_ldflags"; then
      PYTHON_CONFIG_CHOICE="$system_cfg"
      PYTHON_CONFIG_REASON="the active (conda) Python's lib/ shadows the system zlib; this one links"
      return
    fi
  fi

  PYTHON_CONFIG_CHOICE=""
  PYTHON_CONFIG_REASON="no python3-config produced a working zlib link"
}

check_zlib() {
  printf '\n'
  if probe_zlib ""; then
    printf '  %-10s links with a bare g++ -lz -- ok\n' "zlib"
  else
    printf '  %-10s DOES NOT LINK even with no extra flags:\n' "zlib"
    printf '%s\n' "$ZLIB_PROBE_ERROR" | sed 's/^/      /'
    printf '      -> install the development package: sudo apt install zlib1g-dev\n'
    return 1
  fi

  resolve_python_config
  if [ -z "$PYTHON_CONFIG_CHOICE" ]; then
    printf '  %-10s no usable python3-config (%s)\n' "python" "$PYTHON_CONFIG_REASON"
    printf '      Last error:\n'
    printf '%s\n' "${ZLIB_PROBE_ERROR:-none}" | sed 's/^/      /'
    printf '      If the system Python is the problem: sudo apt install python3-dev\n'
    return 1
  fi
  printf '  %-10s %s\n' "py-config" "$PYTHON_CONFIG_CHOICE"
  printf '  %-10s %s\n' "" "($PYTHON_CONFIG_REASON)"

  if [ -n "${CONDA_PREFIX:-}" ] && [[ "$PYTHON_CONFIG_CHOICE" != "$CONDA_PREFIX"* ]]; then
    printf '\n'
    warn "conda env '${CONDA_DEFAULT_ENV:-?}' is active but gem5 will be built against"
    warn "the SYSTEM Python instead. That is fine and intended: gem5 embeds its own"
    warn "interpreter and memsim runs gem5.opt as a subprocess, so it never needs to"
    warn "share the conda environment."
  fi
  return 0
}

check_prereqs() {
  info "Checking prerequisites"
  local missing=()

  for tool in git cmake g++ python3; do
    if command -v "$tool" >/dev/null 2>&1; then
      printf '  %-10s %s\n' "$tool" "$(command -v "$tool")"
    else
      printf '  %-10s MISSING\n' "$tool"
      missing+=("$tool")
    fi
  done

  if command -v scons >/dev/null 2>&1; then
    printf '  %-10s %s (%s)\n' "scons" "$(command -v scons)" "$(scons --version 2>/dev/null | grep -o 'v[0-9.]*' | head -1)"
  else
    printf '  %-10s MISSING  -> pip install scons\n' "scons"
    missing+=("scons")
  fi

  if command -v g++ >/dev/null 2>&1; then
    local gcc_major
    gcc_major="$(g++ -dumpversion | cut -d. -f1)"
    printf '  %-10s version %s' "g++" "$(g++ -dumpversion)"
    if [ "$gcc_major" -lt 10 ]; then
      printf '  -- TOO OLD (gem5 needs >= 10)\n'
      missing+=("g++>=10")
    else
      printf '  -- ok\n'
    fi
  fi

  if python3 -c "import sysconfig,os,sys; sys.exit(0 if os.path.exists(os.path.join(sysconfig.get_paths()['include'],'Python.h')) else 1)" 2>/dev/null; then
    printf '  %-10s %s -- ok\n' "Python.h" "$(python3 -c "import sysconfig;print(sysconfig.get_paths()['include'])")"
  else
    printf '  %-10s MISSING  -> conda install python-devel  (or apt install python3-dev)\n' "Python.h"
    missing+=("python3-dev")
  fi

  printf '\n  build arch   %s\n' "$GEM5_ARCH"
  printf '  gem5 tag     %s\n' "$GEM5_TAG"
  printf '  parallelism  -j%s\n' "$JOBS"
  printf '  install dir  %s\n' "$THIRD_PARTY"

  local avail
  avail="$(df -BG --output=avail "$(dirname "$THIRD_PARTY")" 2>/dev/null | tail -1 | tr -dc '0-9' || echo '?')"
  printf '  free space   %s GB (need ~15 GB)\n' "$avail"
  if [ "$avail" != "?" ] && [ "$avail" -lt 15 ]; then
    warn "less than 15 GB free; the gem5 build will likely fail partway"
  fi

  if [ ${#missing[@]} -gt 0 ]; then
    die "missing prerequisites: ${missing[*]}"
  fi

  check_zlib || die "zlib/python configuration is not usable; see above"

  printf '\n  All prerequisites present.\n'
}

fetch_gem5() {
  mkdir -p "$THIRD_PARTY"
  if [ -d "$GEM5_DIR/.git" ]; then
    info "gem5 source already present at $GEM5_DIR"
    git -C "$GEM5_DIR" describe --tags --always 2>/dev/null | sed 's/^/  at /'
    return
  fi

  info "Cloning gem5 ($GEM5_TAG) -- this is the big download"
  if ! git clone --depth 1 --branch "$GEM5_TAG" https://github.com/gem5/gem5.git "$GEM5_DIR" 2>/dev/null; then
    warn "tag '$GEM5_TAG' not found; available recent tags:"
    git ls-remote --tags --refs https://github.com/gem5/gem5.git \
      | awk -F/ '{print "    " $NF}' | sort -V | tail -12
    die "set ASTERA_GEM5_TAG to one of the above and re-run"
  fi
  git -C "$GEM5_DIR" describe --tags --always | sed 's/^/  checked out /'
}

fetch_dramsim3() {
  if [ -d "$DRAMSIM3_DIR/.git" ]; then
    info "DRAMSim3 source already present"
    return
  fi
  info "Cloning DRAMSim3 into gem5's ext/ tree"
  mkdir -p "$(dirname "$DRAMSIM3_DIR")"
  git clone --depth 1 https://github.com/umd-memsys/DRAMsim3.git "$DRAMSIM3_DIR"
}

build_dramsim3() {
  info "Building DRAMSim3 as a shared library"
  mkdir -p "$DRAMSIM3_DIR/build"
  (
    cd "$DRAMSIM3_DIR/build"
    cmake .. -DCMAKE_BUILD_TYPE=Release -DTHERMAL=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    make -j"$JOBS"
  )

  local lib=""
  for candidate in \
      "$DRAMSIM3_DIR/libdramsim3.so" \
      "$DRAMSIM3_DIR/build/libdramsim3.so"; do
    [ -f "$candidate" ] && { lib="$candidate"; break; }
  done
  if [ -z "$lib" ]; then
    warn "no libdramsim3.so found. What the build did produce:"
    find "$DRAMSIM3_DIR" -name 'libdramsim3*' -printf '    %p\n' 2>/dev/null || true
    die "DRAMSim3 did not produce a shared library; gem5 would build without it"
  fi
  if [ "$lib" != "$DRAMSIM3_DIR/libdramsim3.so" ]; then
    cp "$lib" "$DRAMSIM3_DIR/libdramsim3.so"
    printf '  copied %s -> %s\n' "$lib" "$DRAMSIM3_DIR/libdramsim3.so"
  fi
  printf '  ok: %s\n' "$DRAMSIM3_DIR/libdramsim3.so"
}

build_gem5() {
  resolve_python_config
  [ -n "$PYTHON_CONFIG_CHOICE" ] || die "no usable python3-config ($PYTHON_CONFIG_REASON)"

  info "Building gem5 ($GEM5_ARCH, -j$JOBS) -- 30 to 60 minutes"
  printf '  python-config %s\n' "$PYTHON_CONFIG_CHOICE"
  printf '  reason        %s\n' "$PYTHON_CONFIG_REASON"
  printf '  Log: %s\n' "$THIRD_PARTY/gem5_build.log"

  (
    cd "$GEM5_DIR"
    export PATH="$(dirname "$PYTHON_CONFIG_CHOICE"):$PATH"
    scons "build/$GEM5_ARCH/gem5.opt" -j"$JOBS" --ignore-style \
      "PYTHON_CONFIG=$PYTHON_CONFIG_CHOICE" 2>&1 \
      | tee "$THIRD_PARTY/gem5_build.log"
  )
  [ -x "$GEM5_BINARY" ] || die "build finished but $GEM5_BINARY is missing"
  printf '  built %s\n' "$GEM5_BINARY"
}

verify_build() {
  info "Verifying the build actually has DRAMSim3"
  [ -x "$GEM5_BINARY" ] || die "no gem5 binary at $GEM5_BINARY -- run the build first"

  printf '  binary   %s\n' "$GEM5_BINARY"
  printf '  size     %s\n' "$(du -h "$GEM5_BINARY" | cut -f1)"

  local probe="$THIRD_PARTY/_probe_dramsim3.py"
  cat > "$probe" <<'PROBE'
import sys
import m5
from m5.objects import DRAMsim3  # noqa: F401  -- import IS the test
print("DRAMSIM3_AVAILABLE")
sys.exit(0)
PROBE

  if "$GEM5_BINARY" --outdir="$THIRD_PARTY/_probe_out" "$probe" 2>&1 | grep -q DRAMSIM3_AVAILABLE; then
    printf '\n  \033[32mgem5 has DRAMSim3 support. Stage 2 can run.\033[0m\n'
  else
    warn "gem5 built, but the DRAMsim3 SimObject is not available."
    warn "That means the DRAMSim3 library was not linked in. Re-run:"
    warn "  bash scripts/build_gem5.sh dramsim3   # rebuild just the library"
    warn "  bash scripts/build_gem5.sh gem5       # then relink gem5"
    rm -rf "$THIRD_PARTY/_probe_out" "$probe"
    exit 1
  fi
  rm -rf "$THIRD_PARTY/_probe_out" "$probe"

  printf '\n  Record this in any results you publish:\n'
  printf '    gem5     %s\n' "$(git -C "$GEM5_DIR" describe --tags --always 2>/dev/null || echo unknown)"
  printf '    DRAMSim3 %s\n' "$(git -C "$DRAMSIM3_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
}

case "${1:-all}" in
  check)     check_prereqs ;;
  zlib)
    check_zlib
    ;;

  diagnose)
    info "python3-config in use"
    pycfg="$(command -v python3-config || echo none)"
    printf '  path      %s\n' "$pycfg"
    if [ "$pycfg" != "none" ]; then
      printf '  --includes %s\n' "$("$pycfg" --includes 2>&1)"
      printf '  --ldflags  %s\n' "$("$pycfg" --ldflags 2>&1)"
      printf '  --libs     %s\n' "$("$pycfg" --libs 2>&1)"
      printf '  --ldflags --embed  %s\n' "$("$pycfg" --ldflags --embed 2>&1 || echo '(unsupported)')"
    fi

    info "libz visible to the linker"
    for dir in "${CONDA_PREFIX:-/nonexistent}/lib" /usr/lib/x86_64-linux-gnu /usr/lib64; do
      [ -d "$dir" ] || continue
      printf '  %s:\n' "$dir"
      ls -la "$dir"/libz.so* 2>/dev/null | sed 's/^/    /' || printf '    (no libz here)\n'
    done
    info "zlib.h visible to the compiler"
    for dir in "${CONDA_PREFIX:-/nonexistent}/include" /usr/include; do
      [ -f "$dir/zlib.h" ] && printf '  %s/zlib.h  (version %s)\n' "$dir" \
        "$(grep -m1 ZLIB_VERSION "$dir/zlib.h" | tr -d '\r')"
    done

    info "Reproducing gem5's own check, with the Python flags it adds"
    tmp="$(mktemp -d)"
    cat > "$tmp/z.cc" <<'ZTEST'
#include <zlib.h>
int main() { zlibVersion(); return 0; }
ZTEST
    if [ "$pycfg" != "none" ]; then
      inc="$("$pycfg" --includes 2>/dev/null)"
      ldf="$("$pycfg" --ldflags 2>/dev/null)"
      printf '  g++ %s %s -lz\n' "$inc" "$ldf"
      if g++ "$tmp/z.cc" -o "$tmp/z" $inc $ldf -lz 2>"$tmp/err"; then
        printf '    LINKS OK -- so the failure is not this combination alone\n'
      else
        printf '    FAILED:\n'
        sed 's/^/      /' "$tmp/err"
      fi
    fi
    rm -rf "$tmp"

    info "gem5's configure log (the authoritative error)"
    found=0
    while IFS= read -r log; do
      found=1
      printf '\n  --- %s (last 60 lines) ---\n' "$log"
      tail -60 "$log" | sed 's/^/  /'
    done < <(find "$GEM5_DIR/build" -maxdepth 3 \
               \( -name 'config.log' -o -name '*config*.log' \) 2>/dev/null)
    if [ "$found" -eq 0 ]; then
      warn "no scons config log found under $GEM5_DIR/build"
      warn "Force a fresh configure so one is written:"
      warn "  rm -rf $GEM5_DIR/build/$GEM5_ARCH && bash scripts/build_gem5.sh gem5"
    fi
    ;;
  fetch)     check_prereqs; fetch_gem5; fetch_dramsim3 ;;
  dramsim3)  build_dramsim3 ;;
  gem5)      build_gem5; verify_build ;;
  verify)    verify_build ;;
  all)
    check_prereqs
    fetch_gem5
    fetch_dramsim3
    build_dramsim3
    build_gem5
    verify_build
    info "Done. Next: python -m memsim.cli characterize --help"
    ;;
  *)
    sed -n '2,10p' "$0"
    exit 1
    ;;
esac

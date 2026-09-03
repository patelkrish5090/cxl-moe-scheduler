#!/usr/bin/env bash
# Build gem5 with DRAMSim3 linked in -- stage 2's memory model (docs.md 4.3).
#
#   bash scripts/build_gem5.sh                 # full build, ~30-60 min
#   bash scripts/build_gem5.sh check           # prerequisites only, no download
#   bash scripts/build_gem5.sh zlib            # diagnose the zlib/conda clash
#   bash scripts/build_gem5.sh verify          # is an existing build usable?
#
# Override the interpreter gem5 embeds with ASTERA_PYTHON_CONFIG=/path/to/
# python3-config if the automatic choice is wrong.
#
# Downloads ~10 GB into third_party/ (gitignored). Nothing here touches the
# model weights or the profiler.
#
# WHY A SCRIPT AND NOT A README
# -----------------------------
# gem5 fails in ways that look like success: it will happily build WITHOUT
# DRAMSim3 and then die at runtime with "object 'DRAMsim3' not found", after
# you have already waited an hour. Every step below is checked, and the final
# `verify` step actually instantiates the DRAMsim3 SimObject rather than
# trusting that the link step worked.

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

# ------------------------------------------------------------------ zlib probe
# gem5's scons checks zlib by compiling and linking a test program, and it does
# so with the flags `python3-config --ldflags` hands it. When a conda env is
# active that puts the env's lib/ ahead of /usr/lib on the link line, so a
# perfectly good system zlib1g-dev is never reached and the check fails with
# "Checking for C++ library z... no". Reinstalling the system package cannot fix
# that, which is why this probe reports the real compiler error instead.
#
# Returns 0 if the test links with the given extra flags.
probe_zlib() {
  local extra_flags="$1"
  local tmp
  tmp="$(mktemp -d)"
  cat > "$tmp/z.cc" <<'ZTEST'
#include <zlib.h>
int main() { zlibVersion(); return 0; }
ZTEST
  # shellcheck disable=SC2086 -- extra_flags is deliberately word-split
  if g++ "$tmp/z.cc" -o "$tmp/z" $extra_flags -lz > "$tmp/err" 2>&1; then
    rm -rf "$tmp"
    return 0
  fi
  ZLIB_PROBE_ERROR="$(cat "$tmp/err")"
  rm -rf "$tmp"
  return 1
}

# Which python3-config should gem5 build against? gem5 embeds a Python
# interpreter, and memsim invokes gem5.opt as a subprocess, so it does NOT need
# to be the same interpreter as the conda env the profiler runs in. When conda's
# Python breaks the zlib link, the system Python is the right answer.
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

  # Does the currently-selected Python's link environment find zlib?
  local active_ldflags=""
  [ -n "$active_cfg" ] && active_ldflags="$("$active_cfg" --ldflags 2>/dev/null || true)"

  if [ -n "$active_cfg" ] && probe_zlib "$active_ldflags"; then
    PYTHON_CONFIG_CHOICE="$active_cfg"
    PYTHON_CONFIG_REASON="its link environment finds zlib"
    return
  fi

  # It does not. Fall back to the system Python if that one works.
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

# ---------------------------------------------------------------- prerequisites
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

  # scons is the gem5 build system. It is a pip package, so it may live in the
  # conda env rather than on the system.
  if command -v scons >/dev/null 2>&1; then
    printf '  %-10s %s (%s)\n' "scons" "$(command -v scons)" "$(scons --version 2>/dev/null | grep -o 'v[0-9.]*' | head -1)"
  else
    printf '  %-10s MISSING  -> pip install scons\n' "scons"
    missing+=("scons")
  fi

  # gem5 v22+ needs a C++17 compiler; GCC 10 is the documented floor.
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

  # gem5 links against libpython, so the dev headers must be present.
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

  # zlib is checked last and separately because it is the one that fails two
  # minutes INTO the build rather than at the start, and because the fix is
  # never "install zlib" when a conda env is in the way.
  check_zlib || die "zlib/python configuration is not usable; see above"

  printf '\n  All prerequisites present.\n'
}

# ------------------------------------------------------------------- fetch gem5
fetch_gem5() {
  mkdir -p "$THIRD_PARTY"
  if [ -d "$GEM5_DIR/.git" ]; then
    info "gem5 source already present at $GEM5_DIR"
    git -C "$GEM5_DIR" describe --tags --always 2>/dev/null | sed 's/^/  at /'
    return
  fi

  info "Cloning gem5 ($GEM5_TAG) -- this is the big download"
  # Shallow clone of one tag: the full history is ~2 GB of no use to us.
  if ! git clone --depth 1 --branch "$GEM5_TAG" https://github.com/gem5/gem5.git "$GEM5_DIR" 2>/dev/null; then
    warn "tag '$GEM5_TAG' not found; available recent tags:"
    git ls-remote --tags --refs https://github.com/gem5/gem5.git \
      | awk -F/ '{print "    " $NF}' | sort -V | tail -12
    die "set ASTERA_GEM5_TAG to one of the above and re-run"
  fi
  git -C "$GEM5_DIR" describe --tags --always | sed 's/^/  checked out /'
}

# --------------------------------------------------------------- fetch DRAMSim3
fetch_dramsim3() {
  if [ -d "$DRAMSIM3_DIR/.git" ]; then
    info "DRAMSim3 source already present"
    return
  fi
  info "Cloning DRAMSim3 into gem5's ext/ tree"
  # gem5 looks for this at exactly ext/dramsim3/DRAMsim3 -- the path and the
  # capitalisation are both load-bearing.
  mkdir -p "$(dirname "$DRAMSIM3_DIR")"
  git clone --depth 1 https://github.com/umd-memsys/DRAMsim3.git "$DRAMSIM3_DIR"
}

# --------------------------------------------------------------- build DRAMSim3
build_dramsim3() {
  info "Building DRAMSim3 as a shared library"
  # gem5 links against libdramsim3.so, so a static-only build silently produces
  # a gem5 without DRAMSim3 support. THERMAL=OFF drops a dependency we do not
  # use and that pulls in extra libraries.
  mkdir -p "$DRAMSIM3_DIR/build"
  (
    cd "$DRAMSIM3_DIR/build"
    cmake .. -DCMAKE_BUILD_TYPE=Release -DTHERMAL=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    make -j"$JOBS"
  )

  # gem5's SConscript expects the library at the DRAMsim3 root, not in build/.
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

# ------------------------------------------------------------------- build gem5
build_gem5() {
  # Pick the interpreter before doing anything expensive: gem5 caches its
  # configure results, so switching PYTHON_CONFIG later means a full rebuild.
  resolve_python_config
  [ -n "$PYTHON_CONFIG_CHOICE" ] || die "no usable python3-config ($PYTHON_CONFIG_REASON)"

  info "Building gem5 ($GEM5_ARCH, -j$JOBS) -- 30 to 60 minutes"
  printf '  python-config %s\n' "$PYTHON_CONFIG_CHOICE"
  printf '  reason        %s\n' "$PYTHON_CONFIG_REASON"
  printf '  Log: %s\n' "$THIRD_PARTY/gem5_build.log"

  (
    cd "$GEM5_DIR"
    # Two levers, because gem5 uses both. The build log line
    #   "Info: Using Python config: python3-config"
    # shows it resolves the tool from PATH, so putting the chosen one first is
    # what actually takes effect; PYTHON_CONFIG is set as well for the gem5
    # versions that read it as a scons variable. Whichever mechanism this gem5
    # honours, both now point at the same interpreter.
    export PATH="$(dirname "$PYTHON_CONFIG_CHOICE"):$PATH"
    # --ignore-style skips the pre-commit style hooks, which otherwise prompt.
    scons "build/$GEM5_ARCH/gem5.opt" -j"$JOBS" --ignore-style \
      "PYTHON_CONFIG=$PYTHON_CONFIG_CHOICE" 2>&1 \
      | tee "$THIRD_PARTY/gem5_build.log"
  )
  [ -x "$GEM5_BINARY" ] || die "build finished but $GEM5_BINARY is missing"
  printf '  built %s\n' "$GEM5_BINARY"
}

# ---------------------------------------------------------------------- verify
verify_build() {
  info "Verifying the build actually has DRAMSim3"
  [ -x "$GEM5_BINARY" ] || die "no gem5 binary at $GEM5_BINARY -- run the build first"

  printf '  binary   %s\n' "$GEM5_BINARY"
  printf '  size     %s\n' "$(du -h "$GEM5_BINARY" | cut -f1)"

  # The real test: instantiate the SimObject. A gem5 built without DRAMSim3
  # still starts fine and only fails here, which is the trap this catches.
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

# ------------------------------------------------------------------------ main
case "${1:-all}" in
  check)     check_prereqs ;;
  zlib)
    # Just the zlib/python diagnosis, for when the build fails at
    # "Checking for C++ library z... no".
    check_zlib
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

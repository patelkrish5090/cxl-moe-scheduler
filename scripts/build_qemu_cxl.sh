#!/usr/bin/env bash
# Bring up QEMU with a virtual CXL Type 3 memory device (docs.md 4.4) --
# an OPTIONAL strengthening layer that validates a real OS-level CXL memory
# region behaves as memsim's tier model assumes (DAX device allocation,
# access patterns). It is NOT the source of any energy or latency number
# used anywhere in this project -- that is gem5 + DRAMSim3 (memsim/), fully
# validated already. See memsim/README.md's "QEMU CXL VM (optional)" section
# before running this.
#
#   bash scripts/build_qemu_cxl.sh check              # prerequisites only
#   bash scripts/build_qemu_cxl.sh launch <disk.qcow2> [<cxl-size-gib>]
#   bash scripts/build_qemu_cxl.sh launch-cmd <disk.qcow2> [<cxl-size-gib>]
#
# `check` verifies the QEMU binary is new enough and reports what it finds --
# it does not install or build anything. `launch` runs QEMU directly against
# an EXISTING Linux guest disk image you provide (this script does not create
# one -- see memsim/README.md for how to get a minimal cloud image).
# `launch-cmd` prints the exact command without running it, for review or to
# adapt by hand.
#
# WHY THIS IS SEPARATE FROM build_gem5.sh
# ----------------------------------------
# gem5 is a simulator this project's own code invokes as a library/subprocess
# (memsim/run_sweep.py). QEMU here is a full virtual MACHINE you boot,
# interact with over its own console/SSH, and inspect from the INSIDE (the
# guest kernel's own /sys/bus/cxl, daxctl, cxl-cli) -- this script only gets
# you to a running VM with the device attached; verifying the guest actually
# sees a working CXL region is a manual step documented in memsim/README.md,
# not something this host-side script can do for you.
#
# REAL PREREQUISITES, NOT A FORMALITY
# -------------------------------------
# QEMU's CXL Type 3 device model needs a QEMU version new enough to have it
# at all, working correctly, and stably. QEMU first added basic CXL support
# in 7.0; volatile HDM decoder and Fixed Memory Window handling had real bugs
# fixed through 7.1/7.2/8.0. This script requires >= 8.0 and warns (does not
# block) below that, because "it builds and boots" and "the guest kernel
# actually enumerates a working CXL region" are two different claims, and an
# older QEMU is far more likely to satisfy only the first.
#
# The GUEST kernel needs CXL support compiled in: CONFIG_CXL_BUS,
# CONFIG_CXL_ACPI, CONFIG_CXL_PCI, CONFIG_CXL_MEM (mainline since 5.12,
# default-enabled in most distro kernels 5.19+ -- Ubuntu 22.04's HWE kernel
# and Ubuntu 23.04+'s default kernel both qualify). This script cannot check
# the GUEST kernel from the host; verify inside the VM per memsim/README.md.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MIN_QEMU_MAJOR=8
CXL_DEFAULT_SIZE_GIB=4

info()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
warn()  { printf '\033[33mWARNING: %s\033[0m\n' "$*"; }
die()   { printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

find_qemu() {
  command -v qemu-system-x86_64 2>/dev/null || true
}

qemu_version() {
  local bin="$1"
  "$bin" --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -1
}

check_kvm() {
  if [ -e /dev/kvm ]; then
    if [ -r /dev/kvm ] && [ -w /dev/kvm ]; then
      echo "  /dev/kvm present and accessible -- hardware acceleration available"
      return 0
    fi
    warn "/dev/kvm exists but is not read/write for this user. Either run as a user in the" \
         "'kvm' group, or QEMU will fall back to slow software emulation (-accel tcg)."
    return 1
  fi
  warn "/dev/kvm not found. QEMU will run under software emulation (tcg), which is MUCH" \
       "slower -- fine for a boot-and-check, painful for anything longer. If this is a" \
       "cloud VM, check whether nested virtualization is enabled."
  return 1
}

cmd_check() {
  info "QEMU binary"
  local qemu_bin
  qemu_bin="$(find_qemu)"
  if [ -z "$qemu_bin" ]; then
    die "no qemu-system-x86_64 on PATH. Install it (e.g. 'apt install qemu-system-x86' on" \
        "Debian/Ubuntu) or build from source (https://www.qemu.org/download/#source) if your" \
        "distro's package is older than $MIN_QEMU_MAJOR.0 -- check the version it ships first:" \
        "'apt-cache policy qemu-system-x86' before building from source, which takes a while."
  fi
  echo "  found: $qemu_bin"

  local version major
  version="$(qemu_version "$qemu_bin")"
  if [ -z "$version" ]; then
    warn "could not parse a version from '$qemu_bin --version' -- inspect it manually."
  else
    major="${version%%.*}"
    echo "  version: $version"
    if [ "$major" -lt "$MIN_QEMU_MAJOR" ] 2>/dev/null; then
      warn "QEMU $version is older than the recommended $MIN_QEMU_MAJOR.0. CXL Type 3 support may" \
           "be missing or buggy (basic support landed in 7.0; real fixes through 8.0). This is a" \
           "warning, not a hard stop -- 'launch' will still try -- but if the guest never sees a" \
           "CXL device, upgrading QEMU is the first thing to try, not further QEMU flag tweaking."
    fi
  fi

  info "CXL device support compiled into this QEMU build"
  if "$qemu_bin" -device help 2>/dev/null | grep -qi "cxl-type3"; then
    echo "  cxl-type3 device is available in this build"
  else
    die "'$qemu_bin -device help' does not list cxl-type3. This QEMU binary was not built" \
        "with CXL support (some distro packages disable it) -- rebuild QEMU from source with" \
        "the default configure options (CXL support is on by default when the host is x86_64" \
        "and the QEMU version supports it; no special --enable flag is needed in >= 8.0)."
  fi

  info "KVM (hardware acceleration)"
  check_kvm || true

  info "Existing guest disk images"
  local found_any=0
  for candidate in "$REPO_ROOT"/third_party/qemu-cxl/*.qcow2 "$REPO_ROOT"/third_party/qemu-cxl/*.img; do
    [ -e "$candidate" ] || continue
    found_any=1
    echo "  $candidate"
  done
  if [ "$found_any" -eq 0 ]; then
    echo "  none found under third_party/qemu-cxl/ -- see memsim/README.md's 'QEMU CXL VM" \
         "(optional)' section for how to get a minimal cloud guest image before running 'launch'."
  fi
}

# The exact CXL device chain: a PCIe expander bus (pxb-cxl) carrying a CXL
# root port (cxl-rp) with one CXL Type 3 device (cxl-type3) attached, backed
# by a memory-backend-ram object, plus a machine-level CXL Fixed Memory
# Window (cxl-fmw) declaring which host bridge that memory range routes
# through -- this is QEMU's own documented minimal single-device CXL
# topology (docs/system/devices/cxl.rst in the QEMU source tree), not an
# invented configuration.
build_qemu_args() {
  local disk="$1" size_gib="$2"
  local accel_args=()
  if [ -e /dev/kvm ] && [ -r /dev/kvm ] && [ -w /dev/kvm ]; then
    accel_args=(-enable-kvm -cpu host)
  else
    accel_args=(-accel tcg -cpu max)
  fi

  QEMU_ARGS=(
    -machine "q35,cxl=on"
    -m "8G,maxmem=$((8 + size_gib))G,slots=4"
    "${accel_args[@]}"
    -smp 4
    -drive "file=$disk,format=qcow2,if=virtio"
    -device "pxb-cxl,bus_nr=52,bus=pcie.0,id=cxl.1"
    -device "cxl-rp,port=0,bus=cxl.1,id=root_port0,chassis=1,slot=0"
    -device "cxl-type3,bus=root_port0,memdev=cxl-mem1,id=cxl-vmem1"
    -object "memory-backend-ram,id=cxl-mem1,size=${size_gib}G"
    -M "cxl-fmw.0.targets.0=cxl.1,cxl-fmw.0.size=${size_gib}G"
    -nographic
    -serial mon:stdio
    -netdev "user,id=net0,hostfwd=tcp::2222-:22"
    -device "virtio-net-pci,netdev=net0"
  )
}

cmd_launch_cmd() {
  local disk="${1:?usage: launch-cmd <disk.qcow2> [<cxl-size-gib>]}"
  local size_gib="${2:-$CXL_DEFAULT_SIZE_GIB}"
  [ -f "$disk" ] || die "no such disk image: $disk"
  build_qemu_args "$disk" "$size_gib"
  local qemu_bin
  qemu_bin="$(find_qemu)" || die "no qemu-system-x86_64 on PATH"
  printf '%q ' "$qemu_bin" "${QEMU_ARGS[@]}"
  echo
  echo
  echo "# Once booted, SSH in with:  ssh -p 2222 <user>@localhost"
  echo "# Then verify per memsim/README.md's 'QEMU CXL VM (optional)' section:"
  echo "#   ls /sys/bus/cxl/devices/       # expect a mem0 (or similar) device to appear"
  echo "#   sudo cxl list -M               # cxl-cli: lists CXL memory devices"
  echo "#   sudo daxctl list               # once the region is configured as devdax"
}

cmd_launch() {
  local disk="${1:?usage: launch <disk.qcow2> [<cxl-size-gib>]}"
  local size_gib="${2:-$CXL_DEFAULT_SIZE_GIB}"
  [ -f "$disk" ] || die "no such disk image: $disk"
  build_qemu_args "$disk" "$size_gib"
  local qemu_bin
  qemu_bin="$(find_qemu)" || die "no qemu-system-x86_64 on PATH"
  info "launching (Ctrl-A X to quit the monitor, Ctrl-A C for the QEMU console)"
  echo "+ $qemu_bin ${QEMU_ARGS[*]}"
  exec "$qemu_bin" "${QEMU_ARGS[@]}"
}

case "${1:-}" in
  check)      shift; cmd_check "$@" ;;
  launch)     shift; cmd_launch "$@" ;;
  launch-cmd) shift; cmd_launch_cmd "$@" ;;
  *)
    echo "Usage: $0 {check|launch <disk.qcow2> [size_gib]|launch-cmd <disk.qcow2> [size_gib]}" >&2
    exit 2
    ;;
esac

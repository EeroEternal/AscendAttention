#!/usr/bin/env bash
# Ascend 950 toolchain environment for this repo — deliberately the *same* toolchain, cache and
# library paths as TileInfer (the reference machine is the 102 box), because the operators were
# measured there and a local rebuild against a different CANN would make every ratio incomparable.
#
#   source scripts/env-ascend102.sh
#
# Nothing here creates a venv or installs anything: reuse TileInfer's `.venv_native213`
# (torch 2.13.0+cpu, torch_npu 2.13, tilelang 0.1.15) and put this repo's `src/` on PYTHONPATH.

set -u

_AA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_TI_ROOT="$(dirname "${_AA_ROOT}")/TileInfer"

# --- CANN ---------------------------------------------------------------------------------------
# Discovery order, first hit wins: an explicit ASCEND_HOME, the TileInfer checkout's pinned CANN
# 9.3 tarball, the shared install under /home/dlb, then the system default (9.1.1 — do not use it:
# it is not what the kernels were verified against).
if [ -z "${ASCEND_HOME:-}" ]; then
  for _candidate in \
    "${_TI_ROOT}/.work/cann930/install/cann" \
    /home/dlb/Ascend/install/cann-9.3.0 \
    /usr/local/Ascend/ascend-toolkit/latest; do
    if [ -f "${_candidate}/set_env.sh" ]; then
      export ASCEND_HOME="${_candidate}"
      break
    fi
  done
fi
if [ -n "${ASCEND_HOME:-}" ] && [ -f "${ASCEND_HOME}/set_env.sh" ]; then
  # shellcheck disable=SC1091
  source "${ASCEND_HOME}/set_env.sh" >/dev/null 2>&1
else
  echo "env-ascend102: no CANN found (set ASCEND_HOME); CPU-only work still works" >&2
fi

# --- TileLang kernel cache ----------------------------------------------------------------------
# Cold cache = every kernel recompiled (70-100 s each, holding the GIL). Reuse TileInfer's warmed
# cache so an A/B against the source repo does not pay two cold builds.
export TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR:-${_TI_ROOT}/../tl_cache_213}"
mkdir -p "${TILELANG_CACHE_DIR}" 2>/dev/null || true

# --- runtime flags (same as TileInfer) -----------------------------------------------------------
export ACL_OP_INIT_MODE=1

# glibc symbol versions newer than the system's; CANN needs them on the library path.
if [ -d "${_TI_ROOT}/.work/glibcxx" ]; then
  export LD_LIBRARY_PATH="${_TI_ROOT}/.work/glibcxx:${LD_LIBRARY_PATH:-}"
fi

# --- this repo on the import path ----------------------------------------------------------------
case ":${PYTHONPATH:-}:" in
  *":${_AA_ROOT}/src:"*) ;;
  *) export PYTHONPATH="${_AA_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac

unset _AA_ROOT _TI_ROOT _candidate

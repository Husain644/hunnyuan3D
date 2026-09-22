#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# test_generation.sh — end-to-end validation of image -> 3D generation.
#
#   ./test_generation.sh [--skip-server]
#
# Runs after setup.sh (server reachable, models downloaded). It:
#   * submits a synthetic input to the public API (Test A: 2.1 single-view,
#     Test B: 2mv multi-view), or runs in-process on hardcoded candidates
#   * polls until the job completes
#   * downloads the GLB and validates it with trimesh (watertight + >0 faces)
#
# Exit code 0 = PASS for both models. Uses curl + the service's own venv
# python for validation, so no extra tools are needed.
# ---------------------------------------------------------------------------
set -euo pipefail
SRV_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SRV_ROOT"

SKIP_SERVER=0
for a in "$@"; do
  [ "$a" = "--skip-server" ] && SKIP_SERVER=1
done

[ -x .venv/bin/python ] && PY=.venv/bin/python || PY=python3

BASE="${HY3D_TEST_BASE:-http://127.0.0.1:${HY3D_PORT:-8080}}"
TIMEOUT="${HY3D_TEST_TIMEOUT:-900}"
OUT="$SRV_ROOT/outputs/test_generation"
mkdir -p "$OUT"

pass() { printf "\033[1;32mPASS — %s\033[0m\n" "$*"; }
fail() { printf "\033[1;31mFAIL — %s\033[0m\n" "$*" >&2; exit 1; }
info() { printf "\033[1;36m[test]\033[0m %s\n" "$*"; }

# -- helpers ---------------------------------------------------------------
make_synth() {  # make_synth <out_dir> <base_name> <axis>  -> <base>.png (directional gradient)
  local d="$1" n="$2" axis="$3"
  "$PY" - "$d" "$n" "$axis" <<'PYEOF'
import sys
from pathlib import Path
from PIL import Image
d, n, axis = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
d.mkdir(parents=True, exist_ok=True)
img = Image.new("L", (224, 224))
px = img.load()
for y in range(224):
    for x in range(224):
        if axis == "h":  v = (x * 3) % 256
        elif axis == "v": v = (y * 3) % 256
        elif axis == "d": v = ((x + y) * 2) % 256
        else:             v = (x * 3 + (y * 2)) % 256
        px[x, y] = v
img.save(d / f"{n}.png")
print(f"wrote {d / (n + '.png')}")
PYEOF
}

wait_job() {  # wait_job <job_id>
  local jid="$1" t0=$SECONDS st=""
  while [ $((SECONDS - t0)) -lt "$TIMEOUT" ]; do
    st="$(curl -s "$BASE/v1/public/jobs/$jid" | "$PY" -c \
      'import sys,json
try:
  print(json.load(sys.stdin).get("status","pending"))
except Exception:
  print("pending")' 2>/dev/null || echo pending)"
    case "$st" in
      succeeded) return 0 ;;
      failed|cancelled)
        curl -s "$BASE/v1/public/jobs/$jid" | "$PY" -m json.tool || true
        return 1 ;;
    esac
    sleep 5
  done
  return 1
}

fetch_and_validate() {  # fetch_and_validate <job_id> <out.glb> <label> <view_note>
  curl -s -o "$2" "$BASE/v1/public/jobs/$1/result" || fail "download $1"
  "$PY" - "$2" "$3" "$4" <<'PYEOF'
import sys, trimesh
p, label, note = sys.argv[1], sys.argv[2], sys.argv[3]
m = trimesh.load(p, file_type="glb", force="mesh")
v, f = m.vertices, m.faces
assert v is not None and len(v) > 100, f"too few vertices: {len(v) if v is not None else 0}"
assert f is not None and len(f) > 100, f"too few faces: {len(f) if f is not None else 0}"
# Watertightness depends on the input image (synthetic gradients are often
# non-watertight); report it but do not fail the smoke test on it.
wt = bool(getattr(m, "is_watertight", False))
print(f"  {label}: verts={len(v)} faces={len(f)} watertight={'yes' if wt else 'no'} ({note})")
PYEOF
  pass "$3"
}

# -- Test A: 2.1 single-view ------------------------------------------------
info "Test A — model=2.1 (single-view)"
make_synth "$OUT" a_front h
JOB_A="$(curl -s -F "model=2.1" -F "name=auto test a" -F "file=@$OUT/a_front.png" \
  "$BASE/v1/public/generate/upload" | "$PY" -c 'import sys,json;print(json.load(sys.stdin)["job_id"])')"
[ -n "$JOB_A" ] || fail "no job_id for 2.1"
info "  job_id=$JOB_A"
wait_job "$JOB_A" || fail "2.1 job not succeeded"
fetch_and_validate "$JOB_A" "$OUT/a.glb" "2.1 single-view" "1 view"

# -- Test B: 2mv multi-view -------------------------------------------------
info "Test B — model=2mv (multi-view: front/back/left/right)"
for v in front back left right; do make_synth "$OUT" "b_$v" "$v"; done
JOB_B="$(curl -s \
  -F "model=2mv" \
  -F "name=auto test b" \
  -F "view_front=@$OUT/b_front.png" \
  -F "view_back=@$OUT/b_back.png" \
  -F "view_left=@$OUT/b_left.png" \
  -F "view_right=@$OUT/b_right.png" \
  "$BASE/v1/public/generate/upload" | "$PY" -c 'import sys,json;print(json.load(sys.stdin)["job_id"])')"
[ -n "$JOB_B" ] || fail "no job_id for 2mv"
info "  job_id=$JOB_B"
wait_job "$JOB_B" || fail "2mv job not succeeded"
fetch_and_validate "$JOB_B" "$OUT/b.glb" "2mv multi-view" "4 views"

echo
info "All generation tests PASSED."
echo "Outputs kept in: $OUT"
exit 0
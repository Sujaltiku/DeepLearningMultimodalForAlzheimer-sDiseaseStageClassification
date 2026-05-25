from pathlib import Path
import os
import subprocess
import shutil
import nibabel as nib
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


IN_ROOT  = Path("/Users/sujalsingh/Desktop/450DATASET/MRI")
OUT_ROOT = Path("/Users/sujalsingh/Desktop/450DATASET/MRI_FSL")

BET_FRAC = "0.35"

# Try a few BET fractional thresholds; pick the one with a reasonable mask size
BET_FRACS = [0.30, 0.35, 0.40]
MIN_MASK_VOX = 200_000
MAX_MASK_VOX = 2_000_000


BET_BIN = shutil.which("bet")
FAST_BIN = shutil.which("fast")


FSLDIR = os.environ.get("FSLDIR")
if not FSLDIR:

    try:
        vv = subprocess.run(["fslversion"], capture_output=True, text=True)
        if vv.returncode == 0:
            for line in vv.stdout.splitlines():
                if line.startswith("FSLDIR:"):
                    FSLDIR = line.split(":", 1)[1].strip()
                    break
    except FileNotFoundError:
        pass


if (BET_BIN is None or FAST_BIN is None) and Path.home().joinpath("fslwhich", "share", "fsl", "bin").exists():
    bin_dir = Path.home() / "fslwhich" / "share" / "fsl" / "bin"
    if BET_BIN is None:
        BET_BIN = str(bin_dir / "bet")
    if FAST_BIN is None:
        FAST_BIN = str(bin_dir / "fast")
    if not FSLDIR:
        FSLDIR = str(Path.home() / "fslwhich")

print("BET_BIN:", BET_BIN)
print("FAST_BIN:", FAST_BIN)
print("FSLDIR:", FSLDIR)

if BET_BIN is None or FAST_BIN is None:
    raise FileNotFoundError(
        "FSL commands not found in PATH. Your install appears to be under $HOME/fslwhich. In Terminal, run:\n"
        "  export FSLDIR=$HOME/fslwhich\n"
        "  export PATH=$FSLDIR/share/fsl/bin:$PATH\n"
        "  (optional) source $FSLDIR/share/fsl/etc/fslconf/fsl.sh\n"
        "Then launch PyCharm from that same Terminal (e.g., `open -a PyCharm .`) so it inherits PATH."
    )


nii_files = sorted(
    f for f in IN_ROOT.rglob("*.nii.gz")
    if ("_brain" not in f.name and "_mask" not in f.name and "_restore" not in f.name)
)
print("Found:", len(nii_files))

_iter = tqdm(nii_files, desc="BET+FAST") if tqdm is not None else nii_files
for f in _iter:
    rel = f.relative_to(IN_ROOT)
    out_dir = OUT_ROOT / rel.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = f.name[:-7]
    out_brain = out_dir / f"{stem}_brain.nii.gz"


    if out_brain.exists() and out_brain.with_name(out_brain.name.replace("_brain.nii.gz", "_brain_restore.nii.gz")).exists():
        continue

    env = os.environ.copy()
    if FSLDIR:
        env["FSLDIR"] = FSLDIR


    best = None
    best_frac = None
    best_mask_path = None
    best_brain_path = None

    for frac in BET_FRACS:
        tmp_brain = out_brain.with_name(out_brain.name.replace("_brain.nii.gz", f"_brain_f{frac:.2f}.nii.gz"))
        tmp_mask = tmp_brain.with_name(tmp_brain.name.replace(".nii.gz", "_mask.nii.gz"))

        cmd_bet = [BET_BIN, str(f), str(tmp_brain), "-R", "-B", "-f", f"{frac:.2f}", "-g", "0", "-m"]
        r = subprocess.run(cmd_bet, capture_output=True, text=True, env=env)
        if r.returncode != 0 or (not tmp_mask.exists()):
            continue

        try:
            m = nib.load(str(tmp_mask)).get_fdata() > 0
            mvox = int(m.sum())
        except Exception:
            continue


        score = 0
        if MIN_MASK_VOX <= mvox <= MAX_MASK_VOX:
            score = 10_000_000 - abs(mvox - 900_000)
        else:
            score = -abs(mvox - 900_000)

        if best is None or score > best:
            best = score
            best_frac = frac
            best_mask_path = tmp_mask
            best_brain_path = tmp_brain

    if best_brain_path is None:
        print("BET FAILED (all fracs):", f)
        continue


    final_mask = out_brain.with_name(out_brain.name.replace(".nii.gz", "_mask.nii.gz"))
    if best_brain_path != out_brain:

        if out_brain.exists():
            out_brain.unlink()
        if final_mask.exists():
            final_mask.unlink()
        best_brain_path.replace(out_brain)
        best_mask_path.replace(final_mask)


    for frac in BET_FRACS:
        tmp_brain = out_brain.with_name(out_brain.name.replace("_brain.nii.gz", f"_brain_f{frac:.2f}.nii.gz"))
        tmp_mask = tmp_brain.with_name(tmp_brain.name.replace(".nii.gz", "_mask.nii.gz"))
        if tmp_brain.exists() and tmp_brain != out_brain:
            tmp_brain.unlink(missing_ok=True)
        if tmp_mask.exists() and tmp_mask != final_mask:
            tmp_mask.unlink(missing_ok=True)

    print(f"BET chosen -f={best_frac:.2f} for {f.name}")

    # Bias correction
    cmd_fast = [FAST_BIN, "-B", str(out_brain)]
    r2 = subprocess.run(cmd_fast, capture_output=True, text=True, env=env)
    if r2.returncode != 0:
        print("FAST FAILED:", out_brain)
        print(r2.stderr)

print("Done.")
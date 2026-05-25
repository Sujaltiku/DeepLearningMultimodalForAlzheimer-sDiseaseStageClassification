from pathlib import Path
import numpy as np
import nibabel as nib
from tqdm import tqdm

IN_ROOT  = Path("/Users/sujalsingh/Desktop/180DATASET/MRI_FSL")
OUT_ROOT = Path("/Users/sujalsingh/Desktop/180DATASET/MRI_NORM_FSL")

OUT_ROOT.mkdir(parents=True, exist_ok=True)

restore_files = sorted(IN_ROOT.rglob("*_brain_restore.nii.gz"))
print("Found restore files:", len(restore_files))

for rf in tqdm(restore_files):
    mf = rf.with_name(rf.name.replace("_brain_restore.nii.gz", "_brain_mask.nii.gz"))
    if not mf.exists():
        print("Missing mask for:", rf)
        continue

    img = nib.load(str(rf))
    vol = img.get_fdata().astype(np.float32)

    mask = nib.load(str(mf)).get_fdata() > 0
    if mask.sum() < 1000:
        print("Mask too small, skipping:", rf)
        continue

    vals = vol[mask]
    p1, p99 = np.percentile(vals, [1, 99])

    vol = np.clip(vol, p1, p99)
    vol = (vol - p1) / (p99 - p1 + 1e-8)
    vol[~mask] = 0.0

    rel = rf.relative_to(IN_ROOT)
    out_dir = OUT_ROOT / rel.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / rf.name.replace("_brain_restore.nii.gz", "_norm.nii.gz")
    if out_path.exists():
        print("Skipping as already normalized")
        continue
    nib.save(nib.Nifti1Image(vol.astype(np.float32), img.affine, img.header), str(out_path))

print("Done.")
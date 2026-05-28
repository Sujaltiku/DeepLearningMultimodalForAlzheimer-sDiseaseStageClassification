import os
import glob
import numpy as np
import nibabel as nib
from nilearn.image import resample_to_img

PET_MNI_DIR   = "/home/stuti/fsl_work/New/new_cn_100_pet_mni"
SUVR_OUT_DIR  = "/home/stuti/fsl_work/New/new_cn_100_suvr"
CEREB_MASK    = "/home/stuti/fsl/data/atlases/Cerebellum/Cerebellum-MNIfnirt-maxprob-thr25-1mm.nii.gz"

def compute_suvr(subject_id, pet_mni_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    suvr_path = os.path.join(output_dir, subject_id + "_PET_SUVR.nii.gz")
    if os.path.exists(suvr_path):
        print("  skipping, already done.")
        return suvr_path

    # Load PET and cerebellum mask
    pet_img  = nib.load(pet_mni_path)
    mask_img = nib.load(CEREB_MASK)
    pet_data = pet_img.get_fdata()

    # Resample mask to PET space if needed
    if pet_img.shape != mask_img.shape:
        mask_img  = resample_to_img(mask_img, pet_img, interpolation='nearest')
    mask_data = mask_img.get_fdata()

    # Create binary cerebellum mask (any non-zero label = cerebellum)
    cereb_mask = (mask_data > 0).astype(np.float32)

    # Extract cerebellum voxels from PET
    cereb_voxels = pet_data[cereb_mask > 0]

    # Sanity check
    if len(cereb_voxels) == 0:
        raise RuntimeError("Cerebellum mask has no overlap with PET image!")

    # Compute mean cerebellum uptake
    cereb_mean = np.mean(cereb_voxels)

    # Compute SUVR
    suvr_data = pet_data / cereb_mean

    # Save
    suvr_img = nib.Nifti1Image(suvr_data, pet_img.affine, pet_img.header)
    nib.save(suvr_img, suvr_path)

    print("  Cerebellum mean : " + str(round(cereb_mean, 4)))
    print("  SUVR range      : " + str(round(suvr_data.min(), 3)) + " to " + str(round(suvr_data.max(), 3)))
    print("  Cerebellum SUVR : " + str(round(np.mean(suvr_data[cereb_mask > 0]), 3)) + " (should be ~1.0)")
    print("  Saved           : " + os.path.basename(suvr_path))

    return suvr_path

def run_suvr():
    subjects = sorted([
        d for d in os.listdir(PET_MNI_DIR)
        if os.path.isdir(os.path.join(PET_MNI_DIR, d))
    ])

    print("Computing SUVR for " + str(len(subjects)) + " subjects")
    print("Reference region: FSL Cerebellum mask (MNIfnirt, thr25, 1mm)\n")

    failed = []

    for i, subject_id in enumerate(subjects, 1):
        print("[" + str(i) + "/" + str(len(subjects)) + "] " + subject_id)

        pet_mni   = os.path.join(PET_MNI_DIR,  subject_id, subject_id + "_PET_MNI.nii.gz")
        out_dir   = os.path.join(SUVR_OUT_DIR, subject_id)

        if not os.path.exists(pet_mni):
            print("  PET MNI file not found, skipping.")
            failed.append(subject_id)
            continue

        try:
            compute_suvr(subject_id, pet_mni, out_dir)
            print("  DONE")
        except Exception as e:
            print("  FAILED: " + str(e))
            failed.append(subject_id)

    print("\n" + "="*60)
    print("SUVR complete: " + str(len(subjects)-len(failed)) + "/" + str(len(subjects)) + " succeeded")

    if failed:
        print("Failed: " + str(failed))
    else:
        print("All subjects done! Ready for classification.")

    # Final summary stats across all subjects
    print("\nSummary statistics across all subjects:")
    all_files = glob.glob(os.path.join(SUVR_OUT_DIR, "**", "*_PET_SUVR.nii.gz"), recursive=True)
    means = []
    for f in all_files:
        data = nib.load(f).get_fdata()
        means.append(data[data > 0].mean())
    if means:
        print("  Mean SUVR across subjects : " + str(round(np.mean(means), 3)))
        print("  Std  SUVR across subjects : " + str(round(np.std(means), 3)))
        print("  Min  SUVR across subjects : " + str(round(np.min(means), 3)))
        print("  Max  SUVR across subjects : " + str(round(np.max(means), 3)))

if __name__ == "__main__":
    run_suvr()

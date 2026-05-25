import os
import subprocess
import glob

MRI_BASE         = "MRI_FSL"
PET_BASE         = "PET"
OUTPUT_DIR       = "NEW_PET"
MNI_1MM = "MNI152_T1_1mm.nii.gz"
MNI_2MM = "MNI152_T1_1mm.nii.gz"

def find_file(folder, pattern, required=True):
    matches = glob.glob(os.path.join(folder, "*" + pattern))
    if not matches:
        if required:
            raise FileNotFoundError("No file matching *" + pattern + " in " + folder)
        return None
    return sorted(matches)[0]

def step_A_mri_to_mni(subject_id, mri_dir):
    mri_brain  = find_file(mri_dir, "_rigid_brain.nii.gz")
    affine_mat = os.path.join(mri_dir, subject_id + "_MRI2MNI_affine.mat")
    warp_file  = os.path.join(mri_dir, subject_id + "_MRI2MNI_warp.nii.gz")
    mri_mni    = os.path.join(mri_dir, subject_id + "_MRI_MNI.nii.gz")

    if os.path.exists(affine_mat):
        print("  skipping FLIRT affine, already done.")
    else:
        print("  Running FLIRT MRI to MNI affine 12 DOF...")
        cmd = ("flirt -in " + mri_brain +
               " -ref " + MNI_2MM +          # FIXED: must match FNIRT ref (voxel-space matrix is grid-specific)
               " -omat " + affine_mat +
               " -out " + mri_mni +
               " -dof 12 -cost corratio -interp trilinear")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError("FLIRT failed: " + result.stderr)
        print("  FLIRT done: " + os.path.basename(affine_mat))

    if os.path.exists(warp_file):
        print("  skipping FNIRT warp, already done.")
    else:
        print("  Running FNIRT using 2mm template for speed...")
        cmd = ("fnirt --in=" + mri_brain +
               " --ref=" + MNI_2MM +
               " --aff=" + affine_mat +
               " --cout=" + warp_file +
               " --iout=" + mri_mni +
               " --subsamp=4,2,1,1 --miter=5,5,5,10 --lambda=300,150,100,50 --estint=1,1,1,0")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError("FNIRT failed: " + result.stderr)
        print("  FNIRT done: " + os.path.basename(warp_file))

    return affine_mat, warp_file

def step2_coreg_pet_to_mri(subject_id, pet_dir, mri_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    pet_nii   = find_file(pet_dir, "_PET.nii.gz")
    mri_brain = find_file(mri_dir, "_rigid_brain.nii.gz")
    pet_coreg = os.path.join(output_dir, subject_id + "_PET_coreg.nii.gz")
    pet2mri   = os.path.join(output_dir, subject_id + "_PET2MRI.mat")
    if os.path.exists(pet_coreg):
        print("  skipping PET to MRI coreg, already done.")
        return pet_coreg, pet2mri
    print("  Running FLIRT PET to MRI rigid 6 DOF...")
    cmd = ("flirt -in " + pet_nii +
           " -ref " + mri_brain +
           " -out " + pet_coreg +
           " -omat " + pet2mri +
           " -dof 6 -cost mutualinfo -searchrx -20 20 -searchry -20 20 -searchrz -20 20 -interp trilinear")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("FLIRT PET to MRI failed: " + result.stderr)
    print("  PET coreg done: " + os.path.basename(pet_coreg))
    return pet_coreg, pet2mri

def step3_warp_pet_to_mni(subject_id, pet_coreg, warp_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    pet_mni = os.path.join(output_dir, subject_id + "_PET_MNI.nii.gz")
    if os.path.exists(pet_mni):
        print("  skipping applywarp, already done.")
        return pet_mni
    for label, path in [("pet_coreg", pet_coreg), ("warp_file", warp_file)]:
        if not path or not os.path.exists(path):
            raise FileNotFoundError("Missing " + label + ": " + str(path))
    print("  Running applywarp PET to MNI (1mm output)...")
    cmd = ("applywarp --in=" + pet_coreg +
           " --ref=" + MNI_1MM +              # 1mm output for accuracy, fine since applywarp uses mm-space warp
           " --warp=" + warp_file +
           " --out=" + pet_mni +
           " --interp=trilinear")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("applywarp failed: " + result.stderr)
    print("  applywarp done: " + os.path.basename(pet_mni))
    return pet_mni

def verify_output(pet_mni_path, subject_id):
    import nibabel as nib
    img  = nib.load(pet_mni_path)
    data = img.get_fdata()
    ok   = img.shape == (182, 218, 182) and data.max() > 0
    print("  Shape: " + str(img.shape) + " | Range: " + str(round(data.min(),3)) + " to " + str(round(data.max(),3)) + " | " + ("PASS" if ok else "FAIL"))

def run_pipeline():
    subjects = sorted(set(os.listdir(MRI_BASE)) & set(os.listdir(PET_BASE)))
    print("Found " + str(len(subjects)) + " subjects")
    print("FNIRT uses 2mm template for speed, applywarp outputs 1mm for accuracy\n")
    failed = []
    for i, subject_id in enumerate(subjects, 1):
        print("\n" + "="*60)
        print("  [" + str(i) + "/" + str(len(subjects)) + "] " + subject_id)
        print("="*60)
        mri_dir    = os.path.join(MRI_BASE, subject_id)
        pet_dir    = os.path.join(PET_BASE, subject_id)
        output_dir = os.path.join(OUTPUT_DIR, subject_id)
        pet_mni_check = os.path.join(output_dir, subject_id + "_PET_MNI.nii.gz")
        if os.path.exists(pet_mni_check):
            print("  All outputs exist, skipping entire subject.")
            continue
        try:
            affine_mat, warp_file  = step_A_mri_to_mni(subject_id, mri_dir)
            pet_coreg, pet2mri_mat = step2_coreg_pet_to_mri(subject_id, pet_dir, mri_dir, output_dir)
            pet_mni                = step3_warp_pet_to_mni(subject_id, pet_coreg, warp_file, output_dir)
            verify_output(pet_mni, subject_id)
            print("  DONE: " + subject_id)
        except Exception as e:
            print("  FAILED: " + str(e))
            failed.append(subject_id)
    print("\nPipeline complete: " + str(len(subjects)-len(failed)) + "/" + str(len(subjects)) + " succeeded")
    for s in failed:
        print("  FAILED: " + s)

if __name__ == "__main__":
    run_pipeline()
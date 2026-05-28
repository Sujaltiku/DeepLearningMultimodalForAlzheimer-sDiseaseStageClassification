import os
print("Before ants import")

import ants
print("After ants import")

print("Script started")
print("MRI_ROOT exists:", os.path.exists("/Users/sujalsingh/Desktop/450DATASET/MRI"))
print("PET_ROOT exists:", os.path.exists("/Users/sujalsingh/Desktop/450DATASET/PET"))

MRI_ROOT = "/Users/sujalsingh/Desktop/450DATASET/MRI"
PET_ROOT = "/Users/sujalsingh/Desktop/450DATASET/PET"

for group in ["AD", "CN", "MCI"]:
    pet_group_dir = os.path.join(PET_ROOT, group)
    mri_group_dir = os.path.join(MRI_ROOT, group)

    if not os.path.exists(pet_group_dir):
        continue

    for patient in os.listdir(pet_group_dir):
        pet_patient_dir = os.path.join(pet_group_dir, patient)
        mri_patient_dir = os.path.join(mri_group_dir, patient)

        if not os.path.isdir(pet_patient_dir):
            continue
        if not os.path.isdir(mri_patient_dir):
            print(f"⚠ MRI missing for {patient}, skipping")
            continue

        # Get PET & MRI files
        pet_files = [f for f in os.listdir(pet_patient_dir) if f.lower().endswith((".nii", ".nii.gz"))]
        mri_files = [f for f in os.listdir(mri_patient_dir) if f.lower().endswith((".nii", ".nii.gz"))]

        print("Processing:", group, patient)
        print("PET files:", pet_files)
        print("MRI files:", mri_files)

        if len(pet_files) != 1 or len(mri_files) != 1:
            print(f"⚠ File count issue for {patient}, skipping")
            continue

        pet_path = os.path.join(pet_patient_dir, pet_files[0])
        mri_path = os.path.join(mri_patient_dir, mri_files[0])

        filename = os.path.splitext(os.path.splitext(pet_files[0])[0])[0]

        # Load images
        pet_img = ants.image_read(pet_path)
        mri_img = ants.image_read(mri_path)

        # 🔁 Reorient PET to LPS
        pet_lps = ants.reorient_image2(pet_img, orientation='LPS')


        print(f"Reoriented PET: {pet_path}")

        # 🔗 Rigid registration: PET → MRI
        registration = ants.registration(
            fixed=mri_img,
            moving=pet_lps,
            type_of_transform='Rigid'
        )

        # --- Save rigid-registered PET only (no normalization) ---
        pet_registered = registration['warpedmovout']

        out_path = os.path.join(pet_patient_dir, f"{filename}_rigid.nii.gz")
        ants.image_write(pet_registered, out_path)
        # 🗑 Delete original PET file after successful save
        try:
            if os.path.exists(pet_path):
                os.remove(pet_path)
                print(f"Deleted original PET: {pet_path}")
        except Exception as e:
            print(f"Could not delete {pet_path}: {e}")
        print(f"Rigid PET registered to MRI: {out_path}")
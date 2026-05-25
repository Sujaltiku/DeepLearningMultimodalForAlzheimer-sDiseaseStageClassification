import os
import ants


root_dir = "MRI"
template_path = "MNI152_T1_1mm.nii.gz"
template_img = ants.image_read(template_path)

for group in ["AD", "CN", "MCI"]:
    group_dir = os.path.join(root_dir, group)
    if not os.path.exists(group_dir):
        continue


    for patient in os.listdir(group_dir):
        patient_dir = os.path.join(group_dir, patient)
        if not os.path.isdir(patient_dir):
            continue


        for file in os.listdir(patient_dir):
            if file.endswith(".nii") or file.endswith(".nii.gz"):
                file_path = os.path.join(patient_dir, file)
                filename = os.path.splitext(os.path.splitext(file)[0])[0]

                # Load image using ANTs
                img = ants.image_read(file_path)

                # Reorient to LPS
                img_lps = ants.reorient_image2(img, orientation='LPS')

                print(f"Reoriented: {file_path}")

                # Perform rigid registration to MNI152 LPS template
                registration = ants.registration(fixed=template_img, moving=img_lps, type_of_transform='Rigid')


                rigid_save_path = os.path.join(patient_dir, f"{filename}_rigid.nii.gz")
                ants.image_write(registration['warpedmovout'], rigid_save_path)

                print(f"Rigid registered image saved: {rigid_save_path}")


                os.remove(file_path)
                print(f"Deleted original file: {file_path}")
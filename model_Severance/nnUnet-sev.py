import os

source_directory = "D:\\nnUNet\\nnunetv2\\nnUNet_raw\\Dataset001_Liver\\labelsTr new"



import os
import nibabel as nib

labels_directory = "D:\\nnUNet\\nnunetv2\\nnUNet_raw\\Dataset001_Liver\\labelsTr new"

import os
import nibabel as nib
import numpy as np

labels_directory = "D:\\nnUNet\\nnunetv2\\nnUNet_raw\\Dataset001_Liver\\labelsTr new"

for filename in os.listdir(labels_directory):
    if filename.endswith(".nii.gz"):
        file_path = os.path.join(labels_directory, filename)
        image = nib.load(file_path)
        data = image.get_fdata()
        
        # 부동 소수점 값 반올림

        # 라벨이 3.0 이상인 경우 0으로 변경
        data = np.round(data)
        data[data >= 3] = 0

        
        modified_data = np.round(data)
        # 변경된 데이터로 새 이미지 생성
        new_image = nib.Nifti1Image(modified_data.astype(np.int16), image.affine, image.header)
        
        # 이미지 저장
        nib.save(new_image, file_path)
        print(f"Processed {filename}")

        unique_labels = set(data.flatten())
        print(f"{filename}: {unique_labels}")

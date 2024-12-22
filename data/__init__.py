import os
from pathlib import Path


dataset_dir = "/teamspace/studios/this_studio/datasets/XLSR"
output_dir = "/teamspace/studios/this_studio/XLSR/output"
os.makedirs(output_dir, exist_ok=True)
train_data_dir = os.path.join(dataset_dir, "DIV2K_train_HR")
val_data_dir = os.path.join(dataset_dir, "DIV2K_valid_HR")

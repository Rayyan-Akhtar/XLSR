import os
from pathlib import Path
from utils import current_dir


dataset_dir = os.path.join(current_dir, "dataset")
output_dir = os.path.join(current_dir, "output")
os.makedirs(output_dir, exist_ok=True)
train_data_dir = os.path.join(dataset_dir, "DIV2K_train_HR")
val_data_dir = os.path.join(dataset_dir, "DIV2K_valid_HR")

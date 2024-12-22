from torch.utils.data import Dataset, DataLoader
from lightning import LightningDataModule
from data import train_data_dir, val_data_dir
from glob import glob
from PIL import Image
from torchvision import transforms
import os


class Div2kDataset(Dataset):
    def __init__(self, phase="train"):
        super(Div2kDataset, self).__init__()
        self.phase = phase
        if self.phase == "train":
            self.files = glob(os.path.join(train_data_dir, "*.png"))
        if self.phase == "val":
            self.files = glob(os.path.join(val_data_dir, "*.png"))[:10]
        self.dataset_len = len(self.files)
        self.input_transform = self.transform((128, 128))
        self.output_transform = self.transform((512, 512))
        self.crop = transforms.RandomCrop((512, 512))

    @staticmethod
    def transform(size):
        return transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
            ])

    def read_image(self, path):
        return Image.open(path).convert("RGB")

    def __len__(self):
        return self.dataset_len
    
    def __getitem__(self, index):
        image = self.read_image(self.files[index])
        crop = self.crop(image)
        return self.input_transform(crop), self.output_transform(crop)


class Div2kLightningModule(LightningDataModule):
    def __init__(self, train_bs, val_bs):
        super(Div2kLightningModule, self).__init__()
        self.train_bs, self.val_bs = train_bs, val_bs
        self.train_dataset = DataLoader(
            dataset=Div2kDataset("train"),
            batch_size=self.train_bs,
            shuffle=True,
            num_workers=os.cpu_count(),
            pin_memory=True
        )

        self.val_dataset = DataLoader(
            dataset=Div2kDataset("val"),
            batch_size=self.val_bs,
            shuffle=False,
            num_workers=os.cpu_count(),
            pin_memory=True
        )

    def prepare_data(self):
        pass

    def setup(self, stage):
        pass

    def on_exception(self, exception):
        pass

    def teardown(self, stage):
        pass

    def train_dataloader(self):
        return self.train_dataset
    
    def val_dataloader(self):
        return self.val_dataset
    
    def test_dataloader(self):
        return self.test_dataset
    

Div2kDataloader = Div2kLightningModule(train_bs=16, val_bs=1)


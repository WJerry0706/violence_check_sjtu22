import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pytorch_lightning import LightningDataModule


class CustomDataset(Dataset):
    def __init__(self, split):
        assert split in ["train", "val", "test"]
        data_root = "violence/"
        self.data = [os.path.join(data_root, split, i) for i in os.listdir(data_root + split)]

        if split == "train":
            self.transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                transforms.ToTensor(),
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = self.data[index]
        x = Image.open(img_path).convert('RGB')
        filename = os.path.basename(img_path)
        y = int(filename[0])
        x = self.transforms(x)
        return x, y


class CustomDataModule(LightningDataModule):
    def __init__(self, batch_size=32, num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = CustomDataset("train")
        self.val_dataset = CustomDataset("val")
        self.test_dataset = CustomDataset("test")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

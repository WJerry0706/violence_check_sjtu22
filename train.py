import os
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics import Accuracy

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


class ViolenceClassifier(LightningModule):
    def __init__(self, num_classes=2, learning_rate=1e-3, weight_decay=2e-6):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes)
        )

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=2)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = self.accuracy(logits, y)
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        acc = self.accuracy(logits, y)
        self.log('test_acc', acc)
        return acc


def main():
    gpu_id = [0]
    lr = 1e-5
    batch_size = 256
    log_name = "resnet50_pretrain_test"
    print("{} gpu: {}, batch size: {}, lr: {}".format(log_name, gpu_id, batch_size, lr))

    data_module = CustomDataModule(batch_size=batch_size)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename=log_name + '-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min',
        verbose=True
    )

    logger = TensorBoardLogger("train_logs", name=log_name)

    trainer = Trainer(
        max_epochs=40,
        accelerator='gpu',
        devices=gpu_id,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback]
    )

    model = ViolenceClassifier(learning_rate=lr)
    trainer.fit(model, data_module)

if __name__ == '__main__':
    main()

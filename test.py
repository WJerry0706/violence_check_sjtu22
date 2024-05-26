import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from model import ViolenceClassifier
from dataset import CustomDataModule

def main():
    gpu_id = [0]
    batch_size = 256
    log_name = "train_logs/resnet50_pretrain_test"
    checkpoint_path = log_name + "/version_5/checkpoints/resnet50_pretrain_test-epoch=16-val_loss=0.06.ckpt"  # 替换为你的模型检查点路径

    print(f"{log_name} gpu: {gpu_id}, batch size: {batch_size}")

    data_module = CustomDataModule(batch_size=batch_size)

    # 实例化模型
    model = ViolenceClassifier.load_from_checkpoint(checkpoint_path)

    # 实例化训练器，只进行测试，不进行训练
    trainer = Trainer(
        accelerator='gpu',
        devices=gpu_id
    )

    # 运行测试
    trainer.test(model, data_module)

if __name__ == '__main__':
    main()

import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from model import ViolenceClassifier


def fgsm_attack(image, epsilon, data_grad):
    print("Performing FGSM attack...")
    # 获取扰动方向的符号
    sign_data_grad = data_grad.sign()
    # 生成对抗样本
    perturbed_image = image + epsilon * sign_data_grad
    # 保持图像在有效范围内
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


def generate_adversarial_samples(input_folder, output_folder, model, device, epsilon=0.1):
    print(f"Generating adversarial samples from {input_folder} to {output_folder}...")
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 准备图像转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # 遍历输入文件夹中的所有图片
    for img_file in os.listdir(input_folder):
        img_path = os.path.join(input_folder, img_file)

        if not os.path.isfile(img_path):
            continue

        print(f"Processing {img_file}...")
        # 加载图像并进行转换
        image = Image.open(img_path).convert('RGB')
        image = transform(image)
        image = image.unsqueeze(0)  # 添加批次维度

        image = image.to(device)
        image.requires_grad = True

        # 前向传播
        output = model(image)
        initial_pred = output.max(1, keepdim=True)[1]  # 获取预测类别

        # 计算损失
        label = initial_pred.view(-1)  # 确保label是1D张量
        loss = F.cross_entropy(output, label)

        # 反向传播
        model.zero_grad()
        loss.backward()

        # 收集数据梯度
        data_grad = image.grad.data

        # 调用FGSM生成对抗样本
        perturbed_image = fgsm_attack(image, epsilon, data_grad)

        # 转换对抗样本图像以便保存
        perturbed_image = perturbed_image.squeeze().detach().cpu().numpy()
        perturbed_image = np.transpose(perturbed_image, (1, 2, 0))
        perturbed_image = (perturbed_image * 255).astype(np.uint8)  # 转换为uint8类型

        # 保存对抗样本图像
        perturbed_img = Image.fromarray(perturbed_image)
        output_path = os.path.join(output_folder, f"{img_file}")
        perturbed_img.save(output_path)

        print(f"Saved adversarial image to {output_path}")


# 加载模型并移到指定设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = ViolenceClassifier.load_from_checkpoint('train_logs/resnet18_pretrain_test/version_22/checkpoints/resnet18_pretrain_test-epoch=19-val_loss=0.07.ckpt')
model = model.to(device)
model.eval()

# 输入文件夹和输出文件夹
input_folder = 'before_gan'  # 替换为实际的输入文件夹路径
output_folder = 'after_gan'  # 替换为实际的输出文件夹路径

# 生成对抗样本
generate_adversarial_samples(input_folder, output_folder, model, device, epsilon=0.1)

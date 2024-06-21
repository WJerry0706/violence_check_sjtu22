import os
import torch
from torchvision import transforms
from PIL import Image
from model import ViolenceClassifier
from datetime import datetime


class ViolenceClass:
    def __init__(self, checkpoint_path):
        # 加载模型
        self.model = ViolenceClassifier.load_from_checkpoint(checkpoint_path)
        self.model.eval()  # 设置模型为评估模式

        # 如果有GPU，移动模型到GPU
        if torch.cuda.is_available():
            self.model.cuda()

        # 定义图像预处理
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def classify(self, tensor):
        # 确保输入是PyTorch Tensor
        assert isinstance(tensor, torch.Tensor), "Input must be a PyTorch Tensor"
        assert tensor.size(1) == 3 and tensor.size(2) == 224 and tensor.size(
            3) == 224, "Tensor shape must be n*3*224*224"

        # 如果有GPU，移动输入到GPU
        if torch.cuda.is_available():
            tensor = tensor.cuda()

        # 使用模型进行预测
        with torch.no_grad():
            logits = self.model(tensor)
            predictions = torch.argmax(logits, dim=1)

        # 将预测结果转换为Python列表
        return predictions.tolist()

    def classify_images_in_directory(self, directory_path):
        image_paths = [os.path.join(directory_path, img_name) for img_name in os.listdir(directory_path) if
                       img_name.endswith(('.png', '.jpg', '.jpeg'))]
        images = [self.transforms(Image.open(img_path).convert('RGB')) for img_path in image_paths]
        tensor = torch.stack(images)  # 将所有图像堆叠成一个张量
        return self.classify(tensor)


# 测试接口类
if __name__ == "__main__":
    # 模型检查点路径
    log_name = "train_logs/resnet50_pretrain_test"
    checkpoint_path = log_name + "/version_6/checkpoints/resnet50_pretrain_test-epoch=16-val_loss=0.06.ckpt"  # 替换为你的模型检查点路径

    # 创建接口类实例
    violence_classifier = ViolenceClass(checkpoint_path)

    # violence/test 目录路径
    test_directory = "violence/test"

    # 对目录下的所有图片进行分类
    results = violence_classifier.classify_images_in_directory(test_directory)

    # 打印预测结果
    print(results)  # 打印每张图片的预测结果

    # 获取当前时间并格式化
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 创建输出目录
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 保存预测结果列表到文件
    results_file = os.path.join(output_dir, f"predictions_list_{current_time}.txt")
    with open(results_file, 'w') as f:
        for prediction in results:
            f.write(f"{prediction}\n")

    # 将预测结果转换为张量
    results_tensor = torch.tensor(results)

    # 保存张量到文件
    tensor_file = os.path.join(output_dir, f"predictions_tensor_{current_time}.pt")
    torch.save(results_tensor, tensor_file)

    print(f"Predictions list saved to {results_file}")
    print(f"Predictions tensor saved to {tensor_file}")

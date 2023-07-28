import os
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# 加载预训练的ResNet模型
resnet = models.resnet50(pretrained=True)
# 将模型设置为评估模式
resnet.eval()

# 图像预处理的转换
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 图像文件夹路径
image_folder = r"C:\Users\64683\Desktop\code2023\CCFCRec-main\data\WeMapDatasets"
output_file = r"C:\Users\64683\Desktop\code2023\CCFCRec-main\data\wemap_features"

# 递归遍历文件夹中的所有图像
def process_images(folder):
    features_list = []  # 存储特征向量的列表
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isdir(file_path):
            features = process_images(file_path)  # 递归处理子文件夹
            features_list.extend(features)  # 将子文件夹中的特征向量添加到列表中
        elif filename.endswith(('.jpg', '.jpeg', '.png')):  # 仅处理图像文件
            image = Image.open(file_path).convert("RGBA")  # 将图像转换为RGBA模式
            image = image.convert("RGB")  # 将RGBA图像转换为RGB图像
            input_tensor = preprocess(image)
            input_batch = input_tensor.unsqueeze(0)

            # 使用GPU进行推理（如果可用）
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            input_batch = input_batch.to(device)
            resnet.to(device)

            # 前向传播获取特征向量
            with torch.no_grad():
                features = resnet(input_batch)

            # 将特征向量添加到列表
            feature_array = features.squeeze().cpu().numpy()
            features_list.append(feature_array)

    return features_list

# 处理图像文件夹及其子文件夹中的图像
all_features = process_images(image_folder)

# 合并特征向量并保存为npy文件
combined_features = np.concatenate(all_features, axis=0)
np.save(output_file, combined_features)


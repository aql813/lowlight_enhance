import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from config import Config
import lime  # 导入LIME算法模块


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
uretinex_model = torch.jit.load(os.path.join(Config.MODEL_FOLDER, 'lowlight_enhancement1.pt'))
uretinex_model = uretinex_model.to(device)
uretinex_model.eval()


# 图像预处理 )
def preprocess_image_for_uretinex(image_pil, target_size=(512, 512)):
    """预处理PIL图像为URetinex模型输入"""
    # 调整大小
    if image_pil.size != target_size:
        image_pil = image_pil.resize(target_size, Image.LANCZOS)

    # 转换为Tensor
    image_np = np.array(image_pil).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)  # HWC to BCHW

    return image_tensor


# 后处理
def tensor_to_image(tensor):
    tensor = tensor.squeeze(0).permute(1, 2, 0)  # BCHW to HWC
    tensor = torch.clamp(tensor, 0, 1)
    array = (tensor.cpu().detach().numpy() * 255).astype(np.uint8)
    return Image.fromarray(array)


def enhance_image(input_path, output_path, ratio=5.0):
    """增强图像并支持比例参数"""
    # 读取图像
    image = Image.open(input_path).convert('RGB')

    # 预处理
    input_tensor = preprocess_image_for_uretinex(image).to(device)
    ratio_tensor = torch.tensor([ratio]).to(device)  # 使用传入的比例值

    # 推理
    with torch.no_grad():
        output_tensor = uretinex_model(input_tensor, ratio_tensor)

    # 后处理
    result_image = tensor_to_image(output_tensor)

    # 保存结果
    result_image.save(output_path)

    return output_path

# LIME增强函数
def enhance_image_with_lime(input_path, output_path, weight_strategy=3, gamma=0.7, std_dev=0.06):
    return lime.enhance_image_with_lime(input_path, output_path, weight_strategy, gamma, std_dev)
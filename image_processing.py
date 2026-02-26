from PIL import Image
import numpy as np
import os

def resize_image(image, size):
    """调整图像大小"""
    return image.resize(size, Image.LANCZOS)

def normalize_image(image):
    """归一化图像到0-1范围"""
    image_array = np.array(image).astype(np.float32)
    return image_array / 255.0

def denormalize_image(image_array):
    """反归一化图像"""
    image_array = image_array * 255.0
    return image_array.clip(0, 255).astype(np.uint8)

def save_image(image_array, output_path):
    """保存图像"""
    image = Image.fromarray(image_array)
    image.save(output_path)
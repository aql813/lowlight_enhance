import os
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
from bm3d import bm3d
from typing import Union
from PIL import Image


def d_sparse_matrices(illumination_map: np.ndarray) -> csr_matrix: ##稀释矩阵，图像含大量零，记录非零防止爆炸
    image_x_shape = illumination_map.shape[-1]
    image_size = illumination_map.size
    dx_row, dx_col, dx_value = [], [], []
    dy_row, dy_col, dy_value = [], [], []

    for i in range(image_size - 1):
        if image_x_shape + i < image_size:
            dy_row += [i, i]
            dy_col += [i, image_x_shape + i]
            dy_value += [-1, 1]
        if (i + 1) % image_x_shape != 0 or i == 0:
            dx_row += [i, i]
            dx_col += [i, i + 1]
            dx_value += [-1, 1]

    d_x_sparse = csr_matrix((dx_value, (dx_row, dx_col)), shape=(image_size, image_size))
    d_y_sparse = csr_matrix((dy_value, (dy_row, dy_col)), shape=(image_size, image_size))

    return d_x_sparse, d_y_sparse


def partial_derivative_vectorized(input_matrix: np.ndarray, toeplitz_sparse_matrix: csr_matrix) -> np.ndarray:
    input_size = input_matrix.size
    output_shape = input_matrix.shape
    vectorized_matrix = input_matrix.reshape((input_size, 1))
    matrices_product = toeplitz_sparse_matrix @ vectorized_matrix
    p_derivative = matrices_product.reshape(output_shape)
    return p_derivative


def gaussian_weight(grad: np.ndarray, size: int, sigma: Union[int, float], epsilon: float) -> np.ndarray:
    radius = int((size - 1) / 2)
    denominator = epsilon + gaussian_filter(np.abs(grad), sigma, radius=radius, mode='constant')
    weights = gaussian_filter(1 / denominator, sigma, radius=radius, mode='constant')
    return weights


def initialize_weights(ill_map: np.ndarray, strategy_n: int, epsilon: float = 0.001) -> np.ndarray:
    if strategy_n == 1:
        weights = np.ones(ill_map.shape)
        weights_x = weights
        weights_y = weights
    elif strategy_n == 2:
        d_x, d_y = d_sparse_matrices(ill_map)
        grad_t_x = partial_derivative_vectorized(ill_map, d_x)
        grad_t_y = partial_derivative_vectorized(ill_map, d_y)
        weights_x = 1 / (np.abs(grad_t_x) + epsilon)
        weights_y = 1 / (np.abs(grad_t_y) + epsilon)
    else:
        sigma = 2
        size = 15
        d_x, d_y = d_sparse_matrices(ill_map)
        grad_t_x = partial_derivative_vectorized(ill_map, d_x)
        grad_t_y = partial_derivative_vectorized(ill_map, d_y)
        weights_x = gaussian_weight(grad_t_x, size, sigma, epsilon)
        weights_y = gaussian_weight(grad_t_y, size, sigma, epsilon)

    modified_w_x = weights_x / (np.abs(grad_t_x) + epsilon)
    modified_w_y = weights_y / (np.abs(grad_t_y) + epsilon)
    flat_w_x = modified_w_x.flatten()
    flat_w_y = modified_w_y.flatten()

    return flat_w_x, flat_w_y


def update_illumination_map(ill_map: np.ndarray, weight_strategy: int = 3) -> np.ndarray:
    vectorized_t = ill_map.reshape((ill_map.size, 1))
    epsilon = 0.001
    alpha = 0.15

    d_x_sparse, d_y_sparse = d_sparse_matrices(ill_map)
    flatten_wiegths_x, flatten_wiegths_y = initialize_weights(ill_map, weight_strategy, epsilon)

    diag_weights_x = diags(flatten_wiegths_x)
    diag_weights_y = diags(flatten_wiegths_y)

    x_term = d_x_sparse.transpose() @ diag_weights_x @ d_x_sparse
    y_term = d_y_sparse.transpose() @ diag_weights_y @ d_y_sparse
    identity = diags(np.ones(x_term.shape[0]))
    matrix = identity + alpha * (x_term + y_term)

    updated_t = spsolve(csr_matrix(matrix), vectorized_t)
    return updated_t.reshape(ill_map.shape)


def gamma_correction(ill_map: np.ndarray, gamma: Union[int, float]) -> np.ndarray:
    return ill_map ** gamma


def bm3d_yuv_denoising(image: np.ndarray, cor_ill_map: np.ndarray, std_dev: Union[int, float] = 0.02) -> np.ndarray:
    image_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    y_channel = image_yuv[:, :, 0]
    denoised_y_ch = bm3d(y_channel, std_dev)
    image_yuv[:, :, 0] = denoised_y_ch
    denoised_rgb = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)
    recombined_image = image * cor_ill_map + denoised_rgb * (1 - cor_ill_map)
    return np.clip(recombined_image, 0, 1).astype("float32")


def enhance_image_with_lime(input_path, output_path, weight_strategy=3, gamma_val=0.7, std_dev_val=0.06):
    # 读取图像
    image_read = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if image_read is None:
        raise ValueError(f"无法读取图像: {input_path}")

    # 转换为RGB并归一化
    image_rgb = cv2.cvtColor(image_read, cv2.COLOR_BGR2RGB) / 255.0

    # 计算初始光照图
    illumination_map = np.max(image_rgb, axis=-1)

    # 更新光照图
    updated_ill_map = update_illumination_map(illumination_map, weight_strategy)

    # 伽马校正
    corrected_ill_map = gamma_correction(np.abs(updated_ill_map), gamma_val)
    corrected_ill_map = corrected_ill_map[..., np.newaxis]  # 增加通道维度

    # 反射图
    new_image = image_rgb / corrected_ill_map
    new_image = np.clip(new_image, 0, 1).astype("float32")

    # BM3D去噪
    denoised_image = bm3d_yuv_denoising(new_image, corrected_ill_map, std_dev_val)

    # 将结果转换回0-255范围并保存
    denoised_image_uint8 = (denoised_image * 255).astype(np.uint8)
    denoised_image_bgr = cv2.cvtColor(denoised_image_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, denoised_image_bgr)

    return output_path
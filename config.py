import os

basedir = os.path.abspath(os.path.dirname(__file__))


class Config:
    # 基础配置
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'

    # 文件上传配置
    UPLOAD_FOLDER = os.path.join(basedir, 'static', 'uploads')
    RESULT_FOLDER = os.path.join(basedir, 'static', 'results')
    MODEL_FOLDER = os.path.join(basedir, 'models')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

    # 图像处理配置
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    IMAGE_SIZE = (512, 512)  # 模型期望的输入尺寸
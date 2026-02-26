from flask import Flask, render_template, request, send_from_directory, jsonify
import os
from datetime import datetime
from model_utils import enhance_image, enhance_image_with_lime
from config import Config

app = Flask(__name__)
app.config.from_object(Config)

# 确保上传和结果目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    algorithm = request.form.get('algorithm', 'deeplearning')  # 获取算法选择
    ratio = float(request.form.get('ratio', '5.0'))  # 获取比例值

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        # 保存原始文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # 增强处理
        result_filename = f"enhanced_{filename}"
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)

        try:
            # 根据选择的算法进行处理
            if algorithm == 'lime':
                enhance_image_with_lime(filepath, result_path)
            else:
                # 修改enhance_image函数以接受比例参数
                enhance_image(filepath, result_path, ratio=ratio)

            return jsonify({
                'original': filename,
                'enhanced': result_filename,
                'algorithm': algorithm,
                'ratio': ratio  # 返回使用的比例值
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/uploads/<filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def serve_result(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
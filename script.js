// 页面加载完成后执行
document.addEventListener('DOMContentLoaded', function() {
    // 获取DOM元素
    const fileInput = document.getElementById('fileInput');
    const uploadBtn = document.getElementById('uploadBtn');
    const algorithmSelect = document.getElementById('algorithm');
    const processingSection = document.querySelector('.processing-section');
    const resultSection = document.querySelector('.result-section');
    const originalImage = document.getElementById('originalImage');
    const enhancedImage = document.getElementById('enhancedImage');
    const algorithmResult = document.getElementById('algorithmResult');
    const progressBar = document.querySelector('.progress-bar');
    const processingTime = document.getElementById('processingTime');
    const downloadBtn = document.getElementById('downloadBtn');
    const timeEstimate = document.getElementById('timeEstimate');

    // 上传按钮点击事件
    uploadBtn.addEventListener('click', function() {
        fileInput.click();
    });

    // 文件选择事件
    fileInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (!file) return;

        // 显示处理区域
        processingSection.style.display = 'block';
        resultSection.style.display = 'none';

        // 模拟进度条
        simulateProgress();

        // 显示原始图像
        const reader = new FileReader();
        reader.onload = function(event) {
            originalImage.src = event.target.result;

            // 创建FormData对象用于上传
            const formData = new FormData();
            formData.append('file', file);
            formData.append('algorithm', algorithmSelect.value);
            formData.append('ratio', '5'); // 固定比例值为5

            // 记录开始时间
            const startTime = new Date();

            // 发送请求到服务器
            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('服务器响应错误');
                }
                return response.json();
            })
            .then(data => {
                // 计算处理时间
                const endTime = new Date();
                const duration = (endTime - startTime) / 1000;

                // 隐藏处理区域
                processingSection.style.display = 'none';

                // 显示结果区域
                resultSection.style.display = 'block';

                // 设置增强图像
                enhancedImage.src = `/results/${data.enhanced}`;

                // 设置算法名称
                algorithmResult.textContent = algorithmSelect.options[algorithmSelect.selectedIndex].text;

                // 设置处理时间
                processingTime.textContent = duration.toFixed(2);

                // 设置下载功能
                downloadBtn.onclick = function() {
                    const link = document.createElement('a');
                    link.href = `/results/${data.enhanced}`;
                    link.download = data.enhanced;
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                };
            })
            .catch(error => {
                console.error('处理失败:', error);
                processingSection.style.display = 'none';
                alert('图像处理失败: ' + error.message);
            });
        };
        reader.readAsDataURL(file);
    });

    // 模拟进度条
    function simulateProgress() {
        let width = 0;
        const interval = setInterval(() => {
            if (width >= 80) { // 只模拟到80%，剩余20%等待实际处理完成
                clearInterval(interval);
            } else {
                width += Math.random() * 10;
                if (width > 80) width = 80;
                progressBar.style.width = width + '%';
            }
        }, 150);
    }

    // 根据算法选择更新预计耗时
    algorithmSelect.addEventListener('change', function() {
        if (this.value === 'lime') {
            timeEstimate.textContent = '40-45秒';
        } else {
            timeEstimate.textContent = '10-15秒';
        }
    });
});
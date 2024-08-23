from flask import Flask, request, render_template, url_for
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os
from werkzeug.utils import secure_filename
import torch.nn as nn
# 定义模型
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 加载模型权重
model = ConvNet()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# 定义转换操作
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Flask 应用
app = Flask(__name__, static_folder='static', template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # 保存上传的文件
            filename = secure_filename(file.filename)  # 使用安全的文件名
            file_path = os.path.join(app.static_folder, filename)
            file.save(file_path)

            # 加载并预处理图像
            image = Image.open(file_path)
            # 调整图片尺寸
            resized_image = image.resize((200, 200))  # 调整图片尺寸为 200x200 像素
            resized_image_path = os.path.join(app.static_folder, f'resized_{filename}')
            resized_image.save(resized_image_path)

            image = transform(image).unsqueeze(0)

            # 使用模型进行分类
            with torch.no_grad():
                output = model(image)
                _, predicted = torch.max(output.data, 1)
                class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
                prediction = class_names[predicted[0]]

                # 渲染结果页面
                return render_template('index.html', prediction=prediction, image_url=url_for('static', filename=f'resized_{filename}'))

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

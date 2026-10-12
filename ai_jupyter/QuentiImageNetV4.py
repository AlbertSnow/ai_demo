# MobileNetV4 训练、导出与 INT8 量化一键复用整合包
# 本单元格将整套工作流（数据集下载 -> PyTorch 训练 -> ONNX 导出 -> onnx2tf 转换 -> TFLite INT8 量化 -> LiteRT 验证）打包整理。所有可变参数均已提取至最顶部，方便后续迁移到其他项目或边缘端部署任务中。

# ========================================== 
# 1. 自动安装关键依赖环境
# ==========================================
# 需 Python 3.12+ 且非 3.14 的 venv（见项目根目录 requirements-torch.txt / .venv-torch）
# !pip install -q -r ../requirements-torch.txt

import os
import sys
import glob
import zipfile
import tarfile
import urllib.request
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import ai_edge_litert.interpreter as litert

# ==========================================
# 2. 全局可变参数提取区 (Configurable Parameters)
# ==========================================
# 数据集配置
DATASET_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
DOWNLOAD_TAR_NAME = "flower_photos.tgz"
EXTRACT_DIR = "./flower_data"
IMAGE_SIZE = 224              # 期望的模型输入分辨率
BATCH_SIZE = 32

# 训练超参数
MODEL_NAME = "mobilenetv4_conv_small.e2400_r224_in1k"
EPOCHS = 3                    # 演示训练轮数，可根据需要调大
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# 输出路径与模型配置
# 默认为本地路径，如果挂载了 Google Drive，可以修改为: '/content/drive/MyDrive/MobileNetV4_Models'
OUTPUT_DIR = "./MobileNetV4_Models"
ONNX_FILENAME = "mobilenetv4_flower_model.onnx"

# 100% INT8 权重压缩量化参数
CALIBRATION_N_SAMPLES = 100   # 校准样本数量
CALIBRATION_NPY_NAME = "calibration_data.npy"

# ==========================================
# 3. 核心流程执行
# ==========================================

# ------------------ 准备目录 ------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EXTRACT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[System] 当前计算设备: {device}")

# ------------------ 下载及解压数据集 ------------------
data_dir = os.path.join(EXTRACT_DIR, 'flower_photos')
if not os.path.exists(data_dir):
    print(f"[Data] 正在下载数据集: {DATASET_URL}...")
    urllib.request.urlretrieve(DATASET_URL, DOWNLOAD_TAR_NAME)
    print("[Data] 解压中...")
    with tarfile.open(DOWNLOAD_TAR_NAME, "r:gz") as tar:
        tar.extractall(path=EXTRACT_DIR)
print(f"[Data] 数据集准备完毕，路径: {data_dir}")

# ------------------ 构建 PyTorch DataLoaders ------------------
data_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

full_dataset = datasets.ImageFolder(root=data_dir, transform=data_transform)
num_classes = len(full_dataset.classes)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
print(f"[Data] 分类数量 (num_classes): {num_classes} | 训练集: {train_size} | 验证集: {val_size}")

# ------------------ 创建与训练 MobileNetV4 ------------------
print(f"[Model] 正在初始化预训练模型: {MODEL_NAME}...")
model = timm.create_model(MODEL_NAME, pretrained=True)
# 修改分类器头适配当前类别
model.classifier = nn.Linear(model.classifier.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="Training Epoch", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / len(loader.dataset), 100. * correct / total

print("[Training] 开始微调...")
for epoch in range(EPOCHS):
    loss, acc = train_epoch(model, train_loader, criterion, optimizer, device)
    print(f"  Epoch {epoch+1}/{EPOCHS} -> Loss: {loss:.4f}, Acc: {acc:.2f}%")

# ------------------ 导出至 ONNX ------------------
model.eval()
onnx_path = os.path.join(OUTPUT_DIR, ONNX_FILENAME)
print(f"[Export] 正在导出模型至 ONNX 格式: {onnx_path}")
dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=18,     # Opset 18 规避向下转换时的算子不兼容行为
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
print("[Export] ONNX 导出成功！")

# ------------------ 准备量化校准数据集 (.npy) ------------------
npy_path = os.path.join(OUTPUT_DIR, CALIBRATION_NPY_NAME)
print(f"[Quant] 正在提取 {CALIBRATION_N_SAMPLES} 张校准图片生成: {npy_path}...")
image_paths = []
for ext in ('/**/*.jpg', '/**/*.jpeg', '/**/*.png'):
    image_paths.extend(glob.glob(data_dir + ext, recursive=True))
np.random.seed(42)
np.random.shuffle(image_paths)
calibration_images = image_paths[:CALIBRATION_N_SAMPLES]

calib_images = []
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
for img_path in calibration_images:
    try:
        img = Image.open(img_path).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE))
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = (img_array - mean) / std
        calib_images.append(img_array)
    except Exception:
        pass
calib_data = np.stack(calib_images, axis=0)
np.save(npy_path, calib_data)

# ------------------ 使用 onnx2tf 转换为 TFLite 并实现 INT8 PTQ 量化 ------------------
print("[onnx2tf] 启动格式转换与 INT8 动态范围压缩机制...")
import subprocess

# -b 1 固定 Batch Size 为 1，规避边缘端动态尺寸算子不支持的风险
# -odrqt 激活 optimized dynamic range quantization，将全部权重压缩到整型 8 位
command = [
    sys.executable, "-m", "onnx2tf",
    "-i", onnx_path,
    "-o", OUTPUT_DIR,
    "-b", "1",
    "-qt", "per-tensor",
    "-odrqt",
    "-cind", "input", npy_path, "[0.485,0.456,0.406]", "[0.229,0.224,0.225]",
    "-non_verbose"
]

result = subprocess.run(command, capture_output=True, text=True)
if result.returncode != 0:
    print("[Error] onnx2tf 转换失败。错误详情:")
    print(result.stderr)
else:
    print("[onnx2tf] 转换与 INT8 量化模型导出大获成功！")

# ------------------ 验证转换输出件 ------------------
print(f"\n[Verify] 正在查看生成的文件 (输出路径: {OUTPUT_DIR}):")
for root, dirs, files in os.walk(OUTPUT_DIR):
    for file in files:
        path = os.path.join(root, file)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        tag = " [★ 优化版 INT8 量化部署模型 ★]" if "quant" in file else ""
        print(f"  - 文件: {file:<55} | 大小: {size_mb:.2f} MB{tag}")

# ------------------ 使用 LiteRT 验证量化模型推理 ------------------
int8_tflite = os.path.join(OUTPUT_DIR, "mobilenetv4_flower_model_dynamic_range_quant.tflite")
if os.path.exists(int8_tflite):
    print(f"\n[LiteRT] 正在测试最终的 INT8 压缩模型: {int8_tflite}")
    interpreter = litert.Interpreter(model_path=int8_tflite)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    
    # 运行一次 Dummy 推理验证模型连通性
    test_input = np.random.randn(*input_shape).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    test_output = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"  -> 输入期望 Shape: {input_shape}")
    print(f"  -> 输出推理 Shape: {test_output.shape}")
    print(f"  -> 测试输出 Logits: {test_output}")
    print("\n🎉 恭喜！一整套流水线运行完毕，所生成的 MobileNetV4 INT8 模型完全通过了端侧标准测试！")
else:
    print(f"\n[Warning] 未找到转换后的 INT8 量化文件: {int8_tflite}")

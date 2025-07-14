# 无监督学习图片分类

这个项目实现了对USTC-TFC2016_MFR数据集的无监督学习分类，包含20个类别的图片数据。

## 数据集

- **数据集路径**: `YaTC_datasets/USTC-TFC2016_MFR/train/`
- **类别数量**: 20个类别
- **图片格式**: PNG格式
- **类别名称**: Outlook, Tinba, Virut, Geodo, MySQL, Skype, WorldOfWarcraft, BitTorrent, Gmail, Htbot, Miuref, Neris, Nsis-ay, Weibo, FTP, Cridex, Facetime, SMB, Shifu, Zeus

## 文件说明

### 主要代码文件

1. **`unsupervised_image_classification.py`** - 完整版本的无监督学习代码
   - 包含自编码器训练
   - 预训练模型特征提取
   - 详细的聚类分析和可视化
   - 适合深入研究和完整实验

2. **`quick_unsupervised_demo.py`** - 快速演示版本
   - 简化的特征提取方法
   - 快速测试和演示
   - 适合快速验证效果

3. **`requirements.txt`** - 依赖包列表

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 快速演示（推荐先运行）

```bash
python quick_unsupervised_demo.py
```

这个版本会：
- 每个类别加载50张图片（总共1000张）
- 使用两种特征提取方法：
  1. 简单特征（直方图、统计特征、边缘特征）
  2. 预训练ResNet18特征
- 使用K-means进行聚类
- 生成可视化结果和性能评估

### 完整版本

```bash
python unsupervised_image_classification.py
```

这个版本会：
- 每个类别加载100张图片（总共2000张）
- 包含自编码器训练
- 更详细的分析和可视化
- 保存完整结果

## 输出结果

### 性能指标

- **Silhouette Score**: 衡量聚类质量（-1到1，越高越好）
- **Adjusted Rand Index (ARI)**: 衡量聚类与真实标签的一致性（-1到1，越高越好）
- **Cluster Purity**: 每个聚类中主导类别的比例

### 可视化文件

- `simple_features_clustering.png` - 简单特征聚类结果
- `pretrained_features_clustering.png` - 预训练特征聚类结果
- `cluster_distribution.png` - 聚类分布热力图

### 数据文件

- `quick_demo_results.npy` - 快速演示结果
- `unsupervised_results.npy` - 完整版本结果

## 特征提取方法

### 1. 简单特征
- **直方图特征**: 16维灰度直方图
- **统计特征**: 均值、标准差、偏度
- **边缘特征**: Canny边缘检测密度
- **总维度**: 20维

### 2. 预训练特征
- **模型**: ResNet18（去除最后的分类层）
- **预处理**: 224x224尺寸，ImageNet标准化
- **特征维度**: 512维

### 3. 自编码器特征（完整版本）
- **输入**: 28x28灰度图像
- **编码器**: 784 → 128 → 64
- **潜在特征维度**: 64维

## 聚类方法

- **算法**: K-means
- **聚类数**: 20（与真实类别数相同）
- **初始化**: 10次随机初始化取最佳结果

## 评估方法

### 无监督评估
- **Silhouette Score**: 衡量聚类的紧密度和分离度

### 有监督评估（已知真实标签）
- **Adjusted Rand Index**: 衡量聚类结果与真实标签的一致性
- **Cluster Purity**: 每个聚类中主导类别的比例

## 预期结果

- **预训练特征**通常表现最好，因为ResNet18在ImageNet上预训练，能够提取更丰富的特征
- **简单特征**计算快速，但性能可能较低
- **自编码器特征**在完整版本中提供中等性能

## 自定义配置

可以在代码中修改以下参数：

```python
# 数据配置
max_samples_per_class = 50  # 每类样本数
n_clusters = 20  # 聚类数

# 特征提取配置
input_dim = 784  # 自编码器输入维度
hidden_dim = 128  # 自编码器隐藏层维度
latent_dim = 64  # 自编码器潜在特征维度

# 训练配置
epochs = 50  # 自编码器训练轮数
```

## 注意事项

1. 确保数据集路径正确
2. 首次运行会下载预训练模型（需要网络连接）
3. 完整版本运行时间较长，建议先运行快速演示版本
4. 可视化需要图形界面或保存为文件查看 
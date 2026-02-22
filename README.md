# CCD LAB Project - Recognition

视频动态区域识别与分析工具，支持视频帧差分析、平移检测和运动方向分析。

## 功能版本

| 版本 | 状态 | 描述 |
|------|------|------|
| **V1.0.0** | 稳定 | 支持基本识别，要求有固定元素（两帧做差） |
| **V2.0.0** | 稳定 | 仅支持动态识别（检测平移区域+边界线） |
| **V2.0.1** | 开发中 | 自动辨别整合版（自动识别运动区域和方向） |

## 功能特性

### V1.0.0 - 基本识别模式
- 两帧做差取绝对值
- 适用于有固定元素的视频验证码识别
- 快速检测帧间差异

### V2.0.0 - 动态识别模式
- 三帧检测平移区域
- 自动检测马赛克方块大小
- 绘制运动区域边界线
- 生成叠加可视化图像

### V2.0.1 - 自动辨别整合模式 (开发中)
- 整合平移检测和运动方向分析
- 自动识别主要运动状态
- 8方向运动检测（右、左、下、上、右下、左上、左下、右上）

## 环境要求

- Python 3.7+
- OpenCV
- NumPy
- SciPy

## 安装

```bash
# 克隆仓库
git clone https://github.com/your-username/CCD-LAB-RECOGNITION.git
cd CCD-LAB-RECOGNITION

# 安装依赖
pip install opencv-python numpy scipy
```

## 使用方法

运行主程序：

```bash
python PythonApplication1.py
```

按提示选择版本和输入视频路径：

```
CCD LAB Project - Recognition
==================================================
请输入数字选择对应版本：
1. V1.0.0（支持基本识别，要求有固定元素）
2. V2.0.0（仅支持动态识别）
3. V2.0.1（开发中）
==================================================
请输入版本编号（1-3）：
```

## 输出文件

### V1.0.0 输出
- `frame_difference.jpg` - 帧差异图像

### V2.0.0 输出
- `motion_detection_block{size}.jpg` - 平移检测结果（黑色为平移区域）
- `boundary_block{size}.jpg` - 边界线标注图像
- `overlay_boundary_block{size}.jpg` - 叠加可视化图像

### V2.0.1 输出
- `{prefix}_motion.jpg` - 平移检测结果
- `{prefix}_boundary.jpg` - 边界标注
- `{prefix}_overlay.jpg` - 叠加可视化
- `{prefix}_direction_color.jpg` - 方向颜色图
- `{prefix}_direction_boundary.jpg` - 方向边界图
- `{prefix}_direction_mask.jpg` - 方向掩码图

## 运动方向颜色说明

| 颜色 | 方向 |
|------|------|
| 灰色 | 静止 |
| 绿色 | 右移 |
| 蓝色 | 左移 |
| 红色 | 下移 |
| 青色 | 上移 |
| 橙色 | 右下 |
| 天蓝 | 左上 |
| 紫色 | 左下 |
| 黄色 | 右上 |

## 项目结构

```
CCD-LAB-RECOGNITION/
├── PythonApplication1.py    # 主程序
├── analysis.py              # 视频噪声分析模块
├── README.md                # 项目说明
└── *.jpg                    # 输出图像（运行后生成）
```

## 技术细节

### 马赛克检测
- 使用自相关分析、局部方差分析、功率谱分析和梯度分析
- 自动检测视频中的马赛克方块大小
- 支持手动指定或自动检测

### 运动检测
- 基于三帧差分的平移区域检测
- 自适应阈值算法
- 形态学操作平滑结果

### 方向分析
- 相位相关法计算运动向量
- 8方向分类（22.5度为边界）
- 统计主要运动状态

## License

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request。

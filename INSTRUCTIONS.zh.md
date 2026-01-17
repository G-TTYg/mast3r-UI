
# MASt3R 增强版UI 中文使用指南

欢迎使用 MASt3R 增强版UI！这是一个基于 Gradio 构建的图形化界面，旨在让您更轻松、更直观地使用 MASt3R 的强大功能。

## 目录
1. [环境准备](#1-环境准备)
2. [模型下载](#2-模型下载)
3. [启动UI界面](#3-启动ui界面)
4. [界面使用说明](#4-界面使用说明)
   - [文件上传](#文件上传)
   - [参数调整](#参数调整)
   - [运行与查看结果](#运行与查看结果)
5. [Windows 用户特别说明](#5-windows-用户特别说明)

---

### 1. 环境准备

我们强烈建议使用 Conda 来创建一个独立的 Python 环境，以避免与您系统中的其他库产生冲突。

如果您尚未安装 Conda，请先从 [Anaconda 官网](https://www.anaconda.com/products/distribution) 下载并安装。

**步骤如下:**

1.  **创建 Conda 环境**:
    打开您的终端（在 Windows 上是 `Anaconda Prompt`），然后运行以下命令来创建一个名为 `mast3r` 的新环境。我们推荐使用 Python 3.11。

    ```bash
    conda create -n mast3r python=3.11
    ```

2.  **激活 Conda 环境**:
    创建成功后，使用以下命令激活新环境。在后续所有操作前，请确保您已经激活了此环境。

    ```bash
    conda activate mast3r
    ```

3.  **安装依赖**:
    在新环境中，执行以下两条命令来安装所有必需的依赖项。

    ```bash
    # 安装主要的依赖
    pip install -r requirements.txt

    # 安装 dust3r 模块的依赖
    pip install -r dust3r/requirements.txt
    ```

---

### 2. 模型下载

本程序需要预训练的模型文件才能运行。

- **自动下载**：
  程序默认会从 Hugging Face Hub 自动下载 `MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric` 模型。首次运行时，请确保您的网络连接畅通，并耐心等待下载完成。

- **手动下载**：
  如果自动下载速度很慢或失败，您也可以从 `README.md` 文件中找到模型的手动下载链接。下载后，请将模型文件（例如 `MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth`）放置在项目根目录下的 `checkpoints/` 文件夹中（如果文件夹不存在，请手动创建）。

  手动下载后，您需要修改启动命令，使用 `--weights` 参数来指定本地模型文件的路径，例如：
  `python app.py --weights checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth`

---

### 3. 启动UI界面

完成环境和模型的准备后，您就可以启动UI界面了。

在终端中，执行以下命令：

```bash
# 如果您有NVIDIA显卡并已正确安装CUDA驱动
python app.py --model_name MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric

# 如果您没有NVIDIA显卡，或者希望使用CPU运行（速度会慢很多）
python app.py --model_name MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric --device cpu
```

成功启动后，您会在终端看到类似以下的输出：
`Running on local URL:  http://127.0.0.1:7860`

请复制此 URL 并在您的浏览器中打开，即可看到程序的UI界面。

---

### 4. 界面使用说明

#### 文件上传
在左侧的“上传图像文件”区域，您可以点击或拖拽多个图像文件到此区域进行上传。这是进行3D重建的基础。

#### 参数调整
在文件上传区域的下方，有三个可折叠的参数区域，您可以根据需求调整：

- **优化参数 (Optimization Parameters)**:
  这里包含了控制模型优化过程的核心参数，例如粗略对齐和精细对齐的学习率（LR）与迭代次数。调整这些参数会影响重建的速度和精度。

- **场景图参数 (Scene Graph Parameters)**:
  您可以选择不同的策略来决定如何匹配上传的图像对。`complete`（完整连接）模式会尝试匹配所有可能的图像对，效果最好但最耗时；而 `swin`（滑动窗口）等模式则更适用于有序的图像序列（如视频帧）。

- **可视化参数 (Visualization Parameters)**:
  这些参数控制最终3D模型的可视化效果。您可以调整 `min_conf_thr` 来过滤掉低置信度的点，或调整 `cam_size` 来改变相机视锥在预览中的大小。勾选 `As pointcloud` 可以将结果显示为点云而不是网格。

#### 运行与查看结果
所有参数调整完毕后，点击蓝色的 **"Run"** 按钮开始重建。处理过程可能需要一些时间，具体取决于您的图像数量、分辨率和机器性能。

完成后，右侧的3D模型预览窗口将显示重建结果。您可以用鼠标拖动来旋转、缩放和平移模型。

---

### 5. Windows 用户特别说明

为了方便 Windows 用户，我们提供了一个一键启动脚本 `start_ui.bat`。

您只需双击运行此脚本，它会自动为您执行使用 CPU 的启动命令。一个命令行窗口会保持开启状态，您可以从中看到程序的运行日志和访问 URL。

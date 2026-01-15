# QFM: Quantum Flow Matching (MVP)

这是一个基于 Flow Matching 和 DiT (Diffusion Transformer) 架构的极简文生图模型实现。
项目集成了 MoE (Mixture of Experts) 机制，支持 Qwen-1.5B 作为文本编码器，SDXL VAE 作为潜在空间编解码器。

## 📂 项目架构 (Project Structure)

```text
QFM/
├── main.py                    # [入口] 所有的命令（预处理、训练、推理）都从这里启动
├── pyproject.toml             # [配置] 项目依赖与包管理
├── .gitignore                 # [配置] 告诉 Git 忽略哪些垃圾文件
├── .pre-commit-config.yaml    # [规范] 代码自动检查工具配置
│
├── src/
│   └── qfm/                   # [核心代码库]
│       ├── config.py          # ⚙️ 全局配置中心 (路径、超参数、设备自动检测)
│       ├── model_moe.py       # 🧠 DiT + MoE 模型定义
│       ├── dataset.py         # 💾 数据加载逻辑
│       ├── utils.py           # 🛠️ ODE 采样器 (Euler) 等工具
│       └── engine/            # 🚀 业务引擎
│           ├── preprocess.py  # 数据预处理脚本
│           ├── trainer.py     # 训练循环逻辑
│           └── inferencer.py  # 推理生成逻辑
│
├── data/                      # [数据区] (Git 会自动忽略此文件夹内容)
│   ├── raw_images/            # 👉 把原始图片 (.jpg/.png) 丢这里
│   ├── processed/             # 预处理生成的 .pt 文件 (自动生成)
│   └── data.jsonl             # 图片索引文件 (需手动创建)
│
└── checkpoints/               # [模型区] (Git 会自动忽略)
    └── flux_moe_ep*.pth       # 训练好的权重文件
```

---

## 🛠️ 第一次安装 (First Time Setup)

无论你是 Mac 还是 Windows，请在拉取代码后执行以下步骤：

### 1. 创建虚拟环境 (推荐)

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Mac / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 2. 安装项目依赖

这是最关键的一步！它会安装所有包，并配置好 `qfm` 的导入路径。

* 如果使用 Mac 或 Windows 的基础 CPU 版本：
```bash
pip install -e .
```
*(注意最后有个点 `.`)*

* 如果使用 Linux/WSL (带GPU加速)：
```bash
pip install -e ".[gpu]"
```
* 我们在项目中内置了一个诊断脚本，用于检测 CUDA、DeepSpeed 和 Flash Attention 是否安装正确。
```bash
python tests/test_env.py
```

*如果看到全绿 `[✅]`，说明环境完美。*

### 3. 安装代码检查钩子 (Pre-commit)

为了防止代码冲突和格式规范，必须安装这个。

```bash
pre-commit install
```

*安装成功后，每次 commit 前它会自动帮你检查代码格式。*

---

## 🚀 如何运行 (Usage)

所有操作都通过 `main.py` 进行。

### 1. 准备数据

在 `data/raw_images/` 放入图片。
在 `QFM` 根目录创建 `data.jsonl`，格式如下：

```json
{"image": "data/raw_images/dog.png", "text": "一只戴着墨镜的柯基犬在沙滩上"}
```

### 2. 预处理 (Preprocess)

这会将图片和文本编码为 Latent 和 Embedding，保存到 `data/processed/`。

```bash
python main.py preprocess
```

### 3. 训练器 (Trainer)

开始训练模型。权重会自动保存到 `checkpoints/`。

```bash
python main.py train
```

> **注意**：Mac 上会自动使用 `float32` (MPS)，Windows (NVIDIA) 上会自动使用 `bfloat16` (CUDA)。

### 4. 推理 (Inference)

加载训练好的权重生成图片。

```bash
python main.py inference --ckpt checkpoints/flux_moe_ep10.pth --prompt "A cyberpunk city"
```

---

## 🤝 协作工作流 (Workflow)

为了保证工作流顺畅，请严格遵守以下流程：

### ☀️ 开始工作前 (Pull)

**一定要先拉取最新代码！**

```bash
git pull
```

*(PyCharm 用户：点击右上角蓝色向下箭头)*

### 🌙 写完代码后 (Commit & Push)

* **添加到暂存区**：
```bash
git add .
```

* **提交 (Commit)**：
```bash
git commit -m "描述你修改了什么"
```


> **注意**：如果 `pre-commit` 报错并自动修复了文件，你需要**再运行一次** `git add .` 和 `git commit`。

* **推送到云端 (Push)**：
```bash
git push
```


*(PyCharm 用户：Ctrl/Cmd + K 提交 >> Ctrl/Cmd + Shift + K 推送)*

---

## ⚠️ 常见问题 (FAQ)

**Q: 预处理或训练时报错 `NaN` 或 `Loss爆炸`？**

* **Mac 用户**：这是 MPS 的精度问题。代码已强制设为 `float32`，请勿手动改为 `float16`。
* **Windows 用户**：请确保 `config.py` 中返回的是 `bfloat16`。

**Q: 推送时提示 `Pre-commit failed`？**

* 这是正常的。工具帮你自动格式化了代码。请重新 `add` 并 `commit` 即可。

**Q: 找不到 `qfm` 模块？**

* 请确保你执行了 `pip install -e .`。

---

## 💻 开发环境避坑指南 (WSL2 + PyCharm)

如果你在 Windows 上使用 WSL2 + PyCharm 开发，请务必阅读以下常见问题：

### Q1: DeepSpeed 报错 `CUDA_HOME does not exist`？

PyCharm 直接运行时可能无法加载 `.bashrc` 环境变量。

* **方法**：运行 `python tests/test_env.py`，脚本内置了自动修复补丁。或者在 `.bashrc` 中强制写入：
```bash
export CUDA_HOME=/你的环境路径/
export PATH=$CUDA_HOME/bin:$PATH
```



### Q2: Git 提交时报错 `pre-commit not found`？

这是因为 PyCharm 使用了 Windows 版 Git，找不到 WSL 里的环境。

* **方法 A (推荐)**：在 PyCharm 底部的 Terminal `(qfm)` 中使用命令行提交：
```bash
git commit -m "..."
```
* **方法 B (彻底修复)**：在 PyCharm 设置中，将 Git 可执行文件路径改为 WSL 路径：
```bash
\\wsl$\Ubuntu-22.04\usr\bin\git
```

### Q3: 报错 `detected dubious ownership`？

这是 Windows 访问 Linux 文件权限问题。

* 在 Windows PowerShell 中运行：
```bash
git config --global --add safe.directory '*'
```

---
## AI 对话提示词

```
# Role Setup
你现在是我的高级 AI 架构师和 Python 编程搭档。我正在开发一个名为 **QFM (Quantum Flow Matching)** 的文生图 MVP 项目。
项目目前已经跑通了预处理、训练和推理的最小闭环。

请记住以下项目上下文、技术栈和文件结构，并在后续对话中严格遵守这些规范。

## 1. 项目概况
* **核心架构**: Flow Matching (Rectified Flow) + DiT (Diffusion Transformer) + MoE (Mixture of Experts).
* **组件**:
    * VAE: `madebyollin/sdxl-vae-fp16-fix` (SDXL VAE)
    * Text Encoder 1: `openai/clip-vit-large-patch14`
    * Text Encoder 2: `Qwen/Qwen2.5-1.5B-Instruct`
* **开发环境**:
    * **Mac (Dev)**: 使用 MPS 加速，强制 `float32` (避免 LayerNorm 溢出)，DataLoader `num_workers=0`。
    * **Windows (Train)**: 使用 CUDA (3090 Ti)，强制 `bfloat16` (防止梯度爆炸)。

## 2. 核心文件结构 (Project Structure)
项目遵循 `src` 布局，包名为 `qfm`，通过 `pip install -e .` 安装。

QFM/
├── main.py                    # [总入口] 调度 preprocess/train/inference
├── pyproject.toml             # [配置] 依赖管理 & Ruff 配置
├── .pre-commit-config.yaml    # [规范] Ruff 代码检查
├── src/
│   └── qfm/                   # [源代码包]
│       ├── config.py          # [核心配置] 路径、超参数、设备自动判断(get_device/get_dtype)
│       ├── model_moe.py       # [模型] DiT Block + MoE Layer
│       ├── dataset.py         # [数据] LatentDataset
│       ├── utils.py           # [工具] Euler ODE Solver
│       ├── core/
│       │   └── logger.py      # [日志] 统一日志模块
│       └── engine/            # [业务引擎]
│           ├── preprocess.py  # 预处理 (生成 .pt)
│           ├── trainer.py     # 训练循环
│           └── inferencer.py  # 推理脚本
└── data/                      # 存放 raw_images 和 data.jsonl

## 3. 已建立的工程规范 (Strict Rules)
1.  **配置分离**: 禁止硬编码路径或参数。所有超参数（Batch size, LR, Model Dim）必须从 `qfm.config.cfg` 读取。
2.  **导入规范**: 必须使用绝对导入 `from qfm.xxx import yyy`，禁止相对导入。
3.  **设备兼容**: 涉及 `device` 和 `dtype` 时，必须调用 `cfg.get_device()` 和 `cfg.get_dtype(device)`，严禁写死 "cuda"。
4.  **预处理**: 生成的 Latent 必须在 Encode 时乘以 `0.13025`，Decode 时除以该值。

## 4. 当前任务
我目前已经完成了基础架构搭建，可以成功运行 `python main.py train` 和 `inference`。
接下来如果你准备好了，请回复 "QFM Environment Loaded"，然后等待我的具体指令。
```

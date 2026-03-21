<div align="center">
  <a href="readme.md">English</a> |
  <a href="docs/readme.ja.md">日本語</a> |
  <a href="docs/readme.zh-CN.md">简体中文</a>
</div>

---

# QFM: Quantum Flow Matching (MVP)

This is a minimalist Text-to-Image model implementation based on Flow Matching and DiT (Diffusion Transformer) architectures. The project integrates the MoE (Mixture of Experts) mechanism, supporting Qwen-1.5B as the text encoder and SDXL VAE as the latent space codec.

## 📂 Project Structure

```text
QFM/
├── main.py                    # [Entry Point] All commands (preprocess, train, inference) start here
├── pyproject.toml             # [Config] Project dependencies and package management
├── .gitignore                 # [Config] Tells Git which files to ignore
├── .pre-commit-config.yaml    # [Standard] Automated code formatting/linting hooks
│
├── src/
│   └── qfm/                   # [Core Library]
│       ├── config.py          # ⚙️ Global configuration (Paths, Hyperparameters, Auto-device detection)
│       ├── model_moe.py       # 🧠 DiT + MoE model definition
│       ├── dataset.py         # 💾 Data loading logic
│       ├── utils.py           # 🛠️ ODE Solvers (Euler) and other tools
│       └── engine/            # 🚀 Business Logic Engine
│           ├── preprocess.py  # Data preprocessing scripts
│           ├── trainer.py     # Training loop logic
│           └── inference.py   # Inference and generation logic
│
├── data/                      # [Data Area] (Ignored by Git)
│   ├── raw_images/            # 👉 Drop raw images (.jpg/.png) here
│   ├── processed/             # Preprocessed .pt files (Auto-generated)
│   └── data.jsonl             # Image index file (Requires manual creation)
│
└── checkpoints/               # [Models] (Ignored by Git)
    └── flux_moe_ep*.pth       # Trained weight files
```

---

## 🛠️ First Time Setup

Whether you are on Mac or Windows, please follow these steps after cloning the repository:

### 1. Create a Virtual Environment (Recommended)

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Mac / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Project Dependencies

This is the most crucial step! It installs all required packages and configures the `qfm` import paths.

* For base CPU versions on Mac or Windows:
```bash
pip install -e .
```
*(Note the dot `.` at the end)*

* For Linux/WSL (with GPU acceleration):
```bash
pip install -e ".[gpu]"
```
* We have included a built-in diagnostic script to verify CUDA, DeepSpeed, and Flash Attention installations:
```bash
python tests/test_lib.py
```

*If you see all green `[✅]`, your environment is perfectly configured.*

### 3. Install Pre-commit Hooks

This is mandatory to prevent code conflicts and maintain formatting standards.

```bash
pre-commit install
```

*Once installed, it will automatically check your code formatting before every commit.*

---

## 🚀 Usage

All operations are executed through `main.py`.

### 1. Data Preparation

Place your images in `data/raw_images/`.
Create a `data.jsonl` file in the root `QFM` directory with the following format:

```json
{"image": "data/raw_images/dog.png", "text": "A corgi wearing sunglasses on the beach"}
```

### 2. Preprocess

This encodes the images and text into Latents and Embeddings, saving them to `data/processed/`.

```bash
python main.py preprocess
```

### 3. Trainer

Start training the model. Weights will be automatically saved to `checkpoints/`.

```bash
python main.py train
```

> **Note**: Macs will automatically default to `float32` (MPS), while Windows (NVIDIA) will default to `bfloat16` (CUDA).

### 4. Inference

Load trained weights to generate images.
- Default inference (Auto-loads the latest model + generates 512x512):
```bash
python main.py inference --prompt "A cute cat"
```
- Advanced inference (Generates a 1024x512 widescreen wallpaper):
```bash
python main.py inference --prompt "Wide angle shot of space station" --width 1024 --height 512
```
- Inference with a specific model:
```bash
python main.py inference --ckpt checkpoints/qfm_moe_ep10.pth
```

---
## 🤝 Collaboration Workflow

To ensure a smooth workflow, strictly adhere to the following process:

### ☀️ Before You Start (Pull)

**Always pull the latest code first!**

```bash
git pull
```

*(PyCharm Users: Click the blue downward arrow in the top right corner)*

### 🌙 After Writing Code (Commit & Push)

* **Stage your changes**:
```bash
git add .
```

* **Commit**:
```bash
git commit -m "Describe your changes here"
```

> **Note**: If `pre-commit` throws an error and auto-fixes files, you must **run** `git add .` and `git commit` **again**.

* **Push to the cloud**:
```bash
git push
```

*(PyCharm Users: Ctrl/Cmd + K to commit >> Ctrl/Cmd + Shift + K to push)*

---

## ⚠️ FAQ

**Q: Encountering `NaN` or `Loss Explosion` during preprocessing or training?**

* **Mac Users**: This is an MPS precision issue. The code is forced to `float32`; do not manually change it to `float16`.
* **Windows Users**: Ensure that `config.py` is returning `bfloat16`.

**Q: Prompted with `Pre-commit failed` during a push?**

* This is normal. The tool automatically formatted your code. Simply `add` and `commit` again.

**Q: ModuleNotFoundError for `qfm`?**

* Ensure you have run `pip install -e .`.

---

## 💻 Dev Environment Troubleshooting (WSL2 + PyCharm)

If you are developing on Windows using WSL2 + PyCharm, please read the following:

### Q1: DeepSpeed Error `CUDA_HOME does not exist`?

PyCharm might not load `.bashrc` environment variables when running directly.

* **Solution**: Run `python tests/test_env.py` (contains an auto-fix patch). Alternatively, force write this in your `.bashrc`:
```bash
export CUDA_HOME=/your/env/path/
export PATH=$CUDA_HOME/bin:$PATH
```

### Q2: Git commit throws `pre-commit not found`?

This happens because PyCharm is using the Windows version of Git and cannot find the WSL environment.

* **Solution A (Recommended)**: Use the command line in PyCharm's bottom Terminal `(qfm)`:
```bash
git commit -m "..."
```
* **Solution B (Permanent Fix)**: In PyCharm Settings, change the Git executable path to your WSL path:
```bash
\\wsl$\Ubuntu-22.04\usr\bin\git
```

### Q3: Error `detected dubious ownership`?

This is a Windows permission issue when accessing Linux files.

* Run this in Windows PowerShell:
```bash
git config --global --add safe.directory '*'
```

---
## AI Prompt (For Assistant Context)

```
# Role Setup
You are my Senior AI Architect and Python pair programming partner. I am developing a Text-to-Image MVP project named **QFM (Quantum Flow Matching)**.
The project currently has a working minimum loop for preprocessing, training, and inference.

Please memorize the following project context, tech stack, and file structure, and strictly adhere to these guidelines in our subsequent conversations.

## 1. Project Overview
* **Core Architecture**: Flow Matching (Rectified Flow) + DiT (Diffusion Transformer) + MoE (Mixture of Experts).
* **Components**:
    * VAE: `madebyollin/sdxl-vae-fp16-fix` (SDXL VAE)
    * Text Encoder 1: `openai/clip-vit-large-patch14`
    * Text Encoder 2: `Qwen/Qwen2.5-1.5B-Instruct`
* **Dev Environments**:
    * **Mac (Dev)**: Uses MPS acceleration, forced `float32` (avoids LayerNorm overflow), DataLoader `num_workers=0`.
    * **Windows (Train)**: Uses CUDA (3090 Ti), forced `bfloat16` (prevents gradient explosion).

## 2. Core File Structure
The project follows a `src` layout, the package name is `qfm`, installed via `pip install -e .`.

QFM/
├── main.py                    # [Global Entry] Schedules preprocess/train/inference
├── pyproject.toml             # [Config] Dependency management & Ruff config
├── .pre-commit-config.yaml    # [Standard] Ruff linting
├── src/
│   └── qfm/                   # [Source Code]
│       ├── config.py          # [Core Config] Paths, Hyperparams, Auto-device logic (get_device/get_dtype)
│       ├── model_moe.py       # [Model] DiT Block + MoE Layer
│       ├── dataset.py         # [Data] LatentDataset
│       ├── utils.py           # [Utils] Euler ODE Solver
│       ├── core/
│       │   └── logger.py      # [Logs] Unified logging module
│       └── engine/            # [Engines]
│           ├── preprocess.py  # Preprocessing (generates .pt)
│           ├── trainer.py     # Training loop
│           └── inferencer.py  # Inference script
└── data/                      # Contains raw_images and data.jsonl

## 3. Established Engineering Standards (Strict Rules)
1.  **Config Separation**: No hardcoded paths or parameters. All hyperparameters (Batch size, LR, Model Dim) MUST be read from `qfm.config.cfg`.
2.  **Import Standards**: Must use absolute imports `from qfm.xxx import yyy`. Relative imports are strictly forbidden.
3.  **Device Compatibility**: When dealing with `device` and `dtype`, must call `cfg.get_device()` and `cfg.get_dtype(device)`. Hardcoding "cuda" is forbidden.
4.  **Preprocessing**: Generated Latents MUST be multiplied by `0.13025` during Encode, and divided by this value during Decode.

## 4. Current Task
I have finished the foundational architecture and can successfully run `python main.py train` and `inference`.
Next, if you are ready, please reply "QFM Environment Loaded" and wait for my specific instructions.
```

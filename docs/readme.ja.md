<div align="center">
  <a href="../readme.md">English</a> |
  <a href="readme.ja.md">日本語</a> |
  <a href="readme.zh-CN.md">简体中文</a>
</div>

---

# QFM: Quantum Flow Matching (MVP)

これは、Flow Matching および DiT (Diffusion Transformer) アーキテクチャに基づいた、ミニマルな Text-to-Image（テキストから画像生成）モデルの実装です。プロジェクトには MoE (Mixture of Experts) メカニズムが統合されており、テキストエンコーダーとして Qwen-1.5B、潜在空間 (Latent Space) のコーデックとして SDXL VAE をサポートしています。

## 📂 プロジェクト構成 (Project Structure)

```text
QFM/
├── main.py                    # [エントリーポイント] すべてのコマンド（前処理、学習、推論）の起点
├── pyproject.toml             # [設定] プロジェクトの依存関係とパッケージ管理
├── .gitignore                 # [設定] Git 管理から除外するファイルの指定
├── .pre-commit-config.yaml    # [規約] コード自動フォーマット・検査ツールの設定
│
├── src/
│   └── qfm/                   # [コアライブラリ]
│       ├── config.py          # ⚙️ グローバル設定（パス、ハイパーパラメータ、デバイス自動検出）
│       ├── model_moe.py       # 🧠 DiT + MoE モデル定義
│       ├── dataset.py         # 💾 データロードのロジック
│       ├── utils.py           # 🛠️ ODE サンプラー (Euler) などのツール
│       └── engine/            # 🚀 ビジネスロジックエンジン
│           ├── preprocess.py  # データ前処理スクリプト
│           ├── trainer.py     # 学習ループのロジック
│           └── inference.py   # 推論・画像生成ロジック
│
├── data/                      # [データ領域] (Git では自動的に無視されます)
│   ├── raw_images/            # 👉 元画像 (.jpg/.png) を配置する場所
│   ├── processed/             # 前処理済みの .pt ファイル（自動生成）
│   └── data.jsonl             # 画像インデックスファイル（手動作成が必要）
│
└── checkpoints/               # [モデル領域] (Git では無視されます)
    └── flux_moe_ep*.pth       # 学習済みの重みファイル
```

---

## 🛠️ 初回セットアップ (First Time Setup)

Mac または Windows のいずれを使用している場合でも、コードをクローンした後に以下の手順を実行してください。

### 1. 仮想環境の作成 (推奨)

```bash
# Windows の場合
python -m venv .venv
.venv\Scripts\activate

# Mac / Linux の場合
python3 -m venv .venv
source .venv/bin/activate
```

### 2. プロジェクトの依存関係をインストール

これは最も重要なステップです！必要なすべてのパッケージをインストールし、`qfm` のインポートパスを設定します。

* Mac または Windows のベース CPU バージョンの場合：
```bash
pip install -e .
```
*(最後にドット `.` があることに注意してください)*

* Linux/WSL (GPU アクセラレーション付き) の場合：
```bash
pip install -e ".[gpu]"
```
* プロジェクトには、CUDA、DeepSpeed、Flash Attention が正しくインストールされているかを確認するための診断スクリプトが組み込まれています。
```bash
python tests/test_lib.py
```

*すべてが緑色の `[✅]` で表示されれば、環境は完璧です。*

### 3. Pre-commit フックのインストール

コードの競合を防ぎ、フォーマットを維持するために、これを必ずインストールしてください。

```bash
pre-commit install
```

*インストールが成功すると、コミットするたびにコードフォーマットが自動的にチェックされます。*

---

## 🚀 使用方法 (Usage)

すべての操作は `main.py` を通じて行われます。

### 1. データの準備

画像を `data/raw_images/` に配置します。
`QFM` のルートディレクトリに、以下のフォーマットで `data.jsonl` を作成します：

```json
{"image": "data/raw_images/dog.png", "text": "ビーチでサングラスをかけたコーギー"}
```

### 2. 前処理 (Preprocess)

画像とテキストを Latent と Embedding にエンコードし、`data/processed/` に保存します。

```bash
python main.py preprocess
```

### 3. 学習 (Trainer)

モデルの学習を開始します。重みは自動的に `checkpoints/` に保存されます。

```bash
python main.py train
```

> **注意**: Mac では自動的に `float32` (MPS) が使用され、Windows (NVIDIA) では自動的に `bfloat16` (CUDA) が使用されます。

### 4. 推論 (Inference)

学習済みの重みをロードして画像を生成します。
- デフォルトの推論 (最新のモデルを自動ロード + 512x512 を生成)：
```bash
python main.py inference --prompt "かわいい猫"
```
- 高度な推論 (ワイドスクリーンの壁紙 1024x512 を生成)：
```bash
python main.py inference --prompt "宇宙ステーションの広角撮影" --width 1024 --height 512
```
- 指定したモデルでの推論：
```bash
python main.py inference --ckpt checkpoints/qfm_moe_ep10.pth
```

---
## 🤝 共同作業ワークフロー (Workflow)

スムーズなワークフローを確保するために、以下のプロセスを厳守してください：

### ☀️ 作業開始前 (Pull)

**必ず最初に最新のコードをプルしてください！**

```bash
git pull
```

*(PyCharm ユーザー：右上の青い下向き矢印をクリック)*

### 🌙 コード記述後 (Commit & Push)

* **ステージングエリアに追加**：
```bash
git add .
```

* **コミット (Commit)**：
```bash
git commit -m "変更内容をここに記述"
```

> **注意**: `pre-commit` がエラーをスローし、ファイルを自動修正した場合は、**再度** `git add .` と `git commit` を実行する必要があります。

* **クラウドにプッシュ (Push)**：
```bash
git push
```

*(PyCharm ユーザー：Ctrl/Cmd + K でコミット >> Ctrl/Cmd + Shift + K でプッシュ)*

---

## ⚠️ よくある質問 (FAQ)

**Q: 前処理や学習中に `NaN` または `Loss爆発` のエラーが発生しますか？**

* **Mac ユーザー**: これは MPS の精度に関する問題です。コードは強制的に `float32` に設定されています。手動で `float16` に変更しないでください。
* **Windows ユーザー**: `config.py` が `bfloat16` を返していることを確認してください。

**Q: プッシュ時に `Pre-commit failed` と表示されますか？**

* これは正常です。ツールがコードを自動的にフォーマットしました。再度 `add` して `commit` してください。

**Q: `qfm` モジュールが見つかりませんか？**

* `pip install -e .` を実行したことを確認してください。

---

## 💻 開発環境トラブルシューティング (WSL2 + PyCharm)

Windows 上で WSL2 + PyCharm を使用して開発している場合は、以下を必ずお読みください：

### Q1: DeepSpeed エラー `CUDA_HOME does not exist`？

PyCharm から直接実行する場合、`.bashrc` の環境変数がロードされないことがあります。

* **解決策**: `python tests/test_env.py` を実行してください（自動修正パッチが組み込まれています）。または、`.bashrc` に以下を強制的に記述します：
```bash
export CUDA_HOME=/あなたの環境パス/
export PATH=$CUDA_HOME/bin:$PATH
```

### Q2: Git コミット時に `pre-commit not found` エラーが発生する？

これは、PyCharm が Windows 版の Git を使用しており、WSL 内の環境を見つけられないためです。

* **解決策 A (推奨)**: PyCharm 下部の Terminal `(qfm)` でコマンドラインからコミットします：
```bash
git commit -m "..."
```
* **解決策 B (完全な修正)**: PyCharm の設定で、Git 実行可能ファイルのパスを WSL のパスに変更します：
```bash
\\wsl$\Ubuntu-22.04\usr\bin\git
```

### Q3: `detected dubious ownership` エラーが発生する？

これは、Windows から Linux ファイルにアクセスする際の権限の問題です。

* Windows PowerShell で以下を実行します：
```bash
git config --global --add safe.directory '*'
```

---
## AI 対話プロンプト (Context)

```
# Role Setup
あなたは私のシニア AI アーキテクトであり、Python ペアプログラミングのパートナーです。私は **QFM (Quantum Flow Matching)** という名前の Text-to-Image MVP プロジェクトを開発しています。
プロジェクトは現在、前処理、学習、推論の最小限のループが稼働しています。

以下のプロジェクトのコンテキスト、技術スタック、ファイル構造を記憶し、今後の対話ではこれらのガイドラインを厳守してください。

## 1. プロジェクト概要
* **コアアーキテクチャ**: Flow Matching (Rectified Flow) + DiT (Diffusion Transformer) + MoE (Mixture of Experts).
* **コンポーネント**:
    * VAE: `madebyollin/sdxl-vae-fp16-fix` (SDXL VAE)
    * Text Encoder 1: `openai/clip-vit-large-patch14`
    * Text Encoder 2: `Qwen/Qwen2.5-1.5B-Instruct`
* **開発環境**:
    * **Mac (Dev)**: MPS アクセラレーションを使用、強制 `float32` (LayerNorm のオーバーフローを回避)、DataLoader `num_workers=0`。
    * **Windows (Train)**: CUDA (3090 Ti) を使用、強制 `bfloat16` (勾配爆発を防止)。

## 2. コアファイル構造
プロジェクトは `src` レイアウトに従い、パッケージ名は `qfm` で、`pip install -e .` を介してインストールされます。

QFM/
├── main.py                    # [総合エントリ] preprocess/train/inference をスケジュール
├── pyproject.toml             # [設定] 依存関係管理 & Ruff 設定
├── .pre-commit-config.yaml    # [規約] Ruff コード検査
├── src/
│   └── qfm/                   # [ソースコード]
│       ├── config.py          # [コア設定] パス、ハイパーパラメータ、デバイス自動判定 (get_device/get_dtype)
│       ├── model_moe.py       # [モデル] DiT Block + MoE Layer
│       ├── dataset.py         # [データ] LatentDataset
│       ├── utils.py           # [ツール] Euler ODE Solver
│       ├── core/
│       │   └── logger.py      # [ログ] 統合ログモジュール
│       └── engine/            # [エンジン]
│           ├── preprocess.py  # 前処理 (.pt 生成)
│           ├── trainer.py     # 学習ループ
│           └── inferencer.py  # 推論スクリプト
└── data/                      # raw_images と data.jsonl を格納

## 3. 確立されたエンジニアリング規約 (Strict Rules)
1.  **設定の分離**: パスやパラメータのハードコーディングは禁止。すべてのハイパーパラメータ (Batch size, LR, Model Dim) は `qfm.config.cfg` から読み込むこと。
2.  **インポート規約**: 絶対インポート `from qfm.xxx import yyy` を使用すること。相対インポートは厳禁。
3.  **デバイス互換性**: `device` および `dtype` を扱う際は、`cfg.get_device()` と `cfg.get_dtype(device)` を呼び出すこと。"cuda" のハードコーディングは禁止。
4.  **前処理**: 生成された Latent は、エンコード時に `0.13025` を掛け、デコード時にこの値で割ること。

## 4. 現在のタスク
基礎アーキテクチャの構築が完了し、`python main.py train` および `inference` を正常に実行できます。
次に、準備ができたら「QFM Environment Loaded」と返信し、私の具体的な指示をお待ちください。
```

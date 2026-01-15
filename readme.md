Quantum Flow Matching with MoE

MyFlux_MVP/
├── .git/                      # Git 仓库记录
├── .gitignore                 # [必备] 忽略数据、权重和缓存
├── .editorconfig              # [规范] 统一编辑器缩进和换行
├── .pre-commit-config.yaml    # [规范] 提交前自动代码检查
├── pyproject.toml             # [核心] Ruff/Black 插件的详细配置
├── requirements.txt           # [环境] 项目依赖清单
├── README.md                  # [文档] 项目启动与运行说明
│
├── data/                      # 数据区（不进 Git）
│   ├── raw_images/            # 原始图片
│   ├── processed/             # 预处理后的数据
│   └── data.jsonl             # 图片索引文件
│
├── checkpoints/               # 模型权重区（不进 Git）
│   └── flux_moe_v1.pth
│
├── tests/                     # 测试区
│   └── test_model.py          # 验证模型能否跑通推理
│
├── src/                       # 源代码根目录
│   └── my_flux/               # 项目包名（方便 import）
│       ├── __init__.py        # 使其成为一个可导入的包
│       ├── config.py          # 唯一配置中心（超参数、路径）
│       ├── dataset.py         # 数据加载逻辑
│       ├── model_moe.py       # 模型定义（MoE 结构）
│       ├── utils.py           # 采样、ODE 等工具函数
│       │
│       ├── core/              # [进阶] 存放基础逻辑
│       │   └── logger.py      # 统一日志打印格式
│       │
│       └── engine/            # [核心业务]
│           ├── preprocess.py  # 预处理执行脚本
│           ├── trainer.py     # 训练循环逻辑
│           └── inferencer.py  # 推理/生成逻辑
│
└── main.py                    # [总入口] 用户通过此文件启动训练或推理

安装 pre-commit 工具：pip install pre-commit
安装 Git 钩子：pre-commit install

预处理：python main.py preprocess

训练：python main.py train

推理：python main.py inference --ckpt checkpoints/xxx.pth --prompt "hello world"

# Search-R1 工作区

这个目录就是工作区根目录，同时也是 git 仓库根。仓库只跟踪 `verl/`（代码）；
数据、实验产物、日志和虚拟环境都不进 git。

## 目录

```
<workspace>/                 # git 仓库根（本目录）
├── verl/                    # 代码：verl@3255f6b1 + Search-R1 示例
│   ├── verl/                # Python 包
│   ├── examples/search_r1/  # Search-R1 脚本与文档
│   └── pyproject.toml
├── data/                    # 数据与模型（忽略）
│   ├── nq_hotpotqa_train/
│   ├── search_r1/{train,test,test_sub}.parquet
│   ├── retrieval/{e5_Flat.index,wiki-18.jsonl}
│   └── models/{Qwen3.5-2B,e5-base-v2}
├── experiments/<EXP>/       # 每次实验（忽略）
├── logs/                    # retriever.log 等（忽略）
└── .venv/  .venv-mig/  .tools/  .cache/   # 环境（忽略）
```

只有 `verl/`、根目录的 `.gitignore` 和本 `README.md` 被跟踪。服务器上把仓库 clone 到
`$SEARCH_R1_HOME` 即可，`data/`、`experiments/`、`logs/`、`.venv/` 放在同级。

## 上游

代码基于 [verl-project/verl](https://github.com/verl-project/verl) `3255f6b1`。
`verl/` 下的上游文件与上游保持一致，Search-R1 相关的新增内容有：

- `verl/examples/search_r1/`：训练脚本、检索服务、打包脚本和文档
- `verl/examples/data_preprocess/search_r1.py`：Search-R1 数据预处理
- `verl/verl/tools/search_tool.py` 与 `verl/verl/utils/reward_score/`：Search-R1 的检索工具和 reward

## 使用

从本目录安装代码并开始训练：

```bash
uv pip install --python .venv/bin/python -e verl --no-deps
bash verl/examples/search_r1/start_retriever.sh
SMOKE=1 bash verl/examples/search_r1/run_qwen3_5_2b_search_r1_grpo.sh
EXP=my-run BG=1 bash verl/examples/search_r1/run_qwen3_5_2b_search_r1_grpo.sh
```

完整的预处理、训练、续训和导出流程见
[`verl/examples/search_r1/README.md`](verl/examples/search_r1/README.md)。

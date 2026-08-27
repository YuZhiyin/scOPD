# CellPuzzles × Qwen3-8B × GT-privileged SDPO

该目录是一个独立的细胞注释模块，不修改 SDPO trainer 的核心实现。它包含：

- 官方 `ncbi/CellPuzzles` train/test 下载与统一处理；
- `cell-o1/unseen_data` 的只读转换；
- Qwen3-8B `enable_thinking=False` 的 vLLM 基线评测；
- 正确性、格式和一对一候选约束指标；
- 把训练集 ground truth 作为教师特权反馈的 SDPO reward；
- 可选的 answer-format LoRA SFT、模型合并和 rjob 脚本。
- 官方 o1 蒸馏推理轨迹的 Qwen3 thinking-mode LoRA SFT。
- Cell-o1 对齐的 thinking-mode GRPO，以及按 rollout 路由的 GT-SDPO。

所有命令默认从 `/mnt/shared-storage-user/yuzhiyin/SDPO` 执行。

## 0. 环境

集群镜像应安装本仓库依赖（尤其是 `verl`、`transformers==4.57.1`、
`vllm`、`pyarrow`、`peft` 和 `flash-attn`）：

```bash
cd /mnt/shared-storage-user/yuzhiyin/SDPO
python3 -m pip install -e .
```

若现有 rjob 镜像已经能运行 SDPO，则不需要重复安装。

## 1. 下载和处理数据

```bash
cd /mnt/shared-storage-user/yuzhiyin/SDPO
bash cell_annotation/scripts/download_and_prepare.sh
```

等价的显式命令：

```bash
export PYTHONPATH=/mnt/shared-storage-user/yuzhiyin/SDPO
python3 -m cell_annotation.prepare_data \
  --output-dir /mnt/shared-storage-user/yuzhiyin/SDPO/cell_annotation/data \
  --unseen-dir /mnt/shared-storage-user/yuzhiyin/cell-o1/unseen_data
```

下载来源是 Hugging Face `ncbi/CellPuzzles` 的官方 train/test parquet。
脚本会验证官方记录数并生成：

```text
cell_annotation/data/
├── raw/
│   ├── cellpuzzles_train.parquet
│   └── cellpuzzles_test.parquet
└── processed/
    ├── train.json                 # 推理/SFT 通用格式，6912
    ├── train.parquet              # 完整 verl 格式，6912
    ├── train_fit.parquet          # 默认 SFT/SDPO fit，6566
    ├── train_dev.parquet          # train 内部 validation，346
    ├── test.json                  # 官方 test，1095
    ├── test.parquet
    ├── test_clean.json            # 无 train 单细胞 top-gene 精确重合，604
    ├── test_clean.parquet
    ├── unseen_all.json            # 四个 unseen 合并，539
    ├── unseen_all.parquet
    ├── unseen/*.json|parquet
    ├── sft_train.jsonl            # train 内固定种子 95%
    ├── sft_dev.jsonl              # train 内固定种子 5%
    └── manifest.json              # URL、SHA256、记录数和划分说明
```

训练脚本只读取官方 train 派生的 `train_fit.parquet` 和
`train_dev.parquet`。官方 test 和所有 unseen 文件都不会进入 SFT、SDPO
训练或训练期模型选择。`test_clean` 是额外的抗数据重合报告集：若一个 test batch
中任意单细胞的完整 ranked-gene 列表在 train 中原样出现，则剔除该 batch。

默认处理只把原 system prompt 中“输出 detailed reasoning”的要求改为
non-thinking 的“只输出 `<answer>`”；基因、上下文、候选和标签均不变。这避免
原始 CoT 指令与 Qwen3 `enable_thinking=False` 冲突。若需要复现原始 system
prompt，可重新处理：

```bash
bash cell_annotation/scripts/download_and_prepare.sh \
  --keep-original-system-prompt
```

## 2. 评测 Qwen3-8B non-thinking

提交 2-GPU vLLM 基线任务：

```bash
cd /mnt/shared-storage-user/yuzhiyin/SDPO
bash cell_annotation/cluster/submit_qwen3_8b_eval_rjob.sh
```

只打印 rjob 命令：

```bash
DRY_RUN=1 bash cell_annotation/cluster/submit_qwen3_8b_eval_rjob.sh
```

已经在 GPU 节点时直接运行：

```bash
TP_SIZE=2 \
MODEL_PATH=/mnt/shared-storage-user/ma4tool-shared/hug_ckpts/Qwen3/Qwen3-8B \
bash cell_annotation/cluster/run_qwen3_8b_eval_rjob.sh
```

只评测一个文件的底层命令：

```bash
python3 -m cell_annotation.infer_vllm \
  --model /mnt/shared-storage-user/ma4tool-shared/hug_ckpts/Qwen3/Qwen3-8B \
  --input cell_annotation/data/processed/test.json \
  --output /tmp/qwen3_cellpuzzles_test.json \
  --tensor-parallel-size 2 \
  --temperature 0 \
  --top-p 1

python3 -m cell_annotation.evaluate \
  --predictions /tmp/qwen3_cellpuzzles_test.json \
  --output /tmp/qwen3_cellpuzzles_test_metrics.json
```

`infer_vllm.py` 显式调用：

```python
tokenizer.apply_chat_template(..., enable_thinking=False)
```

因此这是 Qwen3 non-thinking，而不是依靠 prompt 猜测。Qwen3 模板会在
assistant prefix 中预填空的 `<think></think>`，所以生成 continuation 只包含
一个 `<answer>...</answer>` 时视为严格有效；显式
`<think>...</think><answer>...</answer>` 也兼容。

用本地 Qwen3 tokenizer 实测的最大 prompt 长度为：train 3101、官方 test
4335、unseen 3065 tokens。训练上限 4096 不会过滤 train；独立评测使用
`max_model_len=8192`，可覆盖全部 test/unseen。

主要指标：

- `micro_cell_accuracy`：所有 cell 的位置准确率；
- `mean_exact_match`：整批 N 个 cell 全对，忽略外层格式；
- `mean_strict_exact_match`：整批全对且格式严格有效；
- `mean_candidate_set_valid`：数量正确、无重复、恰好使用候选集合；
- `mean_valid_output_format`：语法严格且候选排列合法；
- `mean_strict_format`：外层标签语法有效率。

应分别报告官方 test、test_clean 和四个 unseen；不要只报告合并后的 unseen。

## 2.1 o1 蒸馏推理轨迹 SFT（thinking mode）

官方 `ncbi/CellPuzzles` 的 `reasoning` split 包含 3,912 条由 o1 蒸馏的
expert-like reasoning traces。脚本会验证所有样本均来自官方 train、与 test
零 prompt 重合、包含非空的 `<think>...</think><answer>...</answer>`，且轨迹
答案与 train gold 完全一致。

单独准备数据：

```bash
bash cell_annotation/scripts/prepare_o1_reasoning_sft_data.sh
```

默认生成 3,716 条训练轨迹和 196 条 validation 轨迹：

```text
cell_annotation/data/processed/sft_reasoning_train.jsonl
cell_annotation/data/processed/sft_reasoning_dev.jsonl
cell_annotation/data/processed/reasoning_sft_manifest.json
```

提交 8-GPU Qwen3-8B reasoning SFT：

```bash
EXPERIMENT_NAME=qwen3-8b-o1-reasoning-sft-v1 \
bash cell_annotation/cluster/submit_qwen3_8b_reasoning_sft_rjob.sh
```

训练使用 `target_mode=reasoning`、Qwen3 `enable_thinking=True`，监督完整
assistant 轨迹而非只监督答案。默认 `max_length=6144`、1 epoch、LoRA
`r=64/alpha=128`、学习率 `5e-5`，训练结束后自动合并到
`outputs/cell_annotation/<experiment>/merged_model`。

thinking-mode 评测必须显式开启相同协议，并为推理留出更长输出：

```bash
MODEL_PATH=/path/to/reasoning-sft/merged_model \
ENABLE_THINKING=1 \
MAX_NEW_TOKENS=3072 \
JOB_NAME=qwen3-8b-o1-reasoning-sft-eval \
bash cell_annotation/cluster/submit_qwen3_8b_eval_rjob.sh
```

## 3. Cell-o1 RLVR 与 GT-routed SDPO

两项实验默认从严格复现的 reasoning-SFT 合并模型开始：

```text
outputs/cell_annotation/reasoning-sft-repro-seed0-v2/merged_model
```

训练数据是 `data/rl_thinking/processed/train.parquet` 中完整的 6,912 条
CellPuzzles train；该副本保留原始 reasoning system prompt。Qwen3 chat
template 显式设置 `enable_thinking=true`。测试集只作为 validation，不参与更新。
可用下面的命令从已经下载的 raw parquet 重新生成：

```bash
bash cell_annotation/scripts/prepare_rl_thinking_data.sh
```

GRPO 对齐 Cell-o1 的关键设置：train batch 64、每个 prompt 采样 5 条
rollout、response 上限 3000、actor LR 1e-6、PPO mini-batch 64、low-var
KL 0.001、20 epochs；reward 对非法格式、数量错误或重复标签给 -1，否则为
`(cell accuracy + exact batch match) / 2`。Cell-o1 的 prompt 上限为 3072，
但 Qwen3 tokenizer 下有 3/6912 条 train prompt 长度为 3073--3094，因此这里
将 prompt 上限设为 4096，确保完整使用全部训练样本；这是显式记录的唯一
数据长度适配。

```bash
bash cell_annotation/cluster/submit_qwen3_8b_cello1_grpo_rjob.sh
```

主方法按单条完整 rollout 路由：对内部整个 cell batch 严格全对时只做
GRPO；其余 rollout 只做由 ground truth 提供特权信息的 SDPO。Teacher
只看到原 prompt、GT 和因果 student prefix，不包含错误诊断、marker genes
或 o1 reasoning。当前版本不含 entropy dynamic weighting。

```bash
bash cell_annotation/cluster/submit_qwen3_8b_gt_routed_sdpo_rjob.sh
```

每个训练 step 会同时写入 console、`logs/train.log` 和离线 W&B：

```text
routing/sdpo_sample_fraction
routing/grpo_sample_fraction
routing/sdpo_token_fraction
routing/grpo_token_fraction
routing/strict_success_fraction
routing/sdpo_sample_count
routing/grpo_sample_count
```

分支 loss 和 actor micro-batch 的路由占比另外以
`routing/*_branch_loss`、`routing/*_fraction_actor` 记录。

上面的 `v1` 是一个有效的 reasoning SFT 实验，但其 LoRA rank、dev 比例、
sequence length、scheduler 和 global batch 并未严格照搬 Cell-o1。若要做
“只把 Qwen2.5-7B-Instruct 替换为 Qwen3-8B”的受控复现，使用独立入口：

```bash
EXPERIMENT_NAME=qwen3-8b-cello1-reasoning-sft-repro-seed0 \
bash cell_annotation/cluster/submit_qwen3_8b_cello1_reasoning_sft_rjob.sh
```

该入口把复现数据写入
`cell_annotation/data/cello1_reasoning_sft_reproduction`，不会覆盖前面的
95/5 数据；原始 parquet 则只读复用 `cell_annotation/data/raw`，训练节点
不需要再次访问 Hugging Face。它对齐 Cell-o1 的 90/10 划分、seed 0、LoRA/DoRA
`r=256/alpha=512`、1 epoch、`lr=5e-5`、4096 token、linear schedule、
zero warmup，并通过 8 GPU × micro-batch 1 × accumulation 12 保持 global
batch 96。Qwen3 必需的差异只有 base checkpoint 和
`enable_thinking=True` chat template。

训练完成后使用 Cell-o1 的 greedy decoding 和 3500-token reasoning budget：

```bash
MODEL_PATH=/mnt/shared-storage-user/yuzhiyin/SDPO/outputs/cell_annotation/qwen3-8b-cello1-reasoning-sft-repro-seed0/merged_model \
ENABLE_THINKING=1 \
MAX_MODEL_LEN=8192 \
MAX_NEW_TOKENS=3500 \
OUTPUT_DIR=/mnt/shared-storage-user/yuzhiyin/SDPO/outputs/cell_annotation/eval_qwen3-8b-cello1-reasoning-sft-repro-seed0 \
JOB_NAME=qwen3-8b-cello1-reasoning-sft-repro-seed0-eval \
bash cell_annotation/cluster/submit_qwen3_8b_eval_rjob.sh
```

## 2.2 ChatCell-large 基线

`zjunlp/chatcell-large` 是 `T5ForConditionalGeneration`，不是 chat decoder
模型，不能直接复用 Qwen3 的 chat template/vLLM 入口。下载固定 revision：

```bash
cd /mnt/shared-storage-user/yuzhiyin/SDPO
bash cell_annotation/scripts/download_chatcell_large.sh
```

模型和 Hugging Face 缓存都会写入共享盘，默认模型目录为：

```text
/mnt/shared-storage-user/ma4tool-shared/all_users_shared/yuzhiyin/chatcell/chatcell-large
```

提交单 GPU 评测：

```bash
bash cell_annotation/cluster/submit_chatcell_large_eval_rjob.sh
```

ChatCell 的主结果按 **closed-set cell-level classification** 评测。对于每个
cell，prompt 只包含该 cell 的 top genes 和它所在 batch 的候选 cell type
列表；模型计算该 cell 对每个候选标签的 T5 条件对数似然并独立取最大值。
不同 cell 可以预测为相同标签，不做 Hungarian 一一匹配。默认配置等价于：

```bash
INCLUDE_CONTEXT=0 \
INCLUDE_CANDIDATES=1 \
ASSIGNMENT_MODE=independent \
bash cell_annotation/cluster/submit_chatcell_large_eval_rjob.sh
```

`metrics/*.json` 报告 `cell_accuracy`、`macro_cell_type_recall`、MRR、top-1/3/5
accuracy、不同候选数分组结果和每类 recall；不报告 batch exact match，也不把
输出唯一性当作有效性条件。

如需复现之前的 batch assignment 版本，可显式设置
`ASSIGNMENT_MODE=hungarian`。该模式会恢复最大总似然的一一映射，并使用通用
batch 评测器。`INCLUDE_CONTEXT` 和 `INCLUDE_CANDIDATES` 仍可用于消融。

## 2.3 Cell2Sentence cell-level 基线

该实现只评测 independent cell-level classification，不做 Hungarian 或任何
batch-level 指标。每个 cell 的 prompt 只包含它自己的 top-50 ranked genes
以及来源 batch 的候选类型范围；模型计算每个候选标签的长度归一化条件对数似然，
各 cell 独立取最高分，因此允许不同 cell 预测为同一类型。

模型必须在开发机联网下载到共享盘。下载脚本会先清理大小写代理变量，再执行集群
代理初始化；不要在 GPU rjob 中执行：

```bash
cd /mnt/shared-storage-user/yuzhiyin/SDPO
bash cell_annotation/scripts/download_c2s_model.sh
```

固定的官方模型及默认本地路径为：

```text
vandijklab/C2S-Pythia-410m-cell-type-prediction
revision 122c0ad022342f070549ebb37f3840fc08325cc0
/mnt/shared-storage-user/ma4tool-shared/all_users_shared/yuzhiyin/C2S/C2S-Pythia-410m-cell-type-prediction
```

GPU run 脚本设置 `TRANSFORMERS_OFFLINE=1`、`HF_HUB_OFFLINE=1`，rjob 使用
`--host-network=false`。提交脚本还会拒绝任何包含 `cell` 的 rjob name。

官方 checkpoint 的 zero-shot 评测：

```bash
RUN_LABEL=zero \
bash cell_annotation/cluster/submit_c2s_eval_rjob.sh
```

CellPuzzles fine-tuning 将原有 6,566/346 个 train fit/dev batch 分别展开为
65,252/3,446 个独立单细胞样本，不使用 test、test_clean 或 unseen。默认配方
遵循 C2S tutorial：full-model fine-tuning、5 epochs、LR `1e-5`、global batch
32、cosine scheduler、5% warmup、bf16，并对完整 prompt+response 计算 LM loss。

```bash
EXPERIMENT_NAME=c2s-ft-seed42 \
bash cell_annotation/cluster/submit_c2s_finetune_rjob.sh
```

默认模型输出：

```text
/mnt/shared-storage-user/ma4tool-shared/all_users_shared/yuzhiyin/ckpt-c2s/c2s-ft-seed42/final_model
```

微调模型使用完全相同的评测代码：

```bash
RUN_LABEL=ft-seed42 \
MODEL_PATH=/mnt/shared-storage-user/ma4tool-shared/all_users_shared/yuzhiyin/ckpt-c2s/c2s-ft-seed42/final_model \
bash cell_annotation/cluster/submit_c2s_eval_rjob.sh
```

两次评测均报告 test、test_clean、unseen_all 和四个独立 unseen 数据集的
`cell_accuracy`、macro recall、MRR、top-1/3/5 accuracy、候选数分组、每类
recall，以及按 cell type 是否出现在 CellPuzzles train 中划分的 seen/unseen
label accuracy。它们的唯一实验差别是 `MODEL_PATH`。

## 3. GT 特权信息 SDPO

### 3.1 奖励

`reward.py` 的默认总奖励范围是 `[0, 1]`：

```text
reward = 0.10 × valid_output_format
       + 0.45 × positional_cell_accuracy
       + 0.45 × batch_exact_match
```

其中 `valid_output_format` 要求标签外层格式严格、数量等于 N、无重复且恰好是
给定候选集合的一次排列。权重可通过环境变量覆盖，但三者必须加和为 1：

```bash
CELL_REWARD_FORMAT_WEIGHT=0.10
CELL_REWARD_PARTIAL_WEIGHT=0.45
CELL_REWARD_EXACT_WEIGHT=0.45
```

reward 返回的 `feedback` 包含 ground-truth assignment、格式要求和本次答案诊断。
SDPO 的 reward manager 会把它放进 `reward_extra_info["feedback"]`，trainer 再用
同一个 EMA teacher 在“原 prompt + GT 特权反馈”上重提示，并对学生的原始
on-policy response 做 teacher-forced logit 蒸馏。

训练脚本的关键隔离设置是：

```text
success_reward_threshold = 2.0
include_environment_feedback = True
environment_feedback_only_without_solution = False
```

完整 Qwen3-8B FSDP checkpoint（model、optimizer、extra）约占 85 GiB。
GT-SDPO 脚本因此默认把输出写到：

```text
/mnt/shared-storage-user/ma4tool-shared/all_users_shared/yuzhiyin/ckpt-sdpo/cell_annotation
```

并设置 `trainer.max_actor_ckpt_to_keep=1`。可以用 `OUTPUT_ROOT` 和
`MAX_ACTOR_CKPT_TO_KEEP` 覆盖；不要把完整 checkpoint 写到空间不足的仓库盘。

因为奖励上限为 1，阈值 2.0 禁止把同批其他 rollout 当成“成功示范”；教师只从
GT feedback 获得特权信息。学生 rollout 和最终推理 prompt 从不包含 GT。

### 3.2 从 base model 直接做 GT-SDPO

```bash
cd /mnt/shared-storage-user/yuzhiyin/SDPO
bash cell_annotation/cluster/submit_qwen3_8b_gt_sdpo_rjob.sh
```

先 dry-run：

```bash
DRY_RUN=1 bash cell_annotation/cluster/submit_qwen3_8b_gt_sdpo_rjob.sh
```

常用超参数覆盖示例：

```bash
EXPERIMENT_NAME=cell-gt-sdpo-seed1 \
ROLLOUT_N=4 \
TRAIN_BATCH_SIZE=16 \
PPO_MINI_BATCH_SIZE=16 \
ACTOR_LR=1e-6 \
DISTILLATION_TOPK=100 \
DISTILLATION_ALPHA=1.0 \
TEACHER_UPDATE_RATE=0.01 \
TOTAL_EPOCHS=3 \
bash cell_annotation/cluster/submit_qwen3_8b_gt_sdpo_rjob.sh
```

`DISTILLATION_ALPHA=0/0.5/1` 分别对应 forward KL / JSD / reverse KL。建议首轮
固定其他参数，比较 `0.5` 和 `1.0`。

### 3.3 同 reward 的 GRPO 对照

在已经分配好的节点中，脚本最后的 Hydra 参数会覆盖默认值：

```bash
bash cell_annotation/cluster/run_qwen3_8b_gt_sdpo_rjob.sh \
  actor_rollout_ref.actor.policy_loss.loss_mode=vanilla
```

该对照仍使用完全相同的 correctness/format reward，但不使用 GT feedback 的
teacher logits，可分离“强化学习奖励”与“特权自蒸馏”的收益。

## 4. 是否先做 SFT

结论：SFT **不是算法上的必要前置条件**。GT feedback 对每个 rollout 都存在，
所以即使 base model 没有成功样本，teacher 也能被特权信息引导。这一点正适合作为
论文的核心实验：`Base -> GT-SDPO`。

但为了追求最稳定的最终性能，建议同时做一个很短的 LoRA SFT 分支。这里的 SFT：

- 只使用 CellPuzzles train；
- target 只监督 non-thinking 的 `<answer>gold</answer>` continuation；
- 不使用 Cell-o1 reasoning、test、unseen 或 marker gene；
- 只跑 1 epoch，主要学习标签空间、一对一排列和输出协议。

提交 SFT：

```bash
cd /mnt/shared-storage-user/yuzhiyin/SDPO
bash cell_annotation/cluster/submit_qwen3_8b_sft_rjob.sh
```

默认输出：

```text
outputs/cell_annotation/<experiment>/
├── lora/final_adapter/
└── merged_model/
```

然后以合并后的模型启动 SDPO：

```bash
STUDENT_MODEL=/mnt/shared-storage-user/yuzhiyin/SDPO/outputs/cell_annotation/<sft-experiment>/merged_model \
EXPERIMENT_NAME=cell-sft-then-gt-sdpo \
bash cell_annotation/cluster/submit_qwen3_8b_gt_sdpo_rjob.sh
```

建议至少完成四个主对照：

| 实验 | 初始化 | GT teacher feedback | 目的 |
|---|---|---:|---|
| Base | Qwen3-8B | 否 | zero-shot 基线 |
| SFT | Qwen3-8B + 1 epoch LoRA SFT | 否 | 直接监督基线 |
| GT-SDPO | Qwen3-8B | 是 | 核心自蒸馏贡献 |
| SFT + GT-SDPO | SFT merged model | 是 | 预计最佳性能 |

另外加入“同 reward 的 GRPO”才能证明增益不是仅由规则奖励造成。每个训练设置建议
至少 3 个 seed，并用 train 内部 dev 做选择；官方 test、test_clean 和 unseen
只用于最终报告。

## 5. 当前范围与下一阶段

当前 privileged information 只实现 ground truth，符合第一阶段要求。marker gene
和 top-k 相似细胞尚未加入。后续可保持 reward 不变，只替换 `feedback` 构造器，
形成可控消融：

```text
GT
GT + marker
GT + retrieved cells
GT + marker + retrieved cells
retrieved cells only
```

检索索引必须仅由 train 构建；test/unseen 的真实标签不得进入索引内容或 teacher
feedback。这样才能把提升归因于训练期特权信息，而不是评测泄漏。

## 6. GenePT-s source-only 10-NN baseline

该 baseline 使用 GenePT 仓库中的 sentence embedding 方法：每个 cell 的 50 个
ranked genes 被转换为：

```text
A cell with genes ranked by expression: GENE1 GENE2 ... GENE50
```

然后由 `text-embedding-ada-002` 生成 1536 维向量。API 只允许在可联网的开发机
运行；rjob 只读取已经缓存的向量并执行离线 exact cosine 10-NN。KNN reference
严格使用完整 CellPuzzles train，test_clean 和四个 unseen 数据集的标签不进入
索引或超参数选择。

准备并去重输入：

```bash
cd /mnt/shared-storage-user/yuzhiyin/SDPO
bash cell_annotation/scripts/prepare_genept_s.sh
```

在开发机配置代理，并通过环境变量提供 OpenAI-compatible 中转配置。不要把 API
key 写入仓库或命令行：

```bash
unset https_proxy http_proxy
source <(curl -sSL http://deploy.i.h.pjlab.org.cn/infra/scripts/setup_proxy.sh)
export OPENAI_BASE_URL=http://your-openai-compatible-endpoint/v1/
read -rsp 'OPENAI_API_KEY: ' OPENAI_API_KEY
export OPENAI_API_KEY

# 先验证模型名、接口和 1536 维输出。
python3 -m cell_annotation.embed_genept_s \
  --artifact-dir cell_annotation/data/genept_s_ada002 \
  --model text-embedding-ada-002 \
  --dimension 1536 \
  --probe-only

# 全量请求按 chunk 落盘，可在中断后用同一命令续传。
python3 -m cell_annotation.embed_genept_s \
  --artifact-dir cell_annotation/data/genept_s_ada002 \
  --model text-embedding-ada-002 \
  --dimension 1536 \
  --batch-size 1024 \
  --workers 3

unset OPENAI_API_KEY
```

全量缓存包含 80,108 个 cell 和 76,174 个去重文本。完成后应存在：

```text
cell_annotation/data/genept_s_ada002/
├── manifest.json
├── records.jsonl
├── unique_inputs.jsonl
├── embedding_manifest.json
├── embedding_chunks/
├── embeddings.npy                 # (76174, 1536), float32
└── embedding_completed.npy
```

提交无网络的 1-GPU exact KNN 评测：

```bash
bash cell_annotation/cluster/submit_genept_s_eval_rjob.sh
```

底层离线命令为：

```bash
python3 -m cell_annotation.evaluate_genept_s \
  --artifact-dir cell_annotation/data/genept_s_ada002 \
  --output-dir outputs/genept/local_eval \
  --train-split train \
  --k 10 \
  --device cuda
```

输出分别包含 test_clean、breast cancer、colorectal cancer、melanoma、SLE 和
四个 unseen 合并后的 accuracy、macro precision/recall/F1、train-label cell
coverage，以及 train-seen/train-unseen 标签子集准确率。默认是 GenePT 原论文风格
的 global 10-NN，不读取每个 CellPuzzles batch 的 candidate labels，也不做
Hungarian batch assignment。

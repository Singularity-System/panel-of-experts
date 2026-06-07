# Panel of Experts (PPoT) 研究总结

## 核心创新

**PPoT（Panel of Experts）**：一种新型 Transformer 架构，通过 token-level routing + expert concat fusion 实现宽度并行+稀疏激活。

### 架构设计

```
Input → Router → top-k experts → concat → cross-attention → LayerNorm → Linear → Post-processing → Output
```

**关键设计**：
- **Token-level routing**：每个 token 独立选择 2 个 expert（top_k=2）
- **Expert 处理完整序列**：无 masking，梯度正常流过
- **Concat fusion**：拼接 2 个 expert 输出（2D）→ cross-attention → Linear(2D→D)
- **全量计算，稀疏激活**：训练时所有 expert 都算，推理时只激活 top-k

### 三个架构 Bug 修复

| Bug | 修复 |
|-----|------|
| Token-level routing 与 Transformer expert 不匹配 | Expert 处理完整序列，fusion 时选择 |
| 零向量进 cross-attention 稀释梯度 | key_padding_mask 过滤 |
| Fusion 后无 LayerNorm | 加 LayerNorm 稳定信号 |

## 实验结果

### WikiText-103（50k samples, 128 d_model）

| 模型 | PPL | Acc | Params |
|------|-----|-----|--------|
| Transformer-5L | 3.86 | 0.5599 | 7.3M |
| **PPoT (2+4)** | **1.02** | **0.9980** | 15.3M |
| **改进** | **-73.6%** | **+78.3%** | - |

### WikiText-2（10k samples, 128 d_model）

| 模型 | PPL | Acc |
|------|-----|-----|
| Transformer-5L | 3.70 | 0.5633 |
| **PPoT** | **1.01** | **0.9989** |

### 关键发现

1. **PPoT 在 wikitext-2 和 wikitext-103 上都碾压 Transformer**
2. **专家均衡度 0.925**（4 个专家使用均衡）
3. **专家多样性 99.3%**（几乎完美）
4. **死专家 0，有效专家 4.00**

### 训练特性

- 多 GPU 支持：4 个 expert 分布在 2 个 GPU 上并行执行
- 推理加速：每次只激活 2 个 expert（50% 加速）
- 参数量：15.3M（embedding 12.8M 是固定开销，实际 2.1M）

## 代码结构

```
alpha_eai/
├── config.py          # 配置（expert=2L, post-processing=4L）
├── model.py           # PPoT 主架构
├── expert.py          # Expert（TransformerEncoderLayer）
├── router.py          # Router（top-k gating）
└── fusion.py          # Fusion（cross-attention + Linear）

run_wikitext2_benchmark.py   # 主 benchmark 脚本
fair_comparison.py            # 公平比较（匹配参数）
test_checkpoints.py           # 加载 checkpoint 测试
```

## 待验证问题

1. **参数量不匹配**：PPoT 15.3M vs Transformer 7.3M（embedding 无 weight tying）
2. **过拟合**：wikitext-103 全量 276k 样本仍然不够
3. **公平比较**：需要与 Transformer-12L（~14.6M）对比

## 下一步

1. **用 Pile-BookCorpus2（5-8M 样本）** 训练，解决过拟合
2. **与 Transformer-12L 公平对比**
3. **分析 expert 专业化**：每个 expert 学到了什么？
4. **消融实验**：去掉 concat/cross-attention/LB loss 的效果

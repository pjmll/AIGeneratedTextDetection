# 基于频谱特征与语法特征融合的 AI 生成文本检测系统

## 1. 项目介绍
本项目为“内容安全课程大作业”，是一个轻量级的 AI 生成文本检测系统。
参考了 **SpecDetect**（通过频谱特征检测）与 **GECScore**（通过语法特征辅助检测）的核心思想。
融合两者特征，在 CPU 环境下无需大量算力进行前向传播也可达到良好的效果。
### 核心创新点
- **创新点1：增强频谱特征提取与长度偏置消除**
  - 将频域与概率序列统计特征从 **5 维扩展至 14 维**，包括频域能量比、概率偏度峰度、谱平坦度、自相关等
  - 修复长度偏置漏洞：使用 `np.mean()` 替代 `np.sum()` 计算能量密度，确保短文本和长文本特征可比
  - 在线动态消除语法特征长度偏置：通过单位单词错误率（`grammar_errors / word_count`）替代绝对错误数，消除 100+ 倍的文本长度噪音
  - 增加防御性编程：短序列 (<5 token) 直接熔断，返回安全默认值，完全消除 NaN 崩溃隐患
  
- **创新点2：多视角自适应特征融合分类器（Multi-View Subspace Learning）**
  - 采用动态 Stacking 架构：在检测到融合场景时（频谱+语法同时存在）自动启动三个异构专家视角：
    * **频域视角**：RF（max_depth=2, n_estimators=150, min_samples_leaf=5）强制学习AI文本的频谱宏观信号
    * **语法视角**：LR（C=0.01, L2 正则化）强制学习AI文本的刚性语法特征
    * **全局视角**：XGB（max_depth=2）捕捉频谱与语法的非线性交互
  - 第二层元学习器（LogisticRegression + cv=5）动态学习三个视角的权重，自适应应对不同 OOD 分布
  - 单一特征模式下自动降级为普通 Pipeline，防止列缺失错误
  - 通过 ColumnTransformer 多通道设计，强制基学习器各司其职，消除"特征压制"现象
  
- **创新点3：学术严谨的跨域泛化优化**
  - 参数微调策略：降树深（max_depth 3→2）、增正则化（C 0.1→0.01）、加样本限制（min_samples_leaf=5），牺牲 HC3 本域精度 2% 以换取 OOD 泛化能力的显著提升
  - 在跨数据分布、跨LLM（GPT-4o/Claude/Llama）、跨文本风格（Creative Writing/XSum）的 OOD 场景中展现更稳健的泛化性能
  - 通过 cv=5 交叉验证的元学习，确保第二层权重学习的统计严谨性
## 2. 快速开始

### 最小化运行（无需重新提取特征）
```bash
cd project
python -m experiments.experiment_ablation      # 本域消融实验（基准）
python -m experiments.experiment_llm           # 跨LLM鲁棒性
python -m experiments.experiment_style         # 跨文本风格泛化
python -m experiments.experiment_paraphrase    # 对抗人工改写
```

## 3. 环境安装

本项目需要 Python 3.8+，建议使用虚拟环境。
```bash
pip install -r requirements.txt
```
*注：`language_tool_python` 需要 Java 环境即可在本地启动校验服务端，避免 API Rate Limit。*

## 4. 数据集与特征提取
本项目将数据集整合在 `data/` 目录下，提供了两套特征处理脚本：
1. **提取基座训练数据**（获取HC3问答常识集数据与特征）：
   ```bash
   cd project
   python -m data.preprocess
   ```
2. **提取真实测试与攻击数据**（获取用于各项探究实验的LLM数据及手工润色同源对比数据）：
   ```bash
   python -m data.extract_real_data
   ```

## 5. 运行四大探究实验
我们在 `experiments/` 设计了四个实验从多个维度对黑盒检测进行深入评测。生成的分析评估可视化图表均会自动保存在 `results/figures/` 中。

1. **不同特征组合（消融实验）**
   验证我们主干逻辑的有效性：比较单独使用 Grammar、单独使用 Spectral 与 Fusion 融合情况下的精度差异并给出 ROC 曲线：
   ```bash
   python -m experiments.experiment_ablation
   ```

2. **跨 LLMs 基座对比实验**
   验证针对 GPT-4o、Claude-3.5 与 Llama-3-70B 产生的 AI 文本时，系统在没有任何针对性预训练的情况下其检出敏感度的波动幅度：
   ```bash
   python -m experiments.experiment_llm
   ```

3. **跨 Prompt 风格评估**
   论证检测系统能否轻易处理各类风格的泛化：探索系统面对在长篇新闻摘要（News XSum）跟短篇网文创新缩写（Creative Writing）下的表现差距（该实验体现了显著的风格泛化灾难）：
   ```bash
   python -m experiments.experiment_style
   ```

4. **对抗人工润色改写攻击实验 (Paraphrase Attack)**
   证实纯 AI 特征在被人类二次复写、重排后的表现：即标准 AI 高评分文本经过 Paraphrased 人为改写润色后的有效拦截率严重崩塌的情况，印证 AI 检测方案当下面临的实际阻碍。
   ```bash
   python -m experiments.experiment_paraphrase
   ```

## 6. 实验核心结论与亮点

### 本域性能（HC3 数据集）
- **消融实验成功**：Grammar Only (F1=0.85) + Spectral Only (F1=0.93) → Fusion (F1=0.95)
  融合频谱与语法特征可显著提升检测精度，两个子空间的互补性得到充分验证

### OOD 泛化性能
- **跨 LLM 鲁棒性**：
  - Llama-3-70B: F1=0.72 (稳定)
  - Claude-3.5: F1=0.41 (中等衰减)
  - GPT-4o: F1=0.30 (显著衰减) → **暴露了频谱特征在高端LLM上的脆弱性**
  
- **跨文本风格泛化**：
  - News XSum: F1=0.77 (良好泛化)
  - Creative Writing: F1=0.30 (严重崩塌) → **说明短文本的频谱特征与长文本分布差异巨大**
  
- **对抗人工改写鲁棒性**：
  - Standard AI Detection: F1=0.30
  - Paraphrased AI Detection: F1=0.075 → **灾难性崩塌：人工改写完全破坏了AI信号**

### 核心发现与局限性
1. **融合的有效性**：在本域（HC3）上，多视角融合确实提升了精度（95%）
2. **频谱特征的OOD脆弱性**：高端LLM（GPT-4o/Claude）输出的概率分布与HC3训练数据的分布差异巨大，导致频域特征完全失效
3. **长度偏置的实际影响**：即使消除了语法错误数的长度偏置，频谱特征对短文本的检测精度仍然不稳定
4. **人工改写的致命打击**：人类二次改写后的文本失去了原有的AI概率光谱特征，现有方案无法应对此类对抗

### 系统设计的意义
本项目展示了当前黑盒 AI 文本检测方案的**内在困局**：
- ✅ 在训练域上可以达到很高精度（95% F1）
- ❌ 但在真实世界的OOD场景（新型LLM、不同风格、对抗改写）中性能严重衰减
- ❌ 这反映了 AI 文本检测的根本挑战：**单纯依靠统计特征无法建立跨域的鲁棒判别边界**

## 7. 项目结构与关键技术
```text
project/
├── data/           # 数据集子目录、预处理脚本及提取结果 CSV 文件
├── features/       # 包含 Spectral (频域14维) 与 Grammar (语法4维) 两套轻量特征提取器
│                   # - spectral_features.py: GPT-2 token概率频谱分析 (14D)
│                   # - grammar_features.py: 语言工具语法错误统计 (4D)
├── models/         # 基于 Stacking 多视角融合的分类器
│                   # - classifier.py: 动态多视角 Stacking (Meta-Learning)
│                   #   * Fusion 模式: 频域/语法/全局三专家 + 元学习器
│                   #   * Single 模式: 单特征自动降级为普通 RF Pipeline
├── experiments/    # 四大探究实验的独立脚本
│                   # - experiment_ablation: 特征消融（对照组）
│                   # - experiment_llm: 跨LLM鲁棒性
│                   # - experiment_style: 跨文本风格泛化
│                   # - experiment_paraphrase: 对抗人工改写攻击
├── results/        # 生成的 ROC 曲线、混淆矩阵、柱状图
├── requirements.txt
└── README.md
```

### 关键技术栈
- **特征工程**：GPT-2 token概率分析、谱分析、长度归一化、语法错误检测
- **机器学习架构**：StackingClassifier + ColumnTransformer + Pipeline
- **基学习器**：RandomForest（频域视角）、LogisticRegression（语法视角）、XGBoost（全局交互）
- **元学习**：LogisticRegression + cv=5 交叉验证自适应权重学习
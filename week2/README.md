# 词向量学习实验报告

胡诚成 2022213665

## 实验介绍

实现两种词向量学习方法：1）基于 skip-gram negative sampling 的预测式词向量；2）基于 GloVe 的计数式词向量。语料使用中文维基百科 dump `data/zhwiki-latest-pages-articles.xml.bz2`。

## 实验方法

### 预处理与词表

- 解析 Wiki dump：逐文档读取，按行清洗。
- 切分：正则 `\p{Han}+|[a-zA-Z]+|\d+|[\p{P}\p{S}]` 匹配中文、拉丁字母、数字、标点，统一小写，丢弃空白。
- 构建词表：统计词频，保留 `<pad>`、`<unk>`，过滤低频（默认 `min_freq=5`），可限制最大词表。
- 子采样：对高频词以阈值 $t=10^{-4}$ 进行采样，保留概率 $1-\sqrt{t/f(w)}$（若 $f(w)>t$），减少停用词主导。

### Skip-gram Negative Sampling (SGNS)

- 窗口采样：对每个中心词 $w$，在窗口大小 $k$ 内选取上下文 $c$ 形成正样本；对每个正样本采样 $m$ 个负样本 $n_i$。
- 目标函数：
  $$\mathcal{L} = -\log\sigma(v_w^\top v_c) - \sum_{i=1}^m \log\sigma(-v_w^\top v_{n_i})$$
- 参数：输入嵌入 $v$、输出嵌入 $u$（实现中共用 `in_embed` / `out_embed`）。
- 梯度：对输入向量 $v_w$，梯度为
  $$\nabla_{v_w} = (\sigma(v_w^\top u_c)-1)u_c + \sum_{i=1}^m \sigma(v_w^\top u_{n_i})u_{n_i}$$
  对正样本输出向量 $u_c$，梯度为 $(\sigma(v_w^\top u_c)-1)v_w$；对每个负样本输出向量 $u_{n_i}$，梯度为 $\sigma(v_w^\top u_{n_i})v_w$。
- 优化：使用 Adam，mini-batch 负对数似然下降。

### GloVe

- 共现构建：窗口内按距离加权 $X_{ij} += 1/|i-j|$ 统计共现。
- 损失：
  $$\mathcal{L} = \sum_{i,j} f(X_{ij})\big(v_i^\top v_j + b_i + b_j - \log X_{ij}\big)^2$$
  其中 $f(x) = (x/x_{\max})^{\alpha}$ 当 $x < x_{\max}$，否则为 1。
- 参数与优化：词/上下文嵌入与偏置，使用 Adagrad；最终词向量为两嵌入之和。

### 运行命令

- 训练 SGNS：

```bash
train-sgns --data-root . --epochs 3 --batch-size 512 --embedding-dim 200 --window-size 5 --negative 5
```

- 训练 GloVe：

```bash
train-glove --data-root . --epochs 25 --embedding-dim 200 --window-size 10 --x-max 100 --alpha 0.75
```

- 训练日志与向量将写入 `output/`；SGNS 向量 `sgns.vec.pt`，GloVe 向量 `glove.vec.pt`。

## 实验结果

- 训练曲线与对比可视化：![curves](./assets/curves.png)
- 词向量可视化/类比示例：![tsne](./assets/tsne.png)
- 量化指标（示例占位，需根据下游任务或相似度评测填写）：
  - SGNS：...
  - GloVe：...

## 实验结论

- SGNS 属预测式模型，对高频共现敏感，通过负采样高效优化，适合大规模语料快速训练。
- GloVe 利用全局共现，能捕获更平滑的统计关系，对低频词表现更稳；但构建共现矩阵内存开销较大。
- 后续可尝试：调节子采样、负采样分布，使用分片/稀疏存储构建共现矩阵，或在下游任务上对比两类向量的效果以做更精确评估。

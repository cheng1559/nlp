# Seq2seq  实验报告

胡诚成 2022213665

## 实验介绍
- 目标：复现 fastText 词向量、实现天津话到普通话的 seq2seq 翻译、以及基于字符的 POS/NER 标注。
- 数据：Tianjin_dataset 平行语料、result-rmrb.txt 标注语料。

## 实验方法
### fastText 词向量
- 采用 skip-gram + 子词 n-gram（`<w>` 包裹，n∈[3,6]）构建子词嵌入，整体词向量为主词嵌入与子词均值之和。
- 目标函数为负采样二元分类：$\mathcal{L} = -\log \sigma(\mathbf{v}_c^\top \mathbf{v}_p) - \sum_{n=1}^K \log \sigma(-\mathbf{v}_c^\top \mathbf{v}_{n})$，其中 $\mathbf{v}_c$ 为中心词向量，$\mathbf{v}_p$ 为正样本向量，$\mathbf{v}_n$ 为负样本向量。
- 数据管道：正向窗口采样，随机负采样；Lightning DataModule 负责划分与批处理。
- 运行命令：
  ```bash
  train-fasttext --data-root . --corpus data/result-rmrb.txt --epochs 3 --output-dir output
  ```

### 天津话→普通话 seq2seq
- 字符级词表，添加 `<s>`, `</s>`, `<pad>`, `<unk>` 特殊符号。
- 编码器/解码器可选 RNN/GRU/LSTM；训练时使用教师强制，推理阶段自回归。
- 损失函数：$\mathcal{L} = -\sum_{t} \log p(y_t\mid y_{<t}, x)$，实现为交叉熵并忽略 `<pad>`。
- 运行命令：
  ```bash
  train-seq2seq --data-root . --corpus data/Tianjin_dataset/俗世奇人part1.json --rnn gru --epochs 10 --output-dir output
  ```

### 字符级 POS / NER
- 输入：逐字符 ID；标签：POS 直接平铺到字符，NER 使用简化 BIO（B-/I-，O），未映射的标签归为 OTHER/O。
- 模型：双向 LSTM 编码 + 线性分类；损失为交叉熵（忽略 padding 标签）。
- 运行命令：
  ```bash
  train-tagger --data-root . --corpus data/result-rmrb.txt --task POS --epochs 5 --output-dir output
  train-tagger --data-root . --corpus data/result-rmrb.txt --task NER --epochs 5 --output-dir output
  ```

## 实验结果
- 本地训练日志与检查点保存在 `output/`（Lightning 默认记录 loss/acc）。
- 建议可视化：
  - fastText 损失曲线：`./assets/fasttext_loss.png`
  - seq2seq 验证损失曲线：`./assets/seq2seq_val.png`
  - POS/NER 验证准确率曲线：`./assets/tagger_acc.png`
- 如需客观指标，可在 seq2seq 上追加字符级 BLEU / CER，POS/NER 上报告字符级准确率与 F1。

## 实验结论
- fastText 子词建模适合低资源汉字变体，可缓解 OOV；窗口与负采样数对收敛速度影响明显。
- GRU/LSTM 在该字符级翻译任务中通常优于 vanilla RNN，增加隐藏维度可提升拟合但需防过拟合（验证集监控）。
- 简单双向 LSTM 对字符级 POS/NER 已能取得合理基线，后续可尝试 CRF 解码或预训练字向量以提升序列一致性。

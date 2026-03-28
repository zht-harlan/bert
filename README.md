# BERT baseline for node classification

这个项目提供了一个可直接运行的 `BERT` 风格节点分类基线，覆盖以下数据集：

- `ogbn-arxiv`
- `cora`
- `pubmed`
- `amazon-photo`

代码特点：

- 使用 `transformers==4.42.3` 中的 `BertModel`
- 将每个节点的特征向量转成 `top-k` 特征 token 序列，再送入 BERT
- 每个数据集重复运行 5 次
- 输出每次结果和均值/标准差到 `csv`

## 运行方式

默认直接跑四个数据集，每个数据集 5 次：

```bash
python main.py --output-dir outputs
```

只跑单个数据集示例：

```bash
python main.py --datasets cora --runs 5 --device cuda --output-dir outputs_cora
```

如果服务器显存有限，可以先减小以下参数：

```bash
python main.py ^
  --datasets ogbn-arxiv ^
  --batch-size 128 ^
  --eval-batch-size 512 ^
  --hidden-size 96 ^
  --num-hidden-layers 2 ^
  --num-attention-heads 4 ^
  --max-tokens 32 ^
  --output-dir outputs_arxiv
```

## 输出文件

运行结束后会生成：

- `outputs/per_run_metrics.csv`
- `outputs/summary_metrics.csv`

其中：

- `per_run_metrics.csv` 记录每次运行的 `Acc` 和 `F1-macro`
- `summary_metrics.csv` 记录每个数据集 5 次结果的均值与标准差

## 说明

- `ogbn-arxiv` 使用官方划分
- `cora` 和 `pubmed` 使用 `Planetoid` 自带划分
- `amazon-photo` 没有官方固定划分，这里采用分层随机划分：`20% train / 20% val / 60% test`
- 若你希望完全复现实验，建议固定 `--seed`

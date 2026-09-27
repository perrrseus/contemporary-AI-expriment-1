# 实验一：文本分类

本项目按固定训练/验证划分完成模型比较、正则化参数实验、TF-IDF/预处理消融、错误分析和最终测试集预测。原始 `train_data.csv` 与 `test_data_unlabeled.csv` 不会被修改。

## 环境

建议 Python 3.10 及以上。需要：

```text
numpy
pandas
scikit-learn
matplotlib
```

本次实际运行使用 `random_state=42` 创建一次 80%/20% 分层划分。官方测试集只在最终配置由验证集确定之后参与 `transform` 和预测。

## 运行

在本目录执行：

```powershell
python run_experiments.py
```

脚本默认把已有实验结果复制到 `results/archive/<时间戳>/` 后再运行，避免旧结果丢失。确认不需要归档时可使用：

```powershell
python run_experiments.py --skip-archive
```

## 实验设计

- Experiment A：统一 TF-IDF（unigram、`min_df=2`、`max_df=0.95`、`sublinear_tf=True`、最多 30,000 特征），比较 MultinomialNB、使用 `lbfgs` 的 LogisticRegression、LinearSVC 和单隐藏层 MLPClassifier。
- Experiment B：根据 A 的验证集 Macro-F1 选择 Logistic Regression 与 Linear SVM 中表现最好的两个，对 `C=[0.1,1,10,100]` 做受控比较。使用模型种子 42、2025、3407，但始终保持同一数据划分。
- Experiment C：固定 B 中最佳模型与 C，只改变 unigram/bigram、轻量清洗和 header 去除。另保留一个 30,000 特征上限的 unigram 对照，用于把 A/B 的最佳表示纳入最终选择；核心 A--F 消融均使用同一 80,000 上限。

轻量清洗只做三件事：去除以 `>` 开头的引用行；去除常见的 `In article ... writes:`/`On ... wrote:` 引用引导行；合并多余空白。header 消融独立删除文本开头、空行之前的完整 RFC 风格 header 区块，包括 `Subject`、`From`、`Organization` 等字段。没有使用分词包、词干化、词形还原、Word2Vec、BERT 或 Transformer。

## 输出

- `results/data_inspection.txt`：数据列名、规模、缺失值、类别分布、header 覆盖率和文本节选。
- `results/baseline_results.csv`：四个基线模型。
- `results/tuning_results.csv`：各 C/种子运行及其均值、标准差。
- `results/ablation_results.csv`：TF-IDF、清洗和 header 消融。
- `results/parameter_curve.png`：C 与验证 Macro-F1。
- `results/confusion_matrix.png`、`classification_report.txt`、`misclassified_examples.csv`：最终验证配置的错误分析。
- `results/summary.md`：用于两页报告写作的真实实验记录。
- `predictions.csv`：最终模型对 2,457 条官方测试数据的单列、无表头预测。

时间列为本机墙钟时间。基线表的 `train_time` 是模型拟合时间，因为四个模型共用同一特征矩阵；消融表的 `train_time` 是向量化和模型拟合之和，同时也保留二者的分项时间。

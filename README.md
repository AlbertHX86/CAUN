# CAUN：光伏功率预测区间对比

Adaptive CAUN（Adaptive Cross-Attention Uncertainty Network）与三种常用不确定性量化方法的对比实验。任务是光伏功率的单步预测，并给出 80% 预测区间。

## 文件

| 文件 | 内容 |
|---|---|
| `TCN.py` | 因果卷积 TCN 模块（`TCNBlock`），供编码器调用 |
| `final_comparison.py` | 主实验：训练并评估全部方法，输出下面三个结果文件 |
| `power.xlsx` | 2019-01-01 至 2020-12-31 的 15 分钟光伏数据：辐照度、气温、气压、相对湿度和功率 |
| `power1.xlsx` | 同一时段，只有日期和功率两列 |
| `final_methods_comparison.csv` | 各方法在测试集上的指标 |
| `final_methods_predictions_50_150.csv` | 测试集第 50–150 个点的真实值和各方法的预测区间 |
| `final_comparison_50_150.png` | 上面这段的可视化 |

## 方法

所有模型都用同样结构的 TCN-BiLSTM 编码器。输入是过去 40 步（10 小时）的总辐照度和功率，预测下一步功率。数据按时间顺序划分，最后 2.5% 作为测试集。

1. **Baseline Point Prediction**：只做点预测，也是 NPKDE 的基础模型。
2. **Quantile Regression**：直接预测 10%、50%、90% 分位数。
3. **GP Approximation**：预测均值和方差，按高斯分布给出区间。
4. **NPKDE**：对基准模型在训练集上的残差做高斯核密度估计，叠加到点预测上得到区间。
5. **Adaptive CAUN**：Transformer 解码器输出区间，分两阶段训练。第一阶段直接训练；第二阶段加入自监督学习的不确定性标签，用 KTO 式损失微调。

## 结果

测试集上的 80% 预测区间（`final_methods_comparison.csv`，保留两位小数）：

| 方法 | PICP (%) | MPIW | MAE | RMSE | CWC | IS |
|---|---|---|---|---|---|---|
| Quantile Regression | 86.57 | 11.36 | 2.63 | 4.91 | 11.36 | 15.83 |
| GP Approximation | 90.92 | 15.28 | 2.91 | 5.13 | 15.28 | 18.69 |
| NPKDE | 93.99 | 17.02 | 3.12 | 5.23 | 17.02 | 20.61 |
| Adaptive CAUN | 87.34 | 12.83 | 2.37 | 4.68 | 12.83 | 17.04 |

- **PICP**：真实值落在区间内的比例。
- **MPIW**：区间平均宽度。
- **CWC**：PICP 低于 80% 时对 MPIW 加惩罚，否则等于 MPIW。
- **IS**：区间得分（interval score），越低越好。
- MPIW、MAE、RMSE、CWC 和 IS 的单位与 `power` 列相同。

Adaptive CAUN 的 MAE 和 RMSE 最低；四种方法的 PICP 都高于 80%，区间最窄、IS 最低的是 Quantile Regression。

## 运行

```bash
pip install torch scikit-learn pandas numpy matplotlib openpyxl
python final_comparison.py
```

`final_comparison.py` 的 `load_data()` 里，数据路径是本机的绝对路径（`/Users/albert/Downloads/...`），运行前要改成 `power.xlsx`。

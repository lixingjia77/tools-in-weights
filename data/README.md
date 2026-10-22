# Data

`macslu/label/{train,dev,test}_set.jsonl` 来自 **MAC-SLU**
（[Gatsby1984/MAC_SLU](https://huggingface.co/datasets/Gatsby1984/MAC_SLU)），
以 **Apache-2.0** 许可发布。

本仓库只包含标注文本（约 6.2 MB），**不包含音频**（上游完整数据集约 1.9 GB）。
每条记录的结构：

```json
{"id": "2239", "query": "...", "semantics": {...}, "split_sens": [...]}
```

重新下载：

```bash
huggingface-cli download Gatsby1984/MAC_SLU --repo-type dataset --include "label/*"
```

使用请引用 MAC-SLU 原工作。

# Archived results

这些文件是 2026-10-21 / 10-22 在 RTX 4090 上跑出来的原始产物，**不是手填的**。
重新生成的方式见仓库根目录 README 的 Quickstart。

| 路径 | 内容 | 生成者 |
|---|---|---|
| `eval/*.txt` | `src/eval.py` 的完整终端输出（含分层与分箱） | `make eval` |
| `cost/*.json` | TTFT / KV / 前缀缓存测量 | `src/cost.py` |
| `cost/*_tput_*.json` | 固定 32 token、扫并发的 req/s 与 tok/s | `src/throughput.py` |
| `train_logs/*.json` | 每个训练臂的超参与逐 epoch loss/耗时/峰值显存 | `src/train.py` |
| `manifest.json` | 数据清洗每一步的输入/输出/计数/sha256 | `src/clean/build.py` |
| `tools.tsv` | 41 个工具的训练/评测计数与参数契约 | `src/clean/build.py` |

说明：

- 表中的准确率一律来自 **test（n=647）**；dev 只用于选 epoch，见 `docs/REPORT.md` 附录 D。
- 评测是 greedy，但 vLLM 不是逐字节可复现：同一输入两次运行约 4.6% 输出不同，指标 ±0.5 pt。
  同配置重训差 2.2 pt。小于 2 pt 的差异不做解释。
- `eval_4b_s52.txt` 是 S5.2 参数契约刚落地、尚未加控制臂时的读数，保留作为中间快照。

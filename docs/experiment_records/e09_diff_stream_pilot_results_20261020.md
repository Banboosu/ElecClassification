# E09 pilot：冻结 MOMENT 原序列 vs 一阶差分双流（validation-only）

seed42、10%/40% 标签；只改变 MOMENT 输入流，不训练 backbone，不打开 test。

| 方法 | 10% Macro-F1 | 40% Macro-F1 | 40% 电池 recall/FPR |
|---|---:|---:|---:|
| moment_logit | 81.19 | 83.67 | 76.47/9.85 |
| diff_logit | 90.17 | 91.88 | 92.52/4.11 |
| moment_diff_logit | 91.41 | 93.85 | 93.75/3.48 |
| stack_moment | 93.16 | 95.42 | 95.14/2.71 |
| stack_moment_diff | 94.62 | 96.13 | 95.91/2.17 |
| stack_three | 91.39 | 94.67 | 94.52/3.16 |

## Gate

```json
{
  "checks": {
    "10p": {
      "moment_diff_minus_moment_pp": 10.222843432759454,
      "stack_diff_minus_stack_pp": 1.4571529765354918,
      "stack_three_minus_stack_pp": -1.772643255456141
    },
    "40p": {
      "moment_diff_minus_moment_pp": 10.175916993944789,
      "stack_diff_minus_stack_pp": 0.7188580847992676,
      "stack_three_minus_stack_pp": -0.7449372138023769
    }
  },
  "passed": true,
  "rule": "40p: MOM+DIFF - MOM >= 1.0pp and stack +0.2pp"
}
```

若 40% 的 MOM+DIFF 相对 MOM ≥+1.0pp 且 stack 相对 MOM stack ≥+0.2pp，才扩展四 seed；
否则保留为负结果，不再为追正而改输入/门槛。test 不参与本轮。

- 根目录：`artifacts/improvements/e09_diff_stream/e09_diff_pilot_20261020_v2`；协议：`configs/experiments/improvements/e09_diff_stream_pilot_protocol.json`。

# LDL 证伪实验：周期公式校验

## 1) 计数校验（理论 vs 实测）

| 指标 | 理论 | 实测 | 结论 |
|---|---:|---:|---|
| GEMM_PRELOAD_total | 4896 | 4896 | PASS |
| ADD_total | 3456 | 3456 | PASS |
| DIV_total | 768 | 768 | PASS |
| BARRIER_total | 2688 | 2688 | PASS |

## 2) duration 分布校验

- Cube 实测分布: `{32: 4800, 35: 96}`
- Vector 实测分布: `{1: 6144, 4: 768}`
- Cube 分布判定: **PASS** (期望 `{32:4800,35:96}`)
- Vector 分布判定: **PASS** (期望 `{1:6144,4:768}`)

## 3) 总结

- 总判定: **PASS**
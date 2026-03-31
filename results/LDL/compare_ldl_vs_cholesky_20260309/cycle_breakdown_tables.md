## 8. Cholesky/BlockLDL 操作周期分解（新增）

### 8.1 Cholesky

- Trace: `results/LDL/compare_ldl_vs_cholesky_20260309/cholesky_baseline/trace_c24.csv`\n
- 时间跨度(关键路径近似): `13697` cycles，事件总数: `32448`\n
- 双核图: `cholesky_baseline/timeline_core01.png`\n
| 步骤 | 公式 | 搬运周期 | CUBE周期 | VECTOR周期 | 总周期 | 事件数 |
|---|---|---:|---:|---:|---:|---:|
| 搬运(Load/Store) | $A/B/X 与外存交换$ | 258546 | 0 | 0 | 258546 | 4608 |
| Gram矩阵构建 | $A=H^H H$ | 0 | 3360 | 0 | 3360 | 96 |
| 对角正则 | $A\leftarrow A+\lambda I$ | 0 | 0 | 96 | 96 | 96 |
| POTRF对角开方 | $L_{jj}=\operatorname{chol}(\tilde A_{jj})$ | 0 | 0 | 3072 | 3072 | 768 |
| TRSM列缩放 | $L_{ij}=(\cdots)L_{jj}^{-H}$ | 0 | 0 | 10752 | 10752 | 2688 |
| Schur/RK更新 | $A_{ik}\leftarrow A_{ik}-L_{ij}L_{kj}^H$ | 0 | 258048 | 0 | 258048 | 8064 |
| 前向对角求逆 | $Y_{jj}=1/L_{jj}$ | 0 | 0 | 3072 | 3072 | 768 |
| 前向累乘 | $Y_{ij}\mathrel{-}=L_{ik}Y_{kj}$ | 0 | 86016 | 0 | 86016 | 2688 |
| 前向缩放 | $Y_{ij}\leftarrow Y_{ij}/L_{ii}$ | 0 | 0 | 10752 | 10752 | 2688 |
| 后向组装 | $A^{-1}=Y^HY$ | 0 | 3072 | 0 | 3072 | 96 |
| 同步屏障(Barrier) | $列/阶段依赖同步$ | 0 | 0 | 1824 | 1824 | 1824 |

### 8.2 BlockLDL

- Trace: `results/LDL/compare_ldl_vs_cholesky_20260309/ldl_new/trace_c24.csv`\n
- 时间跨度(关键路径近似): `3844` cycles，事件总数: `16416`\n
- 双核图: `ldl_new/timeline_core01.png`\n
| 步骤 | 公式 | 搬运周期 | CUBE周期 | VECTOR周期 | 总周期 | 事件数 |
|---|---|---:|---:|---:|---:|---:|
| 搬运(Load/Store) | $A/B/X 与外存交换$ | 246520 | 0 | 0 | 246520 | 4608 |
| Gram矩阵构建 | $A=H^H H$ | 0 | 3360 | 0 | 3360 | 96 |
| 对角正则 | $A\leftarrow A+\lambda I$ | 0 | 0 | 96 | 96 | 96 |
| D块更新 | $D_{jj}=A_{jj}-\sum L_{jk}D_{kk}L_{jk}^H$ | 0 | 24576 | 0 | 24576 | 768 |
| D块求逆 | $D_{jj}^{-1}$ | 0 | 0 | 3072 | 3072 | 768 |
| L块更新 | $L_{ij}=(A_{ij}-\sum L_{ik}D_{kk}L_{jk}^H)D_{jj}^{-1}$ | 0 | 21504 | 0 | 21504 | 672 |
| 回代对角初始化 | $X_{jj}=D_{jj}^{-1}$ | 0 | 21504 | 672 | 22176 | 1344 |
| 回代（乘+加） | $tmp\mathrel{+}=L^H X\;\text{并累加到}\;X$ | 0 | 86016 | 2688 | 88704 | 5376 |
| 同步屏障(Barrier) | $列/阶段依赖同步$ | 0 | 0 | 2688 | 2688 | 2688 |

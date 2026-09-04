<h1 align="center">bioIOT</h1>

<p align="center">
  <b>即插即用的半松弛逆最优传输：单细胞状态转移分析</b>
</p>

<p align="center">
  <a href="https://pypi.org/project/bioIOT/"><img src="https://img.shields.io/pypi/v/bioIOT" alt="PyPI"></a>
  <a href="https://github.com/XTSgreen/bioIOT-py/actions/workflows/ci.yml"><img src="https://github.com/XTSgreen/bioIOT-py/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://img.shields.io/pypi/pyversions/bioIOT"><img src="https://img.shields.io/pypi/pyversions/bioIOT" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/pypi/l/bioIOT" alt="License"></a>
</p>

<p align="center">
  <a href="README.md">English</a> | <b>简体中文</b>
</p>

---

**bioIOT** 求解半松弛逆最优传输问题：给定状态转移特征、源/靶状态质量与观测转移，学习特征权重 θ，使线性代价 `C = -einsum(φ, θ)` 诱导的软边际最优传输计划复现观测数据——并进一步给出状态转移矩阵、随机游走 pseudotime 与可直接发表的图件。

求解器在治疗耐药状态转移研究项目中开发并验证，现以通用工具包形式发布。

## 为什么选择 bioIOT？

- **可识别性由构造保证。** 硬边际 OT 下纯列特征不可识别；bioIOT 的 KL 软锚定在保留靶组成约束的同时恢复可识别性。
- **精确隐式梯度。** 前向为 Anderson 加速不动点迭代；反向用隐函数定理——在 unrolled 反传发散之处依然数值稳定。
- **对有限采样噪声去噪。** 50 次重复、每状态 30 细胞的基准中，bioIOT 恢复真实转移矩阵的精度约为直接使用噪声观测的 3 倍（平均逐行 L1 0.013 vs 0.039），特征权重相关 0.96。
- **真正即插即用。** numpy 进、numpy 出；输入自动校验与归一化；一行拟合，一行出图。

## 安装

```bash
pip install bioIOT
```

要求 Python ≥ 3.10。核心依赖：`numpy`、`torch`（CPU 即可）。可选：`matplotlib` 出图、`anndata` 接入 scanpy。

<details>
<summary>从源码安装</summary>

```bash
pip install git+https://github.com/XTSgreen/bioIOT-py.git
```

</details>

## 快速上手

```python
import bioiot

# 1) 准备场景：phi (K, K, F) 特征、a/b (K,) 状态质量、T (K, K) 观测行条件转移
sim = bioiot.simulate_iot_states(K=6, seed=1)   # 含真值的合成数据

# 2) 一行拟合特征权重——多重启 + 两阶段去偏
model = bioiot.IOTModel().fit(sim["phi"], sim["a"], sim["b"], sim["T_true"])
model.theta_        # 去偏系数
model.support_      # 支撑集掩码

# 3) 轨迹层
Q  = bioiot.transition_matrix(model)              # (K, K) 转移矩阵
pt = bioiot.pseudotime_from_transition(Q, root="S1")

# 4) 直接从细胞级数据出发（嵌入 + 聚类标签 + 两时间点）
res = bioiot.run_iot(sim["cell_embedding"], sim["cell_state"],
                     from_mask=sim["cell_time"] == "t0",
                     to_mask=sim["cell_time"] == "t1",
                     root="S1")
res["Q"]; res["pseudotime"]
# AnnData/scanpy: bioiot.run_iot_adata(adata, "state", "time", "t0", "t1")

# 5) 出图（matplotlib）
ax = bioiot.plot_transition_heatmap(Q)
ax = bioiot.plot_transition_flow(Q, sim["embedding"], threshold=0.05)
ax = bioiot.plot_theta(model)
```

## 函数式核心

需要底层求解器的用户（记号与论文一致）：

```python
C = bioiot.make_cost(phi, theta)                  # C = -einsum(phi, theta)
P = bioiot.soft_sinkhorn(C, a, b, mu=0.5, eps=1.0)  # (K, K) 计划，P @ 1 = a
P = bioiot.uot_plan(phi, theta, a, b)             # 构造代价 + 求解一步到位
loss = bioiot.row_ce_loss([T], [P], [a])
phi_z, meta = bioiot.zscore_phi(phi)
```

`soft_sinkhorn` 接受 numpy 数组或 torch 张量；返回的计划可经精确隐式微分对 `C` 求导。

## 方法原理

bioIOT 求解

```text
min_P  <C, P> − eps·H(P) + mu·KL(col(P) ‖ b)    s.t.  P·1 = a
```

源端行边际硬、列边际 KL 锚定：

- `mu → ∞` 退化为硬边际 OT（纯列特征不可识别）；
- `mu → 0` 退化为普通行 softmax（失去靶组成锚定）;
- 有限 `mu` 插值两者——论文工作点为 `mu = 0.5, eps = 1.0, lam = 0.05`。

拟合（`IOTModel`）使用 Adam + 精确隐式梯度、l1 选择后 Bühlmann 式支撑集去偏重拟合、多重启取最优。

## 测试

```bash
python -m unittest discover -s tests -v   # 仓库根目录
python -c "import bioiot; bioiot.self_test()"   # 数值自检
```

## 引用

如果 bioIOT 对你的研究有帮助，请引用：

```bibtex
@misc{dong2026bioiot,
  author       = {Dong, Han},
  title        = {bioIOT: Plug-and-Play Semi-Relaxed Inverse Optimal
                  Transport for Single-Cell State Transitions},
  year         = {2026},
  howpublished = {\url{https://github.com/XTSgreen/bioIOT-py}},
  note         = {Python package version 0.2.0}
}
```

## 许可证

[MIT](LICENSE) © 2026 Han Dong (XTSgreen)

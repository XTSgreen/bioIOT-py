# bioIOT (Python)

**即插即用**的半松弛逆最优传输（semi-relaxed Inverse Optimal Transport）拟合器——论文核心求解器 `script/uot_fitter.py` 的独立发行版，数值实现与论文完全一致：Anderson 加速不动点前向求解 + 隐式微分（隐函数定理）精确梯度。

## 安装

```bash
pip install bioIOT            # 发布后
# 或本地安装（本项目内）
pip install -e packages/bioIOT-py
```

仅依赖 `numpy` 与 `torch`（可选：`matplotlib` 出图、`anndata`/scanpy 接口），Python >= 3.10，CPU 即可运行。

## 60 秒上手

```python
import numpy as np
import bioiot

# 1) 估计器 API：一行拟合（多场景传 list；各场景 K 可不同）
model = bioiot.IOTModel().fit(phi, a, b, T)
model.theta_       # 去偏特征权重
model.support_     # 支撑集掩码
model.loss_; model.restart_losses_

# 2) 轨迹层：状态转移矩阵 + 随机游走 pseudotime
Q = bioiot.transition_matrix(model)
pt = bioiot.pseudotime_from_transition(Q, root="S1")

# 3) 单细胞接口（细胞嵌入 + 状态标签 + 两时间点掩码）
sim = bioiot.simulate_iot_states(seed=1)          # 一行生成含真值的合成数据
res = bioiot.run_iot(sim["cell_embedding"], sim["cell_state"],
                     from_mask=sim["cell_time"] == "t0",
                     to_mask=sim["cell_time"] == "t1", root="S1")
res["Q"]; res["pseudotime"]
# AnnData/scanpy: bioiot.run_iot_adata(adata, state_col=..., time_col=..., from_key="t0", to_key="t1")

# 4) 可视化（matplotlib）
ax = bioiot.plot_transition_heatmap(Q)
ax = bioiot.plot_transition_flow(Q, sim["embedding"], threshold=0.05)
ax = bioiot.plot_theta(model)
```

底层函数式 API（与论文记号一一对应）：

```python
C = bioiot.make_cost(phi, theta)               # C = -einsum(phi, theta)
P = bioiot.soft_sinkhorn(C, a, b, mu=0.5, eps=1.0)
P = bioiot.uot_plan(phi, theta, a, b)
loss = bioiot.row_ce_loss([T], [P], [a])
phi_z, meta = bioiot.zscore_phi(phi)
```

安装自检（隐式梯度 vs 有限差分 + θ 可恢复性）：

```python
bioiot.self_test()
```

## 方法概要

求解行边际硬、列边际 KL 软锚定的半松弛问题：

```
min_P <C,P> - eps·H(P) + mu·KL(col(P) || b),   s.t.  P·1 = a
```

- `mu -> ∞` 退化为硬边际 OT（纯列特征不可识别）
- `mu -> 0` 退化为行 softmax（失去靶组成锚定）
- 有限 `mu` 插值两者：保留靶状态分布约束同时允许偏离，纯列特征恢复可识别性
- 默认超参 `mu=0.5, eps=1.0, lam=0.05` 为论文合成校准选定工作点

## 测试

```bash
python -m unittest discover -s packages/bioIOT-py/tests -v
```

# AI-Native 6G 网络仿真与多模态数据生成框架

> 面向自监督对比学习 (Self-Supervised Contrastive Learning) 与多维资源编排研究

## 1. 设计哲学与顶会标准对齐

为了满足顶级会议对数据**“真实性”、“复杂性”**和**“可复现性”**的要求，本仿真框架（Simulator）超越了传统的日志记录模式，采用 **“状态快照（State Snapshot）”** 的生成范式。

### 1.1 核心设计原则

1. **算网深度融合 (Compute-Network Convergence):** 摒弃仅关注带宽的传统网络仿真，引入 CPU/GPU/内存 维度，模拟计算卸载（Offloading）场景。
2. **空天地一体化动态性 (NTN Dynamics):** 引入低轨卫星（LEO）和无人机（UAV）节点，模拟拓扑的时变性（Time-varying Topology）。
3. **为 AI 而生 (AI-Native):** 仿真器原生支持生成 **<Anchor, Positive, Negative>** 三元组数据，直接服务于对比学习模型的训练，无需二次清洗。

## 2. 仿真架构设计 (Python Stack)

**推荐技术栈：**

- **核心引擎:** `SimPy` (离散事件仿真，支持高并发流模拟)
- **图结构:** `NetworkX` (处理复杂拓扑)
- **数值计算:** `NumPy` & `SciPy` (生成长尾分布和相关性数据)
- **数据存储:** `HDF5` 或 `JSONL` (高性能存储大规模结构化数据)

### 2.1 模块一：分层异构拓扑生成器 (Hierarchical Heterogeneous Topology)

6G 网络不是扁平的，而是分层的。我们需要构建一个 **Cloud-Edge-Device-Satellite** 四层立体架构。

**实现逻辑：**

1. **核心层 (Core/Cloud):** 使用 `barabasi_albert_graph` 生成无标度网络，模拟少数具备超强算力的核心数据中心。
2. **传输层 (Transport):** 使用 `watts_strogatz_graph` 生成小世界网络，模拟光纤骨干网。
3. **接入层 (RAN/Edge):** 使用 `random_geometric_graph` 基于地理位置连接，模拟基站覆盖。
4. **非地面层 (NTN):**
   - **卫星 (Satellites):** 依据简化的轨道模型（如每 $T$ 秒切换一次邻居列表），动态重写与地面站的 `Edge`。
   - **无人机 (UAVs):** 作为移动的中继节点或边缘计算节点。

### 2.2 模块二：多维资源属性注入 (Multi-dimensional Attribute Injection)

**顶会关键点：** 资源属性不能是独立同分布（I.I.D.）的。高性能计算节点通常也拥有高带宽。

**实现逻辑：**

- 使用 **多元高斯分布 (Multivariate Gaussian)** 生成具有相关性的属性，然后映射到正值区间（如 Pareto 分布）。

  $$\begin{pmatrix} \text{CPU} \\ \text{Bandwidth} \\ \text{Memory} \end{pmatrix} \sim \mathcal{N}(\mu, \Sigma), \quad \text{where } \Sigma \text{ represents correlation.}$$

- **节点类型标签 (One-hot Encoding):** 为 GNN 训练准备，标记节点是 `MEC Server` 还是 `IoT Device`。

### 2.3 模块三：SLA 感知的业务流引擎 (SLA-aware Traffic Engine)

6G 业务具有极强的异构性。仿真器需生成携带 **意图 (Intent)** 的流量。

**业务类别定义：**

| 业务类型               | 关键 SLA 指标 | 到达模式 (Arrival Pattern) | 资源消耗特征       |
| ---------------------- | ------------- | -------------------------- | ------------------ |
| **URLLC** (如工业控制) | 时延 < 1ms    | 周期性 + 微小抖动          | 低带宽，高CPU中断  |
| **eMBB** (如全息通信)  | 带宽 > 1Gbps  | 泊松过程 (Poisson)         | 持续高带宽，高内存 |
| **mMTC** (如环境监测)  | 连接数密度    | 批次突发 (Burst)           | 极低资源，海量并发 |

### 2.4 模块四：数据增强与对比样本生成 (Data Augmentation for Contrastive Learning)

这是本方案最核心的创新点。在仿真运行的每个时间步（Time Step），直接生成用于训练的**正样本对**。

- **View 1 (Global Graph):** 完整的网络拓扑状态矩阵 $A$ 和特征矩阵 $X$。
- **View 2 (Temporal Statistics):** 过去 $W$ 秒内的流量统计特征（均值、方差）。
- **Augmented View (Robustness):**
  - **Masking:** 随机将 10% 节点的属性置零（模拟监控数据丢失）。
  - **Perturbation:** 给链路时延增加高斯噪声 $\epsilon \sim \mathcal{N}(0, \sigma^2)$（模拟网络抖动）。

## 3. Python 仿真代码骨架 (核心逻辑)

以下代码展示了如何将上述理念转化为可运行的 Python 代码结构。

```python
import simpy
import networkx as nx
import numpy as np
import random
import json
from dataclasses import dataclass, asdict

# === 配置参数 ===
CONFIG = {
    "num_core_nodes": 5,
    "num_edge_nodes": 20,
    "num_satellites": 3,
    "simulation_time": 1000,
    "snapshot_interval": 1.0,  # 每秒采样一次
    "correlation_factor": 0.8  # 算力与带宽的相关性
}

@dataclass
class ResourceState:
    """定义节点的瞬时状态，用于生成 Feature Matrix"""
    cpu_total: float
    cpu_used: float
    bw_total: float
    bw_used: float
    queue_length: int
    active_flows: int
    node_type: str  # 'cloud', 'edge', 'satellite'

class SixG_Simulator:
    def __init__(self, env):
        self.env = env
        self.graph = nx.Graph()
        self.nodes_state = {}
        self._init_topology()
        self._init_resources()

    def _init_topology(self):
        """构建分层异构拓扑"""
        # 1. Core Network (Barabasi-Albert)
        core_g = nx.barabasi_albert_graph(CONFIG["num_core_nodes"], 2)
        # 2. Edge Network (Random Geometric)
        edge_g = nx.random_geometric_graph(CONFIG["num_edge_nodes"], 0.3)
        
        # 3. 合并并添加连接 (Backhaul links)
        self.graph = nx.compose(core_g, edge_g)
        # ... (此处省略添加跨层连接和卫星节点的代码) ...
        
        # 初始化节点状态容器
        for n in self.graph.nodes():
            self.nodes_state[n] = ResourceState(0,0,0,0,0,0, "unknown")

    def _init_resources(self):
        """基于相关性分布初始化资源容量"""
        # 模拟 CPU 和 带宽 的正相关性
        mean = [100, 1000] # CPU, BW
        cov = [[20, 15], [15, 50]] # 协方差矩阵
        resources = np.random.multivariate_normal(mean, cov, self.graph.number_of_nodes())
        
        for i, n in enumerate(self.graph.nodes()):
            # 确保资源非负，并赋予长尾特性
            cpu_cap = max(10, resources[i][0] * np.random.pareto(3.0))
            bw_cap = max(100, resources[i][1] * np.random.pareto(3.0))
            
            self.nodes_state[n].cpu_total = cpu_cap
            self.nodes_state[n].bw_total = bw_cap
            # 简单的类型标记逻辑
            if cpu_cap > 200: self.nodes_state[n].node_type = 'cloud'
            else: self.nodes_state[n].node_type = 'edge'

    def update_ntn_topology(self):
        """模拟卫星运动导致的拓扑变化"""
        # 在实际代码中，这里根据时间 t 计算卫星可见性并重连边
        if self.env.now % 10 == 0:
            # 示例：随机重连卫星链路
            pass

    def get_multimodal_snapshot(self):
        """核心：生成多模态、成对的数据"""
        
        # 1. 图模态 (Graph View) - 适配 GNN
        node_feats = []
        for n in self.graph.nodes():
            s = self.nodes_state[n]
            # 特征向量: [CPU容量, CPU利用率, BW容量, BW利用率, 类型OneHot...]
            feat = [s.cpu_total, s.cpu_used / s.cpu_total, s.bw_total, s.bw_used / s.bw_total]
            node_feats.append(feat)
        
        adj_matrix = nx.adjacency_matrix(self.graph).todense().tolist()
        
        # 2. 时序模态 (Temporal View) - 适配 Transformer
        # 这里应该返回过去窗口的统计值，此处简化为当前全局统计
        global_load = np.mean([f[1] for f in node_feats])
        
        # 3. 数据增强 (Augmented View) - 用于对比学习的正样本
        # 策略：随机 Mask 掉 20% 的节点特征
        masked_node_feats = np.array(node_feats)
        mask_indices = np.random.choice(len(node_feats), int(len(node_feats)*0.2), replace=False)
        masked_node_feats[mask_indices] = 0
        
        return {
            "timestamp": self.env.now,
            "modality_graph": {
                "x": node_feats,
                "adj": adj_matrix
            },
            "modality_seq": {
                "global_load": global_load
            },
            "modality_aug": {
                "x": masked_node_feats.tolist()
            }
        }

def traffic_generator(env, sim):
    """生成符合潮汐效应的 SLA 业务"""
    while True:
        # 模拟正弦波潮汐流量
        arrival_rate = 10 + 5 * np.sin(env.now * 0.05)
        yield env.timeout(np.random.exponential(1.0 / arrival_rate))
        
        # 业务逻辑：随机选择源和目的，扣减沿途资源
        # ... (省略具体路由和资源扣减代码) ...
        # 更新 self.nodes_state 中的 used 字段

def snapshot_recorder(env, sim, filename="6g_dataset.jsonl"):
    with open(filename, 'w') as f:
        while True:
            yield env.timeout(CONFIG["snapshot_interval"])
            
            # 1. 更新动态拓扑 (NTN)
            sim.update_ntn_topology()
            
            # 2. 获取多模态快照
            snapshot = sim.get_multimodal_snapshot()
            
            # 3. 写入文件
            f.write(json.dumps(snapshot) + "\n")

# === 运行仿真 ===
env = simpy.Environment()
sim = SixG_Simulator(env)
env.process(traffic_generator(env, sim))
env.process(snapshot_recorder(env, sim))
env.run(until=CONFIG["simulation_time"])
```
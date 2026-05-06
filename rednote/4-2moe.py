import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# FFN（SwiGLU）
# 两路并行：Gate 路经 SiLU 激活，Up 路线性升维，逐元素相乘后降维
# ============================================================
class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        # 门控路：升维后经 SiLU 激活
        self.Wgate = nn.Linear(d_model, d_ffn, bias=False)
        # Up 路：直接线性升维
        self.Wup   = nn.Linear(d_model, d_ffn, bias=False)
        # Down 路：降回 d_model
        self.Wdown = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x: torch.Tensor):
        # x: [bsz, seq, d_model]
        gate = F.silu(self.Wgate(x))   # [bsz, seq, d_ffn]，门控值
        up   = self.Wup(x)             # [bsz, seq, d_ffn]，线性升维
        return self.Wdown(gate * up)   # 逐元素相乘后降维

# ============================================================
# MoE（Mixture of Experts）
# 多个 FFN 专家，Router 为每个 token 选 top-k 个专家加权融合
# ============================================================
class MoE(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, num_experts: int, top_k: int):
        super().__init__()
        assert top_k <= num_experts
        self.num_experts = num_experts # 总专家数
        self.top_k       = top_k # 每个输入选几个专家

        # 每个专家是一个独立的 SwiGLU FFN
        self.experts = nn.ModuleList([
            SwiGLUFFN(d_model, d_ffn) for _ in range(num_experts)
        ])
        # Router：线性层，为每个 token 输出各专家的路由得分
        self.router = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor):
        # x: [bsz, seq, d_model]
        bsz, seq, d_model = x.shape
        # 将 token 展平方便路由：[bsz*seq, d_model]
        x_flat = x.view(-1, d_model)

        # 1. Router 打分，选出 top-k 专家
        router_logits  = self.router(x_flat)                         # [N, num_experts]
        # router_weights, expert_indices 分别是专家权重和对应索引
        router_weights, expert_indices = torch.topk(
            F.softmax(router_logits, dim=-1), self.top_k, dim=-1
        )                                                             # 各 [N, top_k]
        # 对选中的权重重新归一化，使其和为 1
        router_weights = router_weights / router_weights.sum(dim=-1, keepdim=True)

        # 2. 每个 token 加权融合 top-k 专家的输出，每个token 循环 top-k 次
        output = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            # 逐个选取候选的 topk 个专家：取第 k 个专家索引和对应权重
            idx     = expert_indices[:, k]        # [N]
            weights = router_weights[:, k]        # [N]
            # 按专家分组计算，避免对每个 token 单独调用
            for e in range(self.num_experts):
                mask = (idx == e)                 # 哪些 token 路由到专家 e，mask 布尔数组[true,false,true,true]
                if mask.any():
                    expert_out       = self.experts[e](x_flat[mask].unsqueeze(0))
                    output[mask]    += weights[mask].unsqueeze(-1) * expert_out.squeeze(0)
        return output.view(bsz, seq, d_model)
    


'''
如何理解下面这行
output[mask]    += weights[mask].unsqueeze(-1) * expert_out.squeeze(0)

用最具体的数字，`e=1` 那轮（token0 和 token2 选了专家1）：

mask    = [T, F, T, F]
weights = [0.6, 0.7, 0.5, 0.8]

### 第一步：`weights[mask]`
weights = [0.6, 0.7, 0.5, 0.8]
mask    = [ T,   F,   T,   F ]
weights[mask] = [0.6, 0.5]   # 只取 True 位置的权重，shape [2]

### 第二步：`.unsqueeze(-1)`
[0.6, 0.5]          # shape [2]
      ↓
[[0.6],
 [0.5]]             # shape [2, 1]

为什么要变成列向量？因为后面要和 `expert_out` 相乘：
expert_out  shape [2, 4]   # 2个token，每个4维
weights     shape [2, 1]   # 必须是[2,1]才能广播

### 第三步：`expert_out.squeeze(0)`
expert_out             # shape [1, 2, 4]，FFN输出自带batch维
expert_out.squeeze(0)  # shape [2, 4]，去掉batch维
# 具体数值假设：
[[0.1, 0.2, 0.3, 0.4],   # 专家1处理token0的结果
 [0.5, 0.6, 0.7, 0.8]]   # 专家1处理token2的结果

### 第四步：相乘（广播）
[[0.6],   *   [[0.1, 0.2, 0.3, 0.4],
 [0.5]]        [0.5, 0.6, 0.7, 0.8]]

# [2,1] 广播到 [2,4]，每行乘自己的权重：
= [[0.6*0.1, 0.6*0.2, 0.6*0.3, 0.6*0.4],   # token0的结果 × 0.6
   [0.5*0.5, 0.5*0.6, 0.5*0.7, 0.5*0.8]]   # token2的结果 × 0.5
= [[0.06, 0.12, 0.18, 0.24],
   [0.25, 0.30, 0.35, 0.40]]

### 第五步：`output[mask] +=`
output = [[0, 0, 0, 0],   # token0
          [?, ?, ?, ?],   # token1（已被专家0更新过）
          [0, 0, 0, 0],   # token2
          [0, 0, 0, 0]]   # token3

mask = [T, F, T, F]
# output[mask] 就是 token0行 和 token2行
output[mask] += [[0.06, 0.12, 0.18, 0.24],
                 [0.25, 0.30, 0.35, 0.40]]

# 结果：
output = [[0.06, 0.12, 0.18, 0.24],   # token0 ← 更新了
          [?,    ?,    ?,    ?   ],    # token1 ← 没动
          [0.25, 0.30, 0.35, 0.40],   # token2 ← 更新了
          [0,    0,    0,    0   ]]    # token3 ← 没动
```

---

### 一张图看清全貌

```
weights[mask]         expert_out.squeeze(0)
  shape[2,1]      ×       shape[2,4]
                          
  [[0.6],             [[0.1, 0.2, 0.3, 0.4],
   [0.5]]              [0.5, 0.6, 0.7, 0.8]]
     ↓                          ↓
  每行是一个token          每行是该token经过专家的输出
  的路由权重               
                  ↓ 广播相乘
          [[0.06, 0.12, 0.18, 0.24],
           [0.25, 0.30, 0.35, 0.40]]
                  ↓ += 写回
          output 中 mask=True 的那几行
```



'''
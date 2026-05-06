import torch
import torch.nn.functional as F

def sft_loss(logits, labels, ignore_index=-100):
    # logits: [bsz, seq, vocab_size]  模型输出
    # labels: [bsz, seq]              真实下一个token，-100表示不计算loss的位置

    bsz, seq, vocab_size = logits.shape

    # 移位：用前seq-1个位置的logits，预测后seq-1个token
    shift_logits = logits[:, :-1, :]   # [bsz, seq-1, vocab_size]
    shift_labels = labels[:, 1:]       # [bsz, seq-1]

    # 展平，方便计算交叉熵
    shift_logits = shift_logits.reshape(-1, vocab_size)   # [bsz*(seq-1), vocab_size]
    shift_labels = shift_labels.reshape(-1)               # [bsz*(seq-1)]

    # mask：ignore_index=-100的位置不计算loss（prompt部分）
    mask = (shift_labels != ignore_index).float()         # [bsz*(seq-1)]

    # 把ignore_index替换成0，防止gather时越界
    shift_labels = shift_labels.clamp(min=0)

    # safe softmax
    logits_max = shift_logits.max(dim=-1, keepdim=True).values
    shift_logits = shift_logits - logits_max

    # log_softmax
    log_sum_exp = torch.log(torch.exp(shift_logits).sum(dim=-1))        # [N]
    logits_true = shift_logits[torch.arange(len(shift_labels)), shift_labels]  # [N]
    log_softmax = logits_true - log_sum_exp                              # [N]

    # mask mean：只对非ignore位置求均值
    loss = -(log_softmax * mask).sum() / mask.sum()
    return loss


# 验证
bsz, seq, vocab_size = 2, 6, 100
logits = torch.randn(bsz, seq, vocab_size)
labels = torch.tensor([
    [-100, -100, 23, 57, 82, 14],   # 前两个是prompt，不计算loss
    [-100, -100, -100, 41, 9, 33],  # 前三个是prompt，不计算loss
])

print("sft_loss:", sft_loss(logits, labels))
print("torch校验:", F.cross_entropy(
    logits[:, :-1, :].reshape(-1, vocab_size),
    labels[:, 1:].reshape(-1),
    ignore_index=-100
))
'''
### 移位逻辑图示
```
输入tokens:  [<bos>, A,  B,  C,  D,  E]
                ↓    ↓   ↓   ↓   ↓   ↓
logits:      [ l0,  l1, l2, l3, l4, l5]

shift_logits:[ l0,  l1, l2, l3, l4]     # 去掉最后一个
shift_labels:[  A,   B,  C,  D,  E]     # 去掉第一个

# 含义：用l0预测A，用l1预测B，用l2预测C...
```

---

### ignore_index 的作用
```
labels:       [-100, -100, 23, 57, 82, 14]
                 ↑     ↑
              prompt部分不监督，只监督回答部分
'''


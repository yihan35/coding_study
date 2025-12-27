import torch
def freqs(dim,seq,theta=10000)->torch.Tensor:
    freqs_base=1.0/(theta** (torch.arange(0,dim,2,dtype=torch.float)/dim))# 默写 dtype=torch.float
    position=torch.arange(seq,dtype=torch.float)
    angles = torch.outer(position,freqs_base)
    # 极坐标的写法，模长为 1，角度为 angles
    freqs_cis = torch.polar(torch.ones_like(angles),angles) # 默写 torch.ones_like 
    return freqs_cis
def apply_rope(xq,xk,freq_cis)->tuple[torch.Tensor,torch.Tensor]:
    # shape:[bsz,seq,num_head,dim]->[bsz,seq,num_head,dim/2,2]
    xq_reshaped = xq.float().reshape(*xq.shape[:-1],-1,2) # 默写这句，先转为 fp32，shape[:-1]这是 python 的切片语法，取倒数第一个元素之前，即(bsz,seq,num_head)，* 是 Python 的 “解包运算符”，将元组变为独立参数bsz,seq,num_head，2：固定拆分的最后一维，-1：让 PyTorch 自动计算该维度的大小
    xk_reshaped = xk.float().reshape(*xq.shape[:-1],-1,2)
    # shape:[bsz,seq,num_head,dim/2]
    xq_complex = torch.view_as_complex(xq_reshaped) # 默写：转换为复数张量
    xk_complex = torch.view_as_complex(xk_reshaped)
    # 输入的频率张量 shape:[seq,dim/2],需要修改以适应 xq_complex
    seq = xq_complex.shape[1]
    freq_cis = freq_cis[:seq].reshape(1,seq,1,-1) # 默写：reshape 比 view 更好，一般情况直接使用 reshape
    # 向量旋转 shape:[bsz,seq,num_head,dim/2]
    xq_rotated = xq_complex * freq_cis
    xk_rotated = xk_complex * freq_cis
    # 将复数转回实数,shape:[bsz,seq,num_head,dim/2,2]
    # flatten(3): 核心是把张量从第 3 维（维度索引从 0 开始）到最后一维的所有维度 “合并” 成一个维度
    # shape:[bsz,seq,num_head,dim]
    xq_out = torch.view_as_real(xq_rotated).flatten(3)
    xk_out = torch.view_as_real(xk_rotated).flatten(3)
    # 输入 xq 的 dtype 可能是 float16/bfloat16（大模型常用的低精度，节省显存 / 提升速度）；
    # 计算后的 xq_out 是 float32 类型；
    return xq_out.type_as(xq),xk_out.type_as(xk)

bsz,seq,num_head,d = 1,2,4,8
freqs_cis = freqs(d,seq)
xq = torch.randn(bsz,seq,num_head,d)
print("旋转之前的维度：",xq.shape)
xk = torch.randn(bsz,seq,num_head,d)
xq_out,xk_out = apply_rope(xq,xk,freqs_cis)
print("旋转之后的维度：",xq_out.shape)
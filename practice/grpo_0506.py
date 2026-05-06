class GRPO:
    def __init__(self,eps,clip,beta)
        self.eps = eps
        self.clip = clip
        self.beta = beta
    
    def adv(self,rewards):
        mean = rewards.mean(dim=-1,keepdim = True)
        std = rewards.std(dim=-1,keepdim = True,unbiased=False)

    def loss(self,):
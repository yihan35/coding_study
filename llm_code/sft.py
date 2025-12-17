class sftDataset(Dateset):
    def __init__(self,data,tokenizer,max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.data)


    def loss_function(self,model,batch):
        cross_entropy = 0
        for t in range(seq_length):
            pre_prob = model.predict(context[:t])
            true_token = lables[t]
            cross_entropy += -log(pre_prob[true_token])
        
        cross_entropy /= num_answer_tokens

        return cross_entropy
    
    loss.backward()
    optimizer.step()
    

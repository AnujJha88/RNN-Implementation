import numpy as np
class EmbedLayer:
    def __init__(self,vocab_size,embed_dim):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embeddings = np.random.randn(vocab_size,embed_dim)*0.01
        self.dE=None
    def forward(self,input_indices):
        return self.embeddings[input_indices]
        
    def backward(self,dout,indices):
        if(self.dE is None):
            self.dE=np.zeros_like(self.embeddings)
        for i,idx in enumerate(indices):
            self.dE[idx]+=dout[i]
    
    def zero_grad(self):
        self.dE=np.zeros_like(self.embeddings)  
    
    def step(self,learning_rate=0.01):
        self.embeddings-=learning_rate*self.dE
        self.zero_grad()
import numpy as np
from gate import UpdateGate,ResetGate,CandidateGate

class GRU:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size

        self.update_gate=UpdateGate()
        self.reset_gate=ResetGate()
        self.candidate_gate=CandidateGate()

    def zero_grad(self):
        self.update_gate.zero_grad()
        self.reset_gate.zero_grad()
        self.candidate_gate.zero_grad()

    def forward(self,input_seq,h_0):
        seq_len=input_seq.shape[0]

        h_dim=self.hidden_size
        h_seq=np.zeros((seq_len,h_dim))
        if h_0 is None:
            h_prev=np.zeroes(hidden_size)
        else:
            h_prev=h_0

        for t in range(seq_len):
            x_t=input_seq[t]
            z_t=self.update_gate.forward(x_t,h_prev)
            r_t=self.reset_gate.forward(x_t,h_prev)

            h_tilde=self.candidate_gate.forward(x_t,r_t*h_prev)
            h_t=(1-z_t)*h_prev+z_t*h_tilde
            h_seq[t]=h_t
            h_prev=h_t
        return h_seq

    def backward(self,gradients):
        pass

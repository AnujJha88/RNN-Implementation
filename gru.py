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

        self.update_gate.init_weights(input_size, hidden_size)
        self.reset_gate.init_weights(input_size, hidden_size)
        self.candidate_gate.init_weights(input_size, hidden_size)

        self.Why=np.random.randn(output_size, hidden_size)*np.sqrt(1/(hidden_size+output_size))
        self.by=np.zeros((output_size, 1))
    def zero_grad(self):
        self.update_gate.zero_grad()
        self.reset_gate.zero_grad()
        self.candidate_gate.zero_grad()
        self.dWhy=np.zeros_like(self.Why)
        self.dby=np.zeros_like(self.by)

    def forward(self,x,h_prev):
        z_t=self.update_gate.forward(x,h_prev)
        r_t=self.reset_gate.forward(x,h_prev)
        h_tilde=self.candidate_gate.forward(x,r_t*h_prev)
        h_t=(1-z_t)*h_prev+z_t*h_tilde
        y_t=np.dot(self.Why,h_t)+self.by
        gates=(z_t,r_t,h_tilde)
        return h_t,y_t,gates
       

    def backward(self,dh_next,dy,h_t,x,h_prev,z,r,h_tilde):

        dWhy=np.dot(dy,h_t.T)
        dby = np.sum(dy, axis=1, keepdims=True)
        dh = np.dot(self.Why.T, dy) + dh_next
        dz=dh*(-h_prev)+h_tilde*dh
        dh_tilde=dh*z
        dh_prev= dh*(1 - z)
        dx_z, dh_z, dWz, dUz, dbz = self.update_gate.backward(dz, 0, 0)
        dx_c, d_rh, dWc, dUc, dbc = self.candidate_gate.backward(dh_tilde, 0, 0)
        dh_prev+=d_rh*r
        dr=d_rh*h_prev
        dx_r, dh_r, dWr, dUr, dbr=self.reset_gate.backward(dr, 0, 0)
        dx=dx_z+dx_c+dx_r
        dh_prev+=dh_z+dh_r
        self.update_gate.dWz+=dWz
        self.candidate_gate.dWc+=dWc
        self.reset_gate.dWr+=dWr
        self.update_gate.dUz+=dUz
        self.candidate_gate.dUc+=dUc    
        self.reset_gate.dUr+=dUr
        self.update_gate.dbz+=dbz
        self.candidate_gate.dbc+=dbc    
        self.reset_gate.dbr+=dbr
        self.dWhy+=dWhy
        self.dby+=dby

        return dx, dh_prev
  
    def step(self,learning_rate=0.01):
        self.update_gate.Wz-=learning_rate*self.update_gate.dWz
        self.update_gate.Uz-=learning_rate*self.update_gate.dUz
        self.update_gate.bz-=learning_rate*self.update_gate.dbz
        self.candidate_gate.Wc-=learning_rate*self.candidate_gate.dWc
        self.candidate_gate.Uc-=learning_rate*self.candidate_gate.dUc
        self.candidate_gate.bc-=learning_rate*self.candidate_gate.dbc
        self.reset_gate.Wr-=learning_rate*self.reset_gate.dWr
        self.reset_gate.Ur-=learning_rate*self.reset_gate.dUr
        self.reset_gate.br-=learning_rate*self.reset_gate.dbr
        self.by-=learning_rate*self.dby
        self.Why-=learning_rate*self.dWhy
        self.zero_grad()
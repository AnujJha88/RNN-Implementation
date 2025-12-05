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

        self.Why=np.random.randn(output_size, hidden_size)*np.sqrt(1/(hidden_size+output_size))
        self.by=np.zeros((output_size, 1))
    def zero_grad(self):
        self.update_gate.zero_grad()
        self.reset_gate.zero_grad()
        self.candidate_gate.zero_grad()

    def forward(self,x,h_prev):
        z_t=self.update_gate.forward(x,h_prev)
        r_t=self.reset_gate.forward(x,h_prev)
        h_tilde=self.candidate_gate.forward(x,r_t*h_prev)
        h_t=(1-z_t)*h_prev+z_t*h_tilde
        y_t=np.dot(self.Why,h_t)+self.by
        return h_t,y_t
       

    def backward(self,dh_next,dy,h_t,x):

        dWhy=np.dot(dy,h_t.T)
        dby = np.sum(dy, axis=1, keepdims=True)
        dh = np.dot(self.Why.T, dy) + dh_next
        x, h_prev, z = self.update_gate.cache
        _, _, r = self.reset_gate.cache
        _, _, h_tilde = self.candidate_gate.cache

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
        return dx, dh_prev, dWz, dWc, dWr, dUz, dUc, dUr, dbz, dbc, dbr
        # so what we need to do is basically find the backprop equation after it
    # has passed through the gates we added above

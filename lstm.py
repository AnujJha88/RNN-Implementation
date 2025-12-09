import numpy as np
from activation import Tanh
from gate import InputGate, ForgetGate, CandidateGate, OutputGate
class LSTM:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        
        self.input_gate=InputGate()
        self.forget_gate=ForgetGate()
        self.candidate_gate=CandidateGate()
        self.output_gate=OutputGate()
        
        self.input_gate.init_weights(input_size, hidden_size)
        self.forget_gate.init_weights(input_size, hidden_size)
        self.candidate_gate.init_weights(input_size, hidden_size)
        self.output_gate.init_weights(input_size, hidden_size)
        
        self.Why=np.random.randn(output_size, hidden_size)*np.sqrt(1/(hidden_size+output_size))
        self.by=np.zeros((1,output_size))

        self.zero_grad()
    def zero_grad(self):
        self.input_gate.zero_grad()
        self.forget_gate.zero_grad()
        self.candidate_gate.zero_grad()
        self.output_gate.zero_grad()
        self.dWhy=np.zeros_like(self.Why)
        self.dby=np.zeros_like(self.by)

    def forward(self, x, h_prev, c_prev):
        """
        Forward pass for LSTM cell.

        Args:
            x: Input at current timestep (batch_size, input_size)
            h_prev: Previous hidden state (batch_size, hidden_size)
            c_prev: Previous cell state (batch_size, hidden_size)

        Returns:
            h: Hidden state at current timestep (batch_size, hidden_size)
            c: Cell state at current timestep (batch_size, hidden_size)
            y: Output at current timestep (batch_size, output_size)
        """
        f = self.forget_gate.forward(x, h_prev, c_prev)
        i = self.input_gate.forward(x, h_prev, c_prev)
        c_tilde = self.candidate_gate.forward(x, h_prev)
        o = self.output_gate.forward(x, h_prev, c_prev)
        c = f * c_prev + i * c_tilde
        h = o * Tanh.forward(c)
        y = np.dot(h,self.Why.T) + self.by
        gates=(f,i,o,c_tilde)
        return h, c, y ,gates
    def backward(self, dh_next, dy, h_prev, c_prev, x, h, c,dc_next,o,i,c_tilde,f):
        dWhy=np.dot(dy.T,h)
        dh=dh_next
        dby=np.sum(dy,axis=1,keepdims=True)
        dh+=np.dot(dy,self.Why)
        do=dh*Tanh.forward(c)
        dc=dc_next+o*dh*Tanh.backward(c)
        df=dc*c_prev
        di=dc*c_tilde
        dc_tilde=dc*i
        dc_prev=dc*f

        dxf,dhf,dWf,dUf,dbf=self.forget_gate.backward(df,0,0,0)
        dxi,dhi,dWi,dUi,dbi=self.input_gate.backward(di,0,0,0)
        dxc_tilde,dhc_tilde,dWc,dUc,dbc=self.candidate_gate.backward(dc_tilde,0,0)
        dxo,dho,dWo,dUo,dbo=self.output_gate.backward(do,0,0,0)

        dx=dxf+dxi+dxc_tilde+dxo
        dh_prev=dhf+dhi+dhc_tilde+dho
        self.forget_gate.dWf+=dWf
        self.input_gate.dWi+=dWi
        self.output_gate.dWo+=dWo
        self.forget_gate.dUf+=dUf
        self.input_gate.dUi+=dUi
        self.output_gate.dUo+=dUo
        self.forget_gate.dbf+=dbf
        self.input_gate.dbi+=dbi
        self.output_gate.dbo+=dbo
        self.candidate_gate.dWc+=dWc
        self.candidate_gate.dUc+=dUc
        self.candidate_gate.dbc+=dbc
        self.dWhy+=dWhy
        self.dby+=dby
        return dx,dh_prev,dc_prev
    
    def step(self,learning_rate=0.01):
        self.input_gate.Wi-=learning_rate*self.input_gate.dWi
        self.forget_gate.Wf-=learning_rate*self.forget_gate.dWf
        self.output_gate.Wo-=learning_rate*self.output_gate.dWo
        self.candidate_gate.Wc-=learning_rate*self.candidate_gate.dWc
        self.input_gate.Ui-=learning_rate*self.input_gate.dUi
        self.forget_gate.Uf-=learning_rate*self.forget_gate.dUf
        self.output_gate.Uo-=learning_rate*self.output_gate.dUo
        self.candidate_gate.Uc-=learning_rate*self.candidate_gate.dUc
        self.forget_gate.bf-=learning_rate*self.forget_gate.dbf
        self.input_gate.bi-=learning_rate*self.input_gate.dbi
        self.output_gate.bo-=learning_rate*self.output_gate.dbo
        self.candidate_gate.bc-=learning_rate*self.candidate_gate.dbc
        self.by-=learning_rate*self.dby
        self.Why-=learning_rate*self.dWhy
        self.zero_grad()
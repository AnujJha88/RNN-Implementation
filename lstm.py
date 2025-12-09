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
        self.forget_gate.dWf += dWf
        self.forget_gate.dUf += dUf
        self.forget_gate.dbf += dbf
        
        self.input_gate.dWi += dWi
        self.input_gate.dUi += dUi
        self.input_gate.dbi += dbi
        
        self.output_gate.dWo += dWo
        self.output_gate.dUo += dUo
        self.output_gate.dbo += dbo
        
        self.candidate_gate.dWc += dWc
        self.candidate_gate.dUc += dUc
        self.candidate_gate.dbc += dbc
        self.dWhy+=dWhy
        self.dby+=dby
        return dx,dh_prev,dc_prev
    
    def step(self, learning_rate=0.01):
        # 1. Update Input Gate (Clip gradients between -5 and 5)
        self.input_gate.Wi -= learning_rate * np.clip(self.input_gate.dWi, -5, 5)
        self.input_gate.Ui -= learning_rate * np.clip(self.input_gate.dUi, -5, 5)
        self.input_gate.bi -= learning_rate * np.clip(self.input_gate.dbi, -5, 5)

        # 2. Update Forget Gate
        self.forget_gate.Wf -= learning_rate * np.clip(self.forget_gate.dWf, -5, 5)
        self.forget_gate.Uf -= learning_rate * np.clip(self.forget_gate.dUf, -5, 5)
        self.forget_gate.bf -= learning_rate * np.clip(self.forget_gate.dbf, -5, 5)

        # 3. Update Output Gate
        self.output_gate.Wo -= learning_rate * np.clip(self.output_gate.dWo, -5, 5)
        self.output_gate.Uo -= learning_rate * np.clip(self.output_gate.dUo, -5, 5)
        self.output_gate.bo -= learning_rate * np.clip(self.output_gate.dbo, -5, 5)

        # 4. Update Candidate Gate
        self.candidate_gate.Wc -= learning_rate * np.clip(self.candidate_gate.dWc, -5, 5)
        self.candidate_gate.Uc -= learning_rate * np.clip(self.candidate_gate.dUc, -5, 5)
        self.candidate_gate.bc -= learning_rate * np.clip(self.candidate_gate.dbc, -5, 5)

        # 5. Update Output Layer
        self.Why -= learning_rate * np.clip(self.dWhy, -5, 5)
        self.by -= learning_rate * np.clip(self.dby, -5, 5)

        # 6. Clear gradients for the next step
        self.zero_grad()
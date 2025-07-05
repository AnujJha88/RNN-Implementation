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

        self.Why=np.random.randn(output_size, hidden_size)*np.sqrt(1/(hidden_size+output_size))
        self.by=np.zeros((output_size, 1))

        self.zero_grad()
    def zero_grad(self):
        self.input_gate.zero_grad()
        self.forget_gate.zero_grad()
        self.candidate_gate.zero_grad()
        self.output_gate.zero_grad()

    def forward(self, x, h_prev, c_prev):
        pass
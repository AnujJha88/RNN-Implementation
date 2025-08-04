class GRU:
    def __init__(self):
        self.forget_gate=ForgetGate()
        self.update_gate=UpdateGate()
        self.input=None
        self.output=None
        self.zero_grad()

    def zero_grad(self):
        self.forget_gate.zero_grad()
        self.update_gate.zero_grad()


import numpy as np
from activation import Sigmoid

class MulGate:
    def forward(self, W, x):
        """
        Forward pass for mul gate

        Args:
            W: Weight matrix
            x: Input

        Returns:
            Output of the multiplication
        """
        return np.dot(W, x)
    def backward(self, W, x, grad_output):
        """
        Backward pass for mul gate

        Args:
            W: Weight matrix
            x: Input
            grad_output: Gradient of output

        Returns:
            grad_W: Gradient of weight matrix
            grad_x: Gradient of input
        """
        grad__W=np.dot(grad_output, np.transpose(x))
        grad_x = np.dot(np.transpose(W), grad_output)
class AddGate:
    def forward(self, x, y):
        """
        Forward pass for add gate

        Args:
            x: Input 1
            y: Input 2

        Returns:
            output: Sum of input 1 and input 2
        """
        return x+y
    def backward(self, x, y, grad_output):
        """
        Backward pass for add gate

        Args:
            x: Input 1
            y: Input 2
            grad_output: Gradient of output

        Returns:
            grad_x: Gradient of input 1
            grad_y: Gradient of input 2
        """
        grad_x = grad_output
        grad_y = grad_output


class ForgetGate:  # LSTM   
    def __init__(self):
        self.Wf = None  
        self.Uf = None  
        self.bf = None 
        self.cache = None 
    
    def forward(self, x, h_prev, c_prev):
        """
        Forward pass for forget gate
        
        Args:
            x: Input at current timestep (batch_size, input_dim)
            h_prev: Hidden state from previous timestep (batch_size, hidden_dim)
            c_prev: Cell state from previous timestep (batch_size, hidden_dim)
            
        Returns:
            f: Forget gate activation (batch_size, hidden_dim)
        """
        # Linear transformation
        linear = np.dot(x, self.Wf.T) + np.dot(h_prev, self.Uf.T) + self.bf
        # Sigmoid activation
        f = Sigmoid.forward(linear)
        
        # Cache values for backward pass
        self.cache = (x, h_prev, f)
        
        return f
    
    def backward(self, df, dnext_h, dnext_c, dnext_f):
        """
        Backward pass for forget gate
        
        Args:
            df: Gradient of loss w.r.t forget gate output (batch_size, hidden_dim)
            dnext_h: Gradient from next hidden state (batch_size, hidden_dim)
            dnext_c: Gradient from cell state (batch_size, hidden_dim)
            dnext_f: Gradient from next forget gate (batch_size, hidden_dim)
            
        Returns:
            dx: Gradient w.r.t input (batch_size, input_dim)
            dh_prev: Gradient w.r.t previous hidden state (batch_size, hidden_dim)
            dWf: Gradient w.r.t weight matrix (hidden_dim, input_dim + hidden_dim)
            dUf: Gradient w.r.t hidden weight matrix (hidden_dim, hidden_dim)
            dbf: Gradient w.r.t bias (hidden_dim,)
        """
        x, h_prev, f = self.cache
        
        # Compute gradient through sigmoid
        dsigma = Sigmoid.backward(f) * (df + dnext_h + dnext_c)
        
        # Compute gradients for parameters
        dWf = np.dot(dsigma.T, np.hstack([x, h_prev]))
        dUf = np.dot(dsigma.T, h_prev)
        dbf = np.sum(dsigma, axis=0)
        
        # Compute gradients for input and previous hidden state
        dx = np.dot(dsigma, self.Wf[:, :x.shape[1]])
        dh_prev = np.dot(dsigma, self.Wf[:, x.shape[1]:])
        
        return dx, dh_prev, dWf, dUf, dbf
class InputGate:  # LSTM
    def __init__(self):
        self.Wi = None  
        self.Ui = None  
        self.bi = None  
        self.cache = None  
    
    def forward(self, x, h_prev, c_prev):
        """
        Forward pass for input gate
        
        Args:
            x: Input at current timestep (batch_size, input_dim)
            h_prev: Hidden state from previous timestep (batch_size, hidden_dim)
            c_prev: Cell state from previous timestep (batch_size, hidden_dim)
            
        Returns:
            i: Input gate activation (batch_size, hidden_dim)
        """
        # Linear transformation
        linear = np.dot(x, self.Wi.T) + np.dot(h_prev, self.Ui.T) + self.bi
        # Sigmoid activation
        i = Sigmoid.forward(linear)
        
        # Cache values for backward pass
        self.cache = (x, h_prev, i)
        
        return i
    
    def backward(self, di, dnext_h, dnext_c, dnext_i):
        """
        Backward pass for input gate
        
        Args:
            di: Gradient of loss w.r.t input gate output (batch_size, hidden_dim)
            dnext_h: Gradient from next hidden state (batch_size, hidden_dim)
            dnext_c: Gradient from cell state (batch_size, hidden_dim)
            dnext_i: Gradient from next input gate (batch_size, hidden_dim)
            
        Returns:
            dx: Gradient w.r.t input (batch_size, input_dim)
            dh_prev: Gradient w.r.t previous hidden state (batch_size, hidden_dim)
            dWi: Gradient w.r.t weight matrix (hidden_dim, input_dim)
            dUi: Gradient w.r.t hidden weight matrix (hidden_dim, hidden_dim)
            dbi: Gradient w.r.t bias (hidden_dim,)
        """
        x, h_prev, i = self.cache
        
        # Compute gradient through sigmoid
        dsigma = Sigmoid.backward(i) * (di + dnext_h + dnext_i)
        
        # Compute gradients for parameters
        dWi = np.dot(dsigma.T, x)
        dUi = np.dot(dsigma.T, h_prev)
        dbi = np.sum(dsigma, axis=0)
        
        # Compute gradients for input and previous hidden state
        dx = np.dot(dsigma, self.Wi)
        dh_prev = np.dot(dsigma, self.Ui)
        
        return dx, dh_prev, dWi, dUi, dbi

class OutputGate:  # LSTM
    def __init__(self):
        self.Wo = None  
        self.Uo = None  
        self.bo = None  
        self.cache = None  
    
    def forward(self, x, h_prev, c_prev):
        """
        Forward pass for output gate
        
        Args:
            x: Input at current timestep (batch_size, input_dim)
            h_prev: Hidden state from previous timestep (batch_size, hidden_dim)
            c_prev: Cell state from previous timestep (batch_size, hidden_dim)
            
        Returns:
            o: Output gate activation (batch_size, hidden_dim)
        """
        # Linear transformation
        linear = np.dot(x, self.Wo.T) + np.dot(h_prev, self.Uo.T) + self.bo
        # Sigmoid activation
        o = Sigmoid.forward(linear)
        
        # Cache values for backward pass
        self.cache = (x, h_prev, o)
        
        return o
    
    def backward(self, do, dnext_h, dnext_c, dnext_o):
        """
        Backward pass for output gate
        
        Args:
            do: Gradient of loss w.r.t output gate (batch_size, hidden_dim)
            dnext_h: Gradient from next hidden state (batch_size, hidden_dim)
            dnext_c: Gradient from cell state (batch_size, hidden_dim)
            dnext_o: Gradient from next output gate (batch_size, hidden_dim)
            
        Returns:
            dx: Gradient w.r.t input (batch_size, input_dim)
            dh_prev: Gradient w.r.t previous hidden state (batch_size, hidden_dim)
            dWo: Gradient w.r.t weight matrix (hidden_dim, input_dim)
            dUo: Gradient w.r.t hidden weight matrix (hidden_dim, hidden_dim)
            dbo: Gradient w.r.t bias (hidden_dim,)
        """
        x, h_prev, o = self.cache
        
        # Compute gradient through sigmoid
        dsigma = Sigmoid.backward(o) * (do + dnext_h + dnext_o)
        
        # Compute gradients for parameters
        dWo = np.dot(dsigma.T, x)
        dUo = np.dot(dsigma.T, h_prev)
        dbo = np.sum(dsigma, axis=0)
        
        # Compute gradients for input and previous hidden state
        dx = np.dot(dsigma, self.Wo)
        dh_prev = np.dot(dsigma, self.Uo)
        
        return dx, dh_prev, dWo, dUo, dbo

class CandidateGate:  # LSTM
    def __init__(self):
        self.Wc = None  
        self.Uc = None  
        self.bc = None  
        self.cache = None  
    
    def forward(self, x, h_prev):
        """
        Forward pass for candidate gate (cell state update)
        
        Args:
            x: Input at current timestep (batch_size, input_dim)
            h_prev: Hidden state from previous timestep (batch_size, hidden_dim)
            
        Returns:
            c_tilde: Candidate cell state (batch_size, hidden_dim)
        """
        from activation import Tanh  # Import here to avoid circular import
        
        # Linear transformation
        linear = np.dot(x, self.Wc.T) + np.dot(h_prev, self.Uc.T) + self.bc
        # Tanh activation
        c_tilde = Tanh.forward(linear)
        
        # Cache values for backward pass
        self.cache = (x, h_prev, c_tilde)
        
        return c_tilde
    
    def backward(self, dc_tilde, dnext_h, dnext_c):
        """
        Backward pass for candidate gate
        
        Args:
            dc_tilde: Gradient of loss w.r.t candidate cell state (batch_size, hidden_dim)
            dnext_h: Gradient from next hidden state (batch_size, hidden_dim)
            dnext_c: Gradient from next cell state (batch_size, hidden_dim)
            
        Returns:
            dx: Gradient w.r.t input (batch_size, input_dim)
            dh_prev: Gradient w.r.t previous hidden state (batch_size, hidden_dim)
            dWc: Gradient w.r.t weight matrix (hidden_dim, input_dim)
            dUc: Gradient w.r.t hidden weight matrix (hidden_dim, hidden_dim)
            dbc: Gradient w.r.t bias (hidden_dim,)
        """
        from activation import Tanh  # Import here to avoid circular import
        
        x, h_prev, c_tilde = self.cache
        
        # Compute gradient through tanh
        dtanh = Tanh.backward(c_tilde) * (dc_tilde + dnext_h + dnext_c)
        
        # Compute gradients for parameters
        dWc = np.dot(dtanh.T, x)
        dUc = np.dot(dtanh.T, h_prev)
        dbc = np.sum(dtanh, axis=0)
        
        # Compute gradients for input and previous hidden state
        dx = np.dot(dtanh, self.Wc)
        dh_prev = np.dot(dtanh, self.Uc)
        
        return dx, dh_prev, dWc, dUc, dbc


class UpdateGate:  # GRU
    def __init__(self):
        self.Wz = None  
        self.Uz = None  
        self.bz = None  
        self.cache = None  
    
    def forward(self, x, h_prev):
        """
        Forward pass for update gate
        
        Args:
            x: Input at current timestep (batch_size, input_dim)
            h_prev: Hidden state from previous timestep (batch_size, hidden_dim)
            
        Returns:
            z: Update gate activation (batch_size, hidden_dim)
        """
        # Linear transformation
        linear = np.dot(x, self.Wz.T) + np.dot(h_prev, self.Uz.T) + self.bz
        # Sigmoid activation
        z = Sigmoid.forward(linear)
        
        # Cache values for backward pass
        self.cache = (x, h_prev, z)
        
        return z
    
    def backward(self, dz, dnext_h, dnext_z):
        """
        Backward pass for update gate
        
        Args:
            dz: Gradient of loss w.r.t update gate (batch_size, hidden_dim)
            dnext_h: Gradient from next hidden state (batch_size, hidden_dim)
            dnext_z: Gradient from next update gate (batch_size, hidden_dim)
            
        Returns:
            dx: Gradient w.r.t input (batch_size, input_dim)
            dh_prev: Gradient w.r.t previous hidden state (batch_size, hidden_dim)
            dWz: Gradient w.r.t weight matrix (hidden_dim, input_dim)
            dUz: Gradient w.r.t hidden weight matrix (hidden_dim, hidden_dim)
            dbz: Gradient w.r.t bias (hidden_dim,)
        """
        x, h_prev, z = self.cache
        
        # Compute gradient through sigmoid
        dsigma = Sigmoid.backward(z) * (dz + dnext_h + dnext_z)
        
        # Compute gradients for parameters
        dWz = np.dot(dsigma.T, x)
        dUz = np.dot(dsigma.T, h_prev)
        dbz = np.sum(dsigma, axis=0)
        
        # Compute gradients for input and previous hidden state
        dx = np.dot(dsigma, self.Wz)
        dh_prev = np.dot(dsigma, self.Uz)
        
        return dx, dh_prev, dWz, dUz, dbz
class ResetGate:  # GRU
    def __init__(self):
        self.Wr = None  
        self.Ur = None  
        self.br = None  
        self.cache = None  
    
    def forward(self, x, h_prev):
        """
        Forward pass for reset gate
        
        Args:
            x: Input at current timestep (batch_size, input_dim)
            h_prev: Hidden state from previous timestep (batch_size, hidden_dim)
            
        Returns:
            r: Reset gate activation (batch_size, hidden_dim)
        """
        # Linear transformation
        linear = np.dot(x, self.Wr.T) + np.dot(h_prev, self.Ur.T) + self.br
        # Sigmoid activation
        r = Sigmoid.forward(linear)
        
        # Cache values for backward pass
        self.cache = (x, h_prev, r)
        
        return r
    
    def backward(self, dr, dnext_h, dnext_r):
        """
        Backward pass for reset gate
        
        Args:
            dr: Gradient of loss w.r.t reset gate (batch_size, hidden_dim)
            dnext_h: Gradient from next hidden state (batch_size, hidden_dim)
            dnext_r: Gradient from next reset gate (batch_size, hidden_dim)
            
        Returns:
            dx: Gradient w.r.t input (batch_size, input_dim)
            dh_prev: Gradient w.r.t previous hidden state (batch_size, hidden_dim)
            dWr: Gradient w.r.t weight matrix (hidden_dim, input_dim)
            dUr: Gradient w.r.t hidden weight matrix (hidden_dim, hidden_dim)
            dbr: Gradient w.r.t bias (hidden_dim,)
        """
        x, h_prev, r = self.cache
        
        # Compute gradient through sigmoid
        dsigma = Sigmoid.backward(r) * (dr + dnext_h + dnext_r)
        
        # Compute gradients for parameters
        dWr = np.dot(dsigma.T, x)
        dUr = np.dot(dsigma.T, h_prev)
        dbr = np.sum(dsigma, axis=0)
        
        # Compute gradients for input and previous hidden state
        dx = np.dot(dsigma, self.Wr)
        dh_prev = np.dot(dsigma, self.Ur)
        
        return dx, dh_prev, dWr, dUr, dbr

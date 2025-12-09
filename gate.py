import numpy as np
from activation import Sigmoid

def _zero_grad(W,U,b,dW,dU,db):
    dW.fill(0)
    dU.fill(0)
    db.fill(0)
    cache = None
    return W,U,b,cache

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
        return grad__W,grad_x
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
        return grad_x, grad_y

class ForgetGate:  # LSTM
    def __init__(self):
        self.Wf = None
        self.Uf = None
        self.bf = None
        self.cache = None
    def init_weights(self, input_size, hidden_size):
        """
        Manually initialize weights. Call this before training!
        """
        # Initialize Weights (Xavier)
        self.Wf = np.random.randn(hidden_size, input_size) * np.sqrt(1 / (hidden_size + input_size))
        self.Uf = np.random.randn(hidden_size, hidden_size) * np.sqrt(1 / (hidden_size + hidden_size))
        self.bf = np.ones(( 1,hidden_size))
        
        # Initialize Gradients
        self.dWf = np.zeros_like(self.Wf)
        self.dUf = np.zeros_like(self.Uf)
        self.dbf = np.zeros_like(self.bf)
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
        dWf = np.dot(dsigma.T, x)
        dUf = np.dot(dsigma.T, h_prev)
        dbf = np.sum(dsigma, axis=0)

        # Compute gradients for input and previous hidden state
        dx = np.dot(dsigma, self.Wf)
        dh_prev = np.dot(dsigma, self.Uf)

        return dx, dh_prev, dWf, dUf, dbf

    def zero_grad(self):
        _zero_grad(self.Wf,self.Uf,self.bf,self.dWf,self.dUf,self.dbf)
class InputGate:  # LSTM
    def __init__(self):
        self.Wi = None
        self.Ui = None
        self.bi = None
        self.cache = None
    def init_weights(self, input_size, hidden_size):
        """
        Manually initialize weights. Call this before training!
        """
        # Initialize Weights (Xavier)
        self.Wi = np.random.randn(hidden_size, input_size) * np.sqrt(1 / (hidden_size + input_size))
        self.Ui = np.random.randn(hidden_size, hidden_size) * np.sqrt(1 / (hidden_size + hidden_size))
        self.bi = np.zeros((1,hidden_size))
        
        # Initialize Gradients
        self.dWi = np.zeros_like(self.Wi)
        self.dUi = np.zeros_like(self.Ui)
        self.dbi = np.zeros_like(self.bi)
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
    def zero_grad(self):
        _zero_grad(self.Wi,self.Ui,self.bi,self.dWi,self.dUi,self.dbi)

class OutputGate:  # LSTM
    def __init__(self):
        self.Wo = None
        self.Uo = None
        self.bo = None
        self.cache = None
    def init_weights(self, input_size, hidden_size):
        """
        Initialize weights and gradients for the output gate
        
        Args:
            input_size: Size of input dimension
            hidden_size: Size of hidden state
        """
        # Initialize weights using Xavier initialization
        self.Wo = np.random.randn(hidden_size, input_size) * np.sqrt(1 / (hidden_size + input_size))
        self.Uo = np.random.randn(hidden_size, hidden_size) * np.sqrt(1 / (hidden_size + hidden_size))
        self.bo = np.zeros((1,hidden_size))
        
        # Initialize gradients
        self.dWo = np.zeros_like(self.Wo)
        self.dUo = np.zeros_like(self.Uo)
        self.dbo = np.zeros_like(self.bo)

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
    
    def zero_grad(self):
        _zero_grad(self.Wo,self.Uo,self.bo,self.dWo,self.dUo,self.dbo)

class CandidateGate:  # LSTM
    def __init__(self):
        self.Wc = None
        self.Uc = None
        self.bc = None
        self.cache = None
        
    def init_weights(self, input_size, hidden_size):
        """
        Initialize weights and gradients for the candidate gate
            
        Args:
            input_size: Size of input dimension
            hidden_size: Size of hidden state
        """
        # Initialize weights using Xavier initialization
        self.Wc = np.random.randn(hidden_size, input_size) * np.sqrt(1 / (hidden_size + input_size))
        self.Uc = np.random.randn(hidden_size, hidden_size) * np.sqrt(1 / (hidden_size + hidden_size))
        self.bc = np.zeros((1,hidden_size))
            
        # Initialize gradients
        self.dWc = np.zeros_like(self.Wc)
        self.dUc = np.zeros_like(self.Uc)
        self.dbc = np.zeros_like(self.bc)

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
        
    def zero_grad(self):
        _zero_grad(self.Wc, self.Uc, self.bc, self.dWc, self.dUc, self.dbc)

class UpdateGate:  # GRU
    def __init__(self):
        self.Wz = None
        self.Uz = None
        self.bz = None
        self.cache = None
        
    def init_weights(self, input_size, hidden_size):
        """
        Initialize weights and gradients for the update gate
            
        Args:
            input_size: Size of input dimension
            hidden_size: Size of hidden state
        """
        # Initialize weights using Xavier initialization
        self.Wz = np.random.randn(hidden_size, input_size) * np.sqrt(1 / (hidden_size + input_size))
        self.Uz = np.random.randn(hidden_size, hidden_size) * np.sqrt(1 / (hidden_size + hidden_size))
        self.bz = np.zeros((1,hidden_size))
            
        # Initialize gradients
        self.dWz = np.zeros_like(self.Wz)
        self.dUz = np.zeros_like(self.Uz)
        self.dbz = np.zeros_like(self.bz)

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
        
    def zero_grad(self):
        _zero_grad(self.Wz, self.Uz, self.bz, self.dWz, self.dUz, self.dbz)
class ResetGate:  # GRU
    def __init__(self):
        self.Wr = None
        self.Ur = None
        self.br = None
        self.cache = None
        
    def init_weights(self, input_size, hidden_size):
        """
        Initialize weights and gradients for the reset gate
            
        Args:
            input_size: Size of input dimension
            hidden_size: Size of hidden state
        """
        # Initialize weights using Xavier initialization
        self.Wr = np.random.randn(hidden_size, input_size) * np.sqrt(1 / (hidden_size + input_size))
        self.Ur = np.random.randn(hidden_size, hidden_size) * np.sqrt(1 / (hidden_size + hidden_size))
        self.br = np.zeros((1,hidden_size))
            
        # Initialize gradients
        self.dWr = np.zeros_like(self.Wr)
        self.dUr = np.zeros_like(self.Ur)
        self.dbr = np.zeros_like(self.br)

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
    def zero_grad(self):
        _zero_grad(self.Wr,self.Ur,self.br,self.dWr,self.dUr,self.dbr)


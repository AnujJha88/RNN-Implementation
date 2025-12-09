import numpy as np
from activation import Tanh

class RNN:
    
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the RNN model.

        Parameters
        ----------
        input_size : int
            The size of the input data.
        hidden_size : int
            The size of the hidden state.
        output_size : int
            The size of the output data.

        Notes
        -----
        The Xavier initialization is used for all weights and biases.
        """
        # Store the input, hidden, and output sizes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights with Xavier initialization
        self.Wxh = np.random.randn(hidden_size, input_size) * np.sqrt(1 / (hidden_size + input_size))
        self.Whh = np.random.randn(hidden_size, hidden_size) * np.sqrt(1 / (hidden_size + hidden_size))
        self.Why = np.random.randn(output_size, hidden_size) * np.sqrt(1 / (hidden_size + output_size))
        
        # Initialize biases to zero
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, x, h_prev):
        """
        Forward pass of the RNN.

        Parameters
        ----------
        x : ndarray
            Input at current timestep (batch_size, input_size).
        h_prev : ndarray
            Hidden state from previous timestep (batch_size, hidden_size).

        Returns
        -------
        h : ndarray
            Hidden state at current timestep (batch_size, hidden_size).
        y : ndarray
            Output at current timestep (batch_size, output_size).
        """
        # Compute hidden state
        h = Tanh.forward(np.dot(self.Whh, h_prev) + np.dot(self.Wxh, x) + self.bh)
        
        # Compute output
        y = np.dot(self.Why, h) + self.by
        
        return h, y

    def backward(self, dh_next, dy, h_prev, x, h):
        """
        Backward pass of the RNN using BPTT.
        
        Parameters
        ----------
        dh_next : ndarray
            Gradient of loss w.r.t next hidden state (batch_size, hidden_size).
        dy : ndarray
            Gradient of loss w.r.t output (batch_size, output_size).
        h_prev : ndarray
            Previous hidden state (batch_size, hidden_size).
        x : ndarray
            Input at current timestep (batch_size, input_size).
        h : ndarray
            Current hidden state (from forward pass).
            
        Returns
        -------
        dWxh : ndarray
            Gradient of loss w.r.t Wxh.
        dWhh : ndarray
            Gradient of loss w.r.t Whh.
        dWhy : ndarray
            Gradient of loss w.r.t Why.
        dbh : ndarray
            Gradient of loss w.r.t bh.
        dby : ndarray
            Gradient of loss w.r.t by.
        dh_prev : ndarray
            Gradient to pass to previous timestep.
        """
        # Gradient of loss w.r.t Why and by
        dWhy = np.dot(dy, h.T)
        dby = np.sum(dy, axis=1, keepdims=True)
        
        # Gradient of loss w.r.t h (from output and next hidden state)
        dh = np.dot(self.Why.T, dy) + dh_next
        
        # Backprop through tanh
        dtanh = Tanh.backward(h) * dh
        
        # Gradients for Wxh, Whh, and bh
        dWxh = np.dot(dtanh, x.T)
        dWhh = np.dot(dtanh, h_prev.T)
        dbh = np.sum(dtanh, axis=1, keepdims=True)
        
        # Gradient for previous hidden state
        dh_prev = np.dot(self.Whh.T, dtanh)
        self.dWxh+=dWxh
        self.dWhh+=dWhh
        self.dWhy+=dWhy
        self.dbh+=dbh
        self.dby+=dby

        return  dh_prev

    def zero_grad(self):
        """
        Zeroes out all gradients.
        
        This should be called after backpropagation to prepare for the next
        iteration.
        """
        self.dWxh = np.zeros_like(self.Wxh)
        self.dWhh = np.zeros_like(self.Whh)
        self.dWhy = np.zeros_like(self.Why)
        self.dbh = np.zeros_like(self.bh)
        self.dby = np.zeros_like(self.by)

    def clip_grads(self, max_norm=5):
        """
        Clips gradients to have a maximum L2 norm of max_norm.
        
        This is done to prevent exploding gradients.
        
        Parameters
        ----------
        max_norm : float
            Maximum L2 norm of gradients to clip to.
        """
        for grad in [self.dWxh, self.dWhh, self.dWhy, self.dbh, self.dby]:
            np.clip(grad, -max_norm, max_norm, out=grad)

    def step(self, learning_rate=0.01):
        """
        Updates the model parameters using gradient descent.
        
        Parameters
        ----------
        learning_rate : float
            Learning rate for gradient descent.
        """
        self.Wxh -= learning_rate * self.dWxh
        self.Whh -= learning_rate * self.dWhh
        self.Why -= learning_rate * self.dWhy
        self.bh -= learning_rate * self.dbh
        self.by -= learning_rate * self.dby
        self.zero_grad()

        
        
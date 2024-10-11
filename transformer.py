import numpy as np  

class SimpleTransformer:  
    def __init__(self, d_model, n_heads):  
        self.d_model = d_model  
        self.n_heads = n_heads  
        self.head_dim = d_model // n_heads  
        
        # Weight matrices for the linear transformations  
        self.W_q = np.random.rand(d_model, d_model)  
        self.W_k = np.random.rand(d_model, d_model)  
        self.W_v = np.random.rand(d_model, d_model)  
        self.W_o = np.random.rand(d_model, d_model)  
        
        # Feed-forward layer weights  
        self.W_ff1 = np.random.rand(d_model, d_model * 4)  
        self.W_ff2 = np.random.rand(d_model * 4, d_model)  
        
        # Layer normalization weights  
        self.gamma = np.ones(d_model)  
        self.beta = np.zeros(d_model)  

    def attention(self, Q, K, V):  
        scores = Q @ K.T / np.sqrt(self.head_dim)  
        weights = self.softmax(scores)  
        return weights @ V  

    def multi_head_attention(self, x):  
        # Linear projections  
        Q = x @ self.W_q  
        K = x @ self.W_k  
        V = x @ self.W_v  
        
        # Split into multiple heads  
        Q = Q.reshape(Q.shape[0], -1, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)  
        K = K.reshape(K.shape[0], -1, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)  
        V = V.reshape(V.shape[0], -1, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)  
        
        # Self-attention for each head  
        attention_output = self.attention(Q, K, V).transpose(0, 2, 1, 3).reshape(x.shape[0], -1)  
        
        # Final linear layer  
        return attention_output @ self.W_o  

    def feed_forward(self, x):  
        return self.W_ff2 @ self.relu(self.W_ff1 @ x)  

    def layer_norm(self, x):  
        return self.gamma * (x - np.mean(x, axis=-1, keepdims=True)) / (np.std(x, axis=-1, keepdims=True) + 1e-6) + self.beta   

    def relu(self, x):  
        return np.maximum(0, x)  

    def forward(self, x):  
        attn_output = self.multi_head_attention(x)  
        x = self.layer_norm(x + attn_output)  
        ff_output = self.feed_forward(x)  
        return self.layer_norm(x + ff_output)  

    def softmax(self, x):  
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # Numerical stability  
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

class SimpleCNN:  
    def __init__(self, input_shape, num_classes):  
        self.input_shape = input_shape  
        self.num_classes = num_classes  

        # Initialize weights for the convolutional layer  
        self.filter_size = (3, 3)  # Example filter size  
        self.num_filters = 8  # Number of filters  
        self.conv_weights = np.random.randn(self.num_filters, *self.filter_size, input_shape[2]) * 0.1  
        
        # Initialize weights for the fully connected layer  
        self.fc_weights = np.random.randn(self.num_filters * (input_shape[0] - self.filter_size[0] + 1) *  
                                           (input_shape[1] - self.filter_size[1] + 1), num_classes) * 0.1  

    def convolution(self, x):  
        """ Perform convolution operation """  
        h, w, c = x.shape  
        f_h, f_w, _ = self.filter_size  

        # Calculate output dimensions  
        out_h = h - f_h + 1  
        out_w = w - f_w + 1  
        conv_output = np.zeros((out_h, out_w, self.num_filters))  

        # Perform convolution  
        for f in range(self.num_filters):  
            for i in range(out_h):  
                for j in range(out_w):  
                    conv_output[i, j, f] = np.sum(x[i:i + f_h, j:j + f_w] * self.conv_weights[f])  

        return conv_output  

    def relu(self, x):  
        """ Apply ReLU activation function """  
        return np.maximum(0, x)  

    def max_pool(self, x, size=2, stride=2):  
        """ Perform max pooling operation """  
        h, w, c = x.shape  
        out_h = (h - size) // stride + 1  
        out_w = (w - size) // stride + 1  
        pooled_output = np.zeros((out_h, out_w, c))  

        for i in range(0, h - size + 1, stride):  
            for j in range(0, w - size + 1, stride):  
                pooled_output[i // stride, j // stride] = np.max(x[i:i + size, j:j + size], axis=(0, 1))  

        return pooled_output  

    def flatten(self, x):  
        """ Flatten the input """  
        return x.flatten()  

    def forward(self, x):  
        """ Forward pass through the CNN """  
        x = self.convolution(x)  
        x = self.relu(x)  # Activation function  
        x = self.max_pool(x)  # Pooling  
        x = self.flatten(x)  # Flatten the output  

        # Fully connected layer  
        return x @ self.fc_weights    

# Example Usage  
if __name__ == "__main__":  
    # # Define input parameters  
    # d_model = 8  # Dimensionality of the model  
    # n_heads = 2  # Number of attention heads  
    # sequences = 5  # Number of input sequences (for batch processing)  
    # seq_length = 4  # Length of each sequence  

    # # Create random input data  
    # input_data = np.random.rand(sequences, seq_length, d_model)  

    # # Create a transformer instance  
    # transformer = SimpleTransformer(d_model, n_heads)  

    # # Forward pass through the transformer  
    # output_data = transformer.forward(input_data)  

    # print("Output shape:", output_data.shape)

    # Input shape: (height, width, channels)  
    input_shape = (32, 32, 3)  # Example input image size (32x32 RGB image)  
    num_classes = 10  # Example number of output classes  

    # Create random input data representing an image  
    input_data = np.random.rand(*input_shape)  

    # Create CNN instance  
    cnn = SimpleCNN(input_shape, num_classes)  

    # Forward pass through the CNN  
    output_data = cnn.forward(input_data)  

    print("Output shape:", output_data.shape) 
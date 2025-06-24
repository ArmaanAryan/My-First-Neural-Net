import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# Load and preprocess data
data = pd.read_csv(r'C:\Users\ALOK\OneDrive\Documents\letter-recognition.csv')
X = data.drop('letter', axis=1).values
y = data['letter'].values

# Normalize features
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Encode labels (A-Z to 0-25)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.T  # Transpose to match NN input format (features x samples)
X_test = X_test.T
_, m_train = X_train.shape

# Initialize parameters
def init_params():
    W1 = np.random.randn(64, 16) * np.sqrt(2.0/16)  # He initialization
    b1 = np.zeros((64, 1))
    W2 = np.random.randn(32, 64) * np.sqrt(2.0/64)
    b2 = np.zeros((32, 1))
    W3 = np.random.randn(26, 32) * np.sqrt(2.0/32)
    b3 = np.zeros((26, 1))
    return W1, b1, W2, b2, W3, b3

# Activation functions
def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))  # Numerical stability
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

def ReLU_deriv(Z):
    return Z > 0

# One-hot encoding
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, 26))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

# Forward propagation
def forward_prop(W1, b1, W2, b2, W3, b3, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = ReLU(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3

# Backward propagation
def backward_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y):
    one_hot_Y = one_hot(Y)
    dZ3 = A3 - one_hot_Y
    dW3 = 1/m_train * dZ3.dot(A2.T)
    db3 = 1/m_train * np.sum(dZ3, axis=1, keepdims=True)
    dZ2 = W3.T.dot(dZ3) * ReLU_deriv(Z2)
    dW2 = 1/m_train * dZ2.dot(A1.T)
    db2 = 1/m_train * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1/m_train * dZ1.dot(X.T)
    db1 = 1/m_train * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2, dW3, db3

# Adam optimizer
class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, params, grads):
        self.t += 1
        new_params = params.copy()
        for key in params:
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])
            
            # Update moments
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)
            
            # Bias correction
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            new_params[key] = params[key] - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return new_params

# Get predictions and accuracy
def get_predictions(A3):
    return np.argmax(A3, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

# Mini-batch gradient descent with early stopping
def gradient_descent(X, Y, X_val, Y_val, batch_size=32, epochs=50, patience=5):
    W1, b1, W2, b2, W3, b3 = init_params()
    adam = Adam(learning_rate=0.001)
    best_val_acc = 0
    best_params = None
    patience_counter = 0
    
    n_samples = X.shape[1]
    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(n_samples)
        X_shuffled = X[:, indices]
        Y_shuffled = Y[indices]
        
        # Mini-batches
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            X_batch = X_shuffled[:, start:end]
            Y_batch = Y_shuffled[start:end]
            
            # Forward and backward prop
            Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X_batch)
            dW1, db1, dW2, db2, dW3, db3 = backward_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X_batch, Y_batch)
            
            # Update parameters
            params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3}
            grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3}
            new_params = adam.update(params, grads)
            
            W1, b1, W2, b2, W3, b3 = [new_params[key] for key in ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']]
        
        # Validation accuracy
        _, _, _, _, _, A3_val = forward_prop(W1, b1, W2, b2, W3, b3, X_val)
        val_predictions = get_predictions(A3_val)
        val_acc = get_accuracy(val_predictions, Y_val)
        
        print(f"Epoch {epoch+1}/{epochs}, Validation Accuracy: {val_acc:.4f}")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = {'W1': W1.copy(), 'b1': b1.copy(), 'W2': W2.copy(), 
                          'b2': b2.copy(), 'W3': W3.copy(), 'b3': b3.copy()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return best_params['W1'], best_params['b1'], best_params['W2'], best_params['b2'], best_params['W3'], best_params['b3']

# Make predictions
def make_predictions(X, W1, b1, W2, b2, W3, b3):
    _, _, _, _, _, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
    predictions = get_predictions(A3)
    return label_encoder.inverse_transform(predictions)

# Test prediction with visualization
def test_prediction(index, W1, b1, W2, b2, W3, b3):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2, W3, b3)
    label = label_encoder.inverse_transform([y_train[index]])
    
    print(f"Prediction: {prediction[0]}")
    print(f"Label: {label[0]}")
    
    # Reshape and visualize (approximate representation)
    current_image = current_image.reshape((4, 4)) * 255  # Simplified visualization
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

# Train the model
W1, b1, W2, b2, W3, b3 = gradient_descent(X_train, y_train, X_test, y_test, batch_size=32, epochs=50)

# Test accuracy
_, _, _, _, _, A3_test = forward_prop(W1, b1, W2, b2, W3, b3, X_test)
test_predictions = get_predictions(A3_test)
test_accuracy = get_accuracy(test_predictions, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Visualize some predictions
for i in range(4):
    test_prediction(i, W1, b1, W2, b2, W3, b3)
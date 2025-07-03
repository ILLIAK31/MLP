import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time
import copy

class MLPApproximator:
    ALGO_NAMES = ["sgd_simple", "sgd_momentum", "rmsprop", "adam"]
    def __init__(self, structure=[16, 8, 4], activation_name="relu", targets_activation_name="linear", initialization_name="uniform", 
                 algo_name="sgd_simple", learning_rate=1e-2,  n_epochs=100, batch_size=10, seed=0,
                 verbosity_e=100, verbosity_b=10):        
        self.structure = structure
        self.activation_name = activation_name
        self.targets_activation_name = targets_activation_name
        self.initialization_name = initialization_name
        self.algo_name = algo_name
        if self.algo_name not in self.ALGO_NAMES:
            self.algo_name = self.ALGO_NAMES[0]                            
        self.loss_name = "squared_loss"
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.seed = seed        
        self.verbosity_e = verbosity_e 
        self.verbosity_b = verbosity_b
        self.history_weights = {}
        self.history_weights0 = {}
        self.n_params = None
        self.momentum_beta = 0.9
        self.rmsprop_beta = 0.9
        self.rmsprop_epsilon = 1e-7
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_epsilon = 1e-7
                
    def __str__(self):
        txt = f"{self.__class__.__name__}(structure={self.structure},"
        txt += "\n" if len(self.structure) > 32 else " "              
        txt += f"activation_name={self.activation_name}, targets_activation_name={self.targets_activation_name}, initialization_name={self.initialization_name}, "
        txt += f"algo_name={self.algo_name}, learning_rate={self.learning_rate}, n_epochs={self.n_epochs}, batch_size={self.batch_size})"
        if self.n_params:
            txt += f" [n_params: {self.n_params}]"     
        return txt
    
    @staticmethod
    def he_uniform(n_in, n_out):
        scaler = np.sqrt(6.0 / n_in)
        return ((np.random.rand(n_out, n_in)  * 2.0 - 1.0) * scaler).astype(np.float32)
    
    @staticmethod
    def he_normal(n_in, n_out):
        scaler = np.sqrt(2.0 / n_in)
        return (np.random.randn(n_out, n_in) * scaler).astype(np.float32)

    @staticmethod
    def glorot_uniform(n_in, n_out):
        scaler = np.sqrt(6.0 / (n_in + n_out))
        return ((np.random.rand(n_out, n_in)  * 2.0 - 1.0) * scaler).astype(np.float32)
    
    @staticmethod
    def glorot_normal(n_in, n_out):
        scaler = np.sqrt(2.0 / (n_in + n_out))
        return (np.random.randn(n_out, n_in) * scaler).astype(np.float32)

    @staticmethod
    def prepare_batch_ranges(m, batch_size):
        n_batches = int(np.ceil(m / batch_size))
        batch_ranges = batch_size * np.ones(n_batches, dtype=np.int32)
        remainder = m % batch_size
        if remainder > 0:        
            batch_ranges[-1] = remainder
        batch_ranges = np.r_[0, np.cumsum(batch_ranges)]                
        return n_batches, batch_ranges    

    @staticmethod
    def sigmoid(S):         
        return 1 / (1 + np.exp(-S))
    
    @staticmethod
    def sigmoid_d(phi_S):
        return phi_S * (1 - phi_S)
        
    @staticmethod
    def relu(S):
        return np.maximum(0, S)

    @staticmethod
    def relu_d(phi_S):
        return (phi_S > 0).astype(float)

    @staticmethod
    def linear(S):
        return S

    @staticmethod
    def linear_d(phi_S):
        return np.ones_like(phi_S)
    
    @staticmethod
    def squared_loss(y_MLP, y_target):
        return 0.5 * np.mean((y_MLP - y_target)**2)
        
    @staticmethod
    def squared_loss_d(y_MLP, y_target):
        return y_MLP - y_target
    
    def pre_algo_sgd_simple(self):
        return

    def algo_sgd_simple(self, l):
        self.weights_[l] -= self.learning_rate * self.gradients[l]
        self.weights0_[l] -= self.learning_rate * self.gradients0[l]

    def pre_algo_sgd_momentum(self):
        self.velocity_w = [np.zeros_like(w) for w in self.weights_]
        self.velocity_b = [np.zeros_like(b) for b in self.weights0_]

    def algo_sgd_momentum(self, l):
        self.velocity_w[l] = self.momentum_beta * self.velocity_w[l] + self.learning_rate * self.gradients[l]
        self.velocity_b[l] = self.momentum_beta * self.velocity_b[l] + self.learning_rate * self.gradients0[l]
        self.weights_[l] -= self.velocity_w[l]
        self.weights0_[l] -= self.velocity_b[l]

    def pre_algo_rmsprop(self):
        self.squared_grad_w = [np.zeros_like(w) for w in self.weights_]
        self.squared_grad_b = [np.zeros_like(b) for b in self.weights0_]

    def algo_rmsprop(self, l):
        self.squared_grad_w[l] = self.rmsprop_beta * self.squared_grad_w[l] + (1 - self.rmsprop_beta) * (self.gradients[l] ** 2)
        self.squared_grad_b[l] = self.rmsprop_beta * self.squared_grad_b[l] + (1 - self.rmsprop_beta) * (self.gradients0[l] ** 2)
        self.weights_[l] -= self.learning_rate * self.gradients[l] / (np.sqrt(self.squared_grad_w[l]) + self.rmsprop_epsilon)
        self.weights0_[l] -= self.learning_rate * self.gradients0[l] / (np.sqrt(self.squared_grad_b[l]) + self.rmsprop_epsilon)

    def pre_algo_adam(self):
        self.m_w = [np.zeros_like(w) for w in self.weights_]
        self.v_w = [np.zeros_like(w) for w in self.weights_]
        self.m_b = [np.zeros_like(b) for b in self.weights0_]
        self.v_b = [np.zeros_like(b) for b in self.weights0_]

    def algo_adam(self, l):
        t = self.t + 1
        self.m_w[l] = self.adam_beta1 * self.m_w[l] + (1 - self.adam_beta1) * self.gradients[l]
        self.v_w[l] = self.adam_beta2 * self.v_w[l] + (1 - self.adam_beta2) * (self.gradients[l] ** 2)
        m_w_hat = self.m_w[l] / (1 - self.adam_beta1 ** t)
        v_w_hat = self.v_w[l] / (1 - self.adam_beta2 ** t)
        self.weights_[l] -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.adam_epsilon)

        self.m_b[l] = self.adam_beta1 * self.m_b[l] + (1 - self.adam_beta1) * self.gradients0[l]
        self.v_b[l] = self.adam_beta2 * self.v_b[l] + (1 - self.adam_beta2) * (self.gradients0[l] ** 2)
        m_b_hat = self.m_b[l] / (1 - self.adam_beta1 ** t)
        v_b_hat = self.v_b[l] / (1 - self.adam_beta2 ** t)
        self.weights0_[l] -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.adam_epsilon)

    def fit(self, X, y):
        np.random.seed(self.seed)
        self.activation_ = getattr(MLPApproximator, self.activation_name)
        self.activation_d_ = getattr(MLPApproximator, self.activation_name + "_d")
        self.initialization_ = getattr(MLPApproximator, ("he_" if self.activation_name == "relu" else "glorot_") + self.initialization_name)
        self.targets_activation_ = getattr(MLPApproximator, self.targets_activation_name)
        self.targets_activation_d_ = getattr(MLPApproximator, self.targets_activation_name + "_d")        
        self.loss_ = getattr(MLPApproximator, self.loss_name)
        self.loss_d_ = getattr(MLPApproximator, self.loss_name + "_d")
        self.pre_algo_ = getattr(self, "pre_algo_" + self.algo_name)
        self.algo_ = getattr(self, "algo_" + self.algo_name)                
        self.weights_ = [None]  
        self.weights0_ = [None]
        m, n = X.shape
        if len(y.shape) == 1:
            y = np.array([y]).T
        self.n_ = n
        self.n_targets_ = 1 if len(y.shape) == 1 else y.shape[1]
        self.n_params = 0
        for l in range(len(self.structure) + 1):
            n_in = n if l == 0 else self.structure[l - 1]
            n_out = self.structure[l] if l < len(self.structure) else self.n_targets_ 
            w = self.initialization_(n_in, n_out)
            w0 = np.zeros((n_out, 1), dtype=np.float32)            
            self.weights_.append(w)
            self.weights0_.append(w0)
            self.n_params += w.size
            self.n_params += w0.size
        t1 = time.time()
        if self.verbosity_e > 0:
            print(f"FIT [total of weights (params): {self.n_params}]")
        self.pre_algo_() # if some preparation needed         
        n_batches, batch_ranges = MLPApproximator.prepare_batch_ranges(m, self.batch_size)
        self.t = 0
        for e in range(self.n_epochs):
            t1_e = time.time()
            if e % self.verbosity_e == 0 or e == self.n_epochs - 1:
                print("-" * 3)
                print(f"EPOCH {e + 1}/{self.n_epochs}:")
                self.forward(X)
                loss_e_before = np.mean(self.loss_(self.signals[-1], y))                
            p = np.random.permutation(m)          
            for b in range(n_batches):
                indexes = p[batch_ranges[b] : batch_ranges[b + 1]]
                X_b = X[indexes] 
                y_b = y[indexes]                
                self.forward(X_b)
                loss_b_before = np.mean(self.loss_(self.signals[-1], y_b))                
                self.backward(y_b)
                for l in range(1, len(self.structure) + 2):
                    self.algo_(l)                    
                if (e % self.verbosity_e == 0 or e == self.n_epochs - 1) and b % self.verbosity_b == 0:
                    self.forward(X_b)
                    loss_b_after = np.mean(self.loss_(self.signals[-1], y_b))                    
                    print(f"[epoch {e + 1}/{self.n_epochs}, batch {b + 1}/{n_batches} -> loss before: {loss_b_before}, loss after: {loss_b_after}]")                                                                        
                self.t += 1
            t2_e = time.time()
            if e % self.verbosity_e == 0 or e == self.n_epochs - 1:
                self.forward(X)
                loss_e_after = np.mean(self.loss_(self.signals[-1], y))
                self.history_weights[e] = copy.deepcopy(self.weights_)
                self.history_weights0[e] = copy.deepcopy(self.weights0_)
                print(f"ENDING EPOCH {e + 1}/{self.n_epochs} [loss before: {loss_e_before}, loss after: {loss_e_after}; epoch time: {t2_e - t1_e} s]")                  
        t2 = time.time()
        if self.verbosity_e > 0:
            print(f"FIT DONE. [time: {t2 - t1} s]")
                                                          
    def forward(self, X_b):
        self.signals = [None] * (len(self.structure) + 2)
        self.signals[0] = X_b
        for l in range(1, len(self.structure) + 2):
            self.signals[l] = np.dot(self.signals[l - 1], self.weights_[l].T) + self.weights0_[l].T
            self.signals[l] = self.activation_(self.signals[l])

    def backward(self, y_b):        
        self.deltas = [None] * len(self.signals)        
        self.gradients = [None] * len(self.signals)
        self.gradients0 = [None] * len(self.signals)
        self.deltas[-1] = self.targets_activation_d_(self.signals[-1]) * self.loss_d_(self.signals[-1], y_b)
        for l in range(len(self.structure) + 1, 0, -1):
            self.gradients[l] = np.dot(self.deltas[l].T, self.signals[l - 1]) / y_b.shape[0]
            self.gradients0[l] = np.mean(self.deltas[l], axis=0, keepdims=True).T
            if l > 1:
                self.deltas[l - 1] = self.activation_d_(self.signals[l - 1]) * np.dot(self.deltas[l], self.weights_[l])
                            
    def predict(self, X):
        self.forward(X)        
        y_pred = self.signals[-1]  
        if self.n_targets_ == 1:
            y_pred = y_pred[:, 0]
        return y_pred

if __name__ == '__main__':
    print("MLP DEMO...")
    domain = 1.5 * np.pi
    noise_std = 0.1
    m_train, m_test = 1000, 10000
    print(f"DATA SETTINGS: domain={domain}, noise_std={noise_std}, m_train={m_train}, m_test={m_test}")

    def fake_data(m, domain, noise_std):
        np.random.seed(0)
        X = np.random.rand(m, 2) * domain
        y = np.cos(X[:, 0] * X[:, 1]) * np.cos(2 * X[:, 0]) + np.random.randn(m) * noise_std
        return X, y
    X_train, y_train = fake_data(m_train, domain, noise_std)
    X_test, y_test = fake_data(m_test, domain, noise_std)
    approx = MLPApproximator(structure=[32, 16, 8], activation_name="relu", targets_activation_name="linear", initialization_name="uniform", 
                             algo_name="adam", learning_rate=1e-3, n_epochs=1000, batch_size=10)
    print(f"APPROXIMATOR: {approx}")
    approx.fit(X_train, y_train)
    y_pred_train = approx.predict(X_train)
    y_pred_test = approx.predict(X_test)
    train_loss = np.mean((y_pred_train - y_train)**2)
    test_loss = np.mean((y_pred_test - y_test)**2)
    print(f"LOSS TRAIN (MSE): {train_loss}")
    print(f"LOSS TEST (MSE): {test_loss}")
    print("MLP DEMO DONE.")

    epochs = list(approx.history_weights.keys())
    train_losses = []
    test_losses = []
    for epoch in epochs:
        approx.weights_ = approx.history_weights[epoch]
        approx.weights0_ = approx.history_weights0[epoch]
        train_losses.append(np.mean((approx.predict(X_train) - y_train)**2))
        test_losses.append(np.mean((approx.predict(X_test) - y_test)**2))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Train Loss", color="blue")
    plt.plot(epochs, test_losses, label="Test Loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.title("Loss During Training")
    plt.legend()
    plt.grid()
    plt.show()

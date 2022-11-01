import numpy as np

def train(features, labels, lr=0.1, n_iter=10):

    n, d = features.shape
    assert len(labels) == n

    # initialize weights
    w = np.zeros((d,))

    # NB: if all is zero-mean, we are scale-free.
    # With bias things would change.

    for i_iter in range(10):
        for i_example in range(n):
            y_hat = np.sign(np.dot(features[i_example], w))
            w += lr * (labels[i_example] - y_hat) * features[i_example]

    return w


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # you can also illustrate the difference between 
    # random and non-random patterns (like mnist)! (better bounds?)

    n = 300
    d = 160

    X = np.random.randn(n, d)

    # assign random classes (use a normal function for binary 
    # random)
    y = np.sign(np.random.randn(n))

    w = train(X, y, n_iter=100)

    print("The cos(angle) between the true and the estimated label vector is:", 
            np.dot(np.sign(X @ w), y) / n)
    w_orth = np.random.randn(d) 
    w_orth -= w * np.dot(w_orth, w) / np.linalg.norm(w)**2
    print(w_orth.shape)

    X_w = X @ w / np.linalg.norm(w)
    X_orth = X @ w_orth / np.linalg.norm(w_orth)

    plt.scatter(X_orth[y==1], X_w[y==1])
    plt.scatter(X_orth[y==-1], X_w[y==-1])
    plt.show()

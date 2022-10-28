#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from ipywidgets import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

get_ipython().run_line_magic('matplotlib', 'notebook')

mpl.rcParams['axes.spines.top'] = 0
mpl.rcParams['axes.spines.right'] = 0
mpl.rcParams['axes.spines.left'] = 1
mpl.rcParams['axes.spines.bottom'] = 1
mpl.rcParams.update({'font.size': 12})


# # Lecture 5: Different Perspectives on Linear Regression 
# ## Pattern Recognition, Fall 2022
# 
# **Ivan Dokmanić**
# 
# 
# <img src="figs/sada-unibas-0.svg" width="60%" float="left">
# 

# # Updates
# 
# - HW1 is out!
# - Discussion on Wednesday

# # Plan for today
# 
# 

# # Recap: sample vs population
# 
# - In supervised learning we assume having access to $n$ ``labeled'' patterns (or instances)
# 
# $$
#     (x_1, y_1), (x_2, y_2), \ldots, (x_n, y_n)
# $$
# 
# - we assume that each pattern–label pair $(x_i, y_i)$ is a _draw_ from the same distribution as that of $(X, Y)$ (sometimes you'll hear the word _realization_)
# - both _draw_ and _realization_ are a bit imprecise but convenient; deterministic numbers admit no notion of distribution or independence
# - when we want to do math and statistics of learning, we assume that the sample consists of $n$ iid copies of $(X, Y)$ 
# 
# $$
#     (X_1, Y_1), (X_2, Y_2), \ldots, (X_n, Y_n)
# $$
# 
# - by copies we mean variables with the same joint distribution but independent for different $i$
# - key assumption of ``independent, identically-distributed random variables'' (iid, i.i.d.)

# # Linear regression
# 
# We assume a linear model
# 
# $$
# \hat{Y}(x) = \hat{y} = w_0 + w_1 \cdot x
# $$
# 
# - Which $w_0$, $w_1$ should we take? Those that minimize the error!
# - How do we measure error for 1 sample? Typically for linear regression, quadratic error,
# $$
#     \mathrm{loss}(\hat{y}, y) = (\hat{y} - y)^2
# $$
# - Earlier we would have tried something like $\min_{w_0, w_1} \mathbb{E} ~ \mathrm{loss}(\hat{Y}(X), Y) = \min_{w_0, w_1} \mathbb{E} ~ (w_0 + w_1 X - Y)^2$ but now we are in the realistic supervised learning mode so we cannot compute expectations
# - We then attempt to solve
# $$
#     \min_{w_0, w_1} \frac{1}{n} \sum_{i = 1}^n (w_0 + w_1 x_i - y_i)^2
# $$

# In[2]:


from sklearn.linear_model import LinearRegression

n = 10
a = 1.0
b = 0.2
sig = 0.08
deg = n - 1

x = np.random.rand(n, 1)
x = np.sort(x, axis=0)
y = a*x + b + sig*np.random.randn(n, 1)

fig, ax = plt.subplots(1, 1)
ax.scatter(x, y, color='g')

reg = LinearRegression().fit(x, y)
y_hat = reg.predict(x)

x_grid = np.linspace(0, 1, 200).reshape(-1, 1)
line, = ax.plot(x_grid, reg.predict(x_grid), 'k')

for i in range(n):
    plt.plot([x[i], x[i]], [y[i], y_hat[i]], 'y')


# # Recap: linear regression and least squares
# 
# 
# $$
# \begin{eqnarray}
#     \sum_{i = 1}^n (y_i - (w_0 + w_1 \cdot x_i))^2 
#     &=& 
#     \sum_{i = 1}^n \left(y_i - [1 \quad x] \begin{bmatrix}w_0 \\ w_1\end{bmatrix}\right)^2 
#     &=&
#     \| \mathbf{y} - \mathbf{X} \mathbf{w} \|^2
# \end{eqnarray}
# $$
# 
# NB: if $(X_i, Y_i) \overset{\text{iid}}{\sim} \mathbb{P}$, then by the law of large numbers
# $$
#     \lim_{n \to \infty} \frac{1}{n} \sum_{i = 1}^n (Y_i - (w_0 + w_1 \cdot X_i))^2 \to \mathbb{E} ~ (Y - (w_0 + w_1 \cdot X))^2
# $$

# We want to find the best weights that minimize the loss
# 
# $$
#     \mathbf{w}^\star = \arg\min_{\mathbf{w}} \underbrace{\| \mathbf{y} - \mathbf{X} \mathbf{w} \|^2}_{= \text{loss} ~ \mathcal{L}(\mathbf{w})}
# $$
# 
# Solution as usual: compute the partial derivatives with respect to the components of $\mathbf{w} = [w_0, w_1, \ldots, w_d]$
# 
# It turns out:
# $$
# \nabla_\mathbf{w} \left( \| \mathbf{y} - \mathbf{X} \mathbf{w} \|^2 \right)
# =
# \frac{\partial \mathcal{L}(\mathbf{w})}{\partial \mathbf{w}}
# =
# \begin{bmatrix}
#     \frac{\partial \mathcal{L}}{\partial w_0} \\
#     \frac{\partial \mathcal{L}}{\partial w_1} \\
#     \vdots \\
#     \frac{\partial \mathcal{L}}{\partial w_d} \\
# \end{bmatrix}
# = 
# -2 \mathbf{X}^T(\mathbf{y} - \mathbf{X} \mathbf{w})
# $$
# 
# ### Optimal weights
# $$
# -2 \mathbf{X}^T(\mathbf{y} - \mathbf{X} \mathbf{w}) = \mathbf{0}
# \Rightarrow
# \mathbf{w}^\star = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}
# $$

# # Higher dimension
# 
# - In ``real'' pattern recognition we more or less never have simple scalar patterns / features $x_i \in \mathbb{R}$
# - In digit classification we had vector features $\mathbf{x} \in \mathbb{R}^{784}$ or $\mathbf{x} \in \mathbb{R}^{28 \times 28}$
# - The prediction is
# $$
#     \hat{Y}(x) = w_0 + \sum_{i = 1}^d w_i x_i
#     =
#     [1, x_1, \ldots, x_d]
#     \begin{bmatrix}
#         w_0 \\ w_1 \\ \vdots \\ w_d
#     \end{bmatrix}
#     =: \mathbf{x}^T \mathbf{w}
# $$
# 
# For a training set $(\mathbf{x}_1, y_1), \ldots, (\mathbf{x}_n, y_n)$ we can write
# $$
#     \mathbf{X} =
#     \begin{bmatrix}
#         - & \mathbf{x}_1^T & - \\
#         - & \vdots & - \\
#         - & \mathbf{x}_n^T & -
#     \end{bmatrix}
# $$
# and the loss becomes
# $$
# \arg \min_{\mathbf{w} \in \mathbb{R}^2} \frac{1}{2} \| \mathbf{y} - \mathbf{X} \mathbf{w} \|^2
# $$
# which is suspiciously similar to what we had before!

# # An alternative derivation of optimal weights
# 
# $$
#     \mathbf{a}^T \mathbf{Q} \mathbf{a} = \sum_{i = 1}^d \sum_{j = 1}^d a_i q_{ij}  a_j
# $$
# 
# so that
# 
# $$
# \begin{aligned}
#     \frac{\partial \mathbf{a}^T \mathbf{Q} \mathbf{a}}{\partial a_k} 
#     &= \sum_{j = 1}^d a_j q_{kj} + \sum_{i = 1}^d a_i q_{ik}  \\ 
#     &= (\mathbf{Q} \mathbf{a})_k + (\mathbf{Q}^T \mathbf{a})_k
# \end{aligned}
# $$
# 
# and
# 
# $$
#     \nabla_{\mathbf{a}} (\mathbf{a}^T \mathbf{Q} \mathbf{a})
#     =
#     \begin{bmatrix}
#     \frac{\partial \mathbf{a}^T \mathbf{Q} \mathbf{a}}{\partial a_1} \\
#     \frac{\partial \mathbf{a}^T \mathbf{Q} \mathbf{a}}{\partial a_2} \\
#     \vdots \\
#     \frac{\partial \mathbf{a}^T \mathbf{Q} \mathbf{a}}{\partial a_d} \\
#     \end{bmatrix}
#     =
#     \begin{bmatrix}
#     (\mathbf{Q} \mathbf{a})_1 + (\mathbf{Q}^T \mathbf{a})_1 \\
#     (\mathbf{Q} \mathbf{a})_2 + (\mathbf{Q}^T \mathbf{a})_2 \\
#     \vdots \\
#     (\mathbf{Q} \mathbf{a})_d + (\mathbf{Q}^T \mathbf{a})_d \\
#     \end{bmatrix}
#     =
#     \mathbf{Q} \mathbf{a} + \mathbf{Q}^T \mathbf{a}
# $$

# Then 
# 
# $$
#     \| \mathbf{X} \mathbf{w} - \mathbf{y} \|^2 = (\mathbf{X} \mathbf{w} - \mathbf{y})^T(\mathbf{X} \mathbf{w} - \mathbf{y}) = \mathbf{w}^T \mathbf{X}^T \mathbf{X} \mathbf{w} - 2 \mathbf{w}^T (\mathbf{X}^T \mathbf{y}) + \mathbf{y}^T \mathbf{y}
# $$
# 
# $$
#     \frac{\partial \mathbf{w}^T \mathbf{X}^T \mathbf{X} \mathbf{w}}{\partial \mathbf{w}}
#     =
#     \mathbf{X}^T \mathbf{X} \mathbf{w} + (\mathbf{X}^T \mathbf{X})^T \mathbf{w}
#     = 
#     2 \mathbf{X}^T \mathbf{X} \mathbf{w}
# $$
# 
# 
# $$
#     \frac{\partial \mathbf{w}^T (\mathbf{X}^T \mathbf{y})}{\partial \mathbf{w}}
#     =
#     \mathbf{X}^T \mathbf{y}
# $$
# 
# $$
#     \frac{\partial \mathbf{y}^T \mathbf{y}}{\partial \mathbf{w}} = \mathbf{0}
# $$
# 

# # A geometric perspective
# 
# We can rewrite
# 
# $$
#     \min \left\{ \| \mathbf{y} - \mathbf{X} \mathbf{w} \|^2   \mid \mathbf{w} \in \mathbb{R}^d \right\}
# $$
# 
# as
# 
# $$
#     \min \left\{ \| \mathbf{y} - \mathbf{v} \|^2 \mid \mathbf{v} = \mathbf{X} \mathbf{w}, \mathbf{w} \in \mathbb{R}^d \right\}
# $$
# 
# If we ignore $\mathbf{w}$ for the moment and only care about $\mathbf{v}$, then we can also write
# 
# $$
# \min \left\{ \| \mathbf{y} - \mathbf{v} \|^2 \mid \mathbf{v} \in \mathrm{span} \{ \mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(d)} \}\right\}
# $$
# 
# where $\mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(d)}$ are the columns of $\mathbf{X}$.
# 
# So which vector $\mathbf{v} \in S = \mathrm{span} \{ \mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(d)} \}$ minimizes this distance? Consider a small example with $d = 2$, $n = 3$
# 
# 

# In[252]:


x1 = np.array([1, 1, 1])
x2 = np.array([1, -1, 1])
y = np.array([-3, 1, 2])

X = np.vstack((v1, v2)).T
w = np.linalg.solve(V.T @ V, V.T @ v3)
y_hat = X @ w

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

xx, yy = np.meshgrid(range(-3, 4), range(-3, 4))
zz = xx

ax.quiver(
        0, 0, 0,
        x1[0], x1[1], x1[2],
        color='k', alpha = .8, lw = 1,
    )
ax.quiver(
        0, 0, 0,
        x2[0], x2[1], x2[2],
        color='k', alpha=.8, lw=1,
    )
ax.quiver(
        0, 0, 0,
        y[0], y[1], y[2],
        color='green', alpha=.8, lw=3,
    )
ax.quiver(
        0, 0, 0,
        y_hat[0], y_hat[1], y_hat[2],
        color='green', alpha=.8, lw=3,
    )
ax.plot([y[0], y_hat[0]], [y[1], y_hat[1]], [y[2], y_hat[2]], 'g:')
ax.plot_surface(xx, yy, zz, alpha=0.5)

ax.text(*x1, '$x_1$')
ax.text(*x2, '$x_2$')
ax.text(*y, '$y$')
ax.text(*y_hat, '$\widehat{y}$')


# # A probabilistic perspective on linear regression
# 
# In linear regression we assume (or hope) that
# 
# $$
#     Y = \mathbf{w}^T \mathbf{X} + \epsilon
# $$
# 
# - $\epsilon$ is often assumed Gaussian, $\epsilon \sim \mathcal{N}(\mu, \sigma^2)$
# 
# Thus can write
# 
# $$
#     p(y \mid \mathbf{x};  \mathbf{\theta}) = \mathcal{N}(y \mid \mu(\mathbf{x}), \sigma^2(\mathbf{x}))
# $$

# In[31]:


from sklearn.linear_model import LinearRegression

n = 10000
a = 1.0
b = 0.2
sig = 0.08
deg = n - 1

x = np.random.rand(n, 1)
x = np.sort(x, axis=0)
y = a*x + b + sig*np.random.randn(n, 1)

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].scatter(x, y, color='g', marker='.')

reg = LinearRegression().fit(x, y)
x_grid = np.linspace(0, 1, 200).reshape(-1, 1)

line, = axs[0].plot([],[], 'k')
line.set_xdata(x_grid)
line.set_ydata(reg.predict(x_grid))

axs[1].hexbin(x.flatten(), y.flatten(), gridsize=32)


# In[51]:


from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm

x_ = np.linspace(0, 1.0, 200)
y_ = np.linspace(0, 1.5, 200)
xx, yy = np.meshgrid(x_, y_)
zz = 1 / sig / np.sqrt(2*np.pi) * np.exp(-(a * xx + b - yy)**2 / 2 / sig**2)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(xx, yy, zz, cmap=cm.coolwarm,  rcount=100, ccount=100)


# # _Statistically_ optimal parameters: Maximum likelihood
# 
# Remember our training set: we model it as independent, identically distributied random draws (copies) 
# 
# We can thus compute the probability (likelihood) of observing a given training set
# 
# $$
#     \hat{\theta} = \arg\max_{\theta} p(\mathcal{D} ; \theta) = \arg\max_\theta \prod_{i = 1}^n p(y_i | \mathbf{x}; \theta)
# $$
# 
# $$
#     \ell(\theta) := \log p(\mathcal{D}; \theta) = \sum_{i = 1}^n \log p(y_i | \mathbf{x}_i ; \theta)
# $$
# 
# $$
# \mathrm{NLL}(\theta) := - \ell(\theta) = -  \sum_{i = 1}^n \log p(y_i | \mathbf{x}_i ; \theta)
# $$
# 
# 
# 
# 
# 
# 
# 

# # Applying this to our model
# 
# $$
# \begin{aligned}
#     \ell(\theta) 
#     &= \sum_{i = 1}^n \log \left[ \frac{1}{\sigma \sqrt{2 \pi}} \exp \left( -\frac{1}{2\sigma^2}(y_i - \mathbf{w}^T \mathbf{x}_i)^2 \right) \right] \\
#     &= - \frac{1}{2 \sigma^2} \underbrace{\sum_{i = 1}^n (y_i - \mathbf{w}^T \mathbf{x}_i)^2}_{\text{residual sum of squares}~\mathrm{RSS}(\mathbf{w})} - \frac{n}{2} \log(2\pi\sigma^2)
# \end{aligned}    
# $$
# 
# Note that since our underlying functional model is $Y = \mathbf{w}^T \mathbf{X} + \epsilon$, we have that 
# $$
#     \mathrm{RSS}(\mathbf{w}) = \| \boldsymbol{\epsilon} \|^2 = \sum_{i = 1}^n \epsilon_i^2
# $$

# # Nonlinear extensions

# # A crazy idea? Random features?

# # Examples with code

# In[137]:


from mlxtend.data import loadlocal_mnist

X_train, y_train = loadlocal_mnist(
        images_path='/Users/dokman0000/Downloads/train-images-idx3-ubyte', 
        labels_path='/Users/dokman0000/Downloads/train-labels-idx1-ubyte'
        )

d = 28**2
n = 5000
shuffle_idx = np.random.choice(X_train.shape[0], n)
X_train = X_train[shuffle_idx]
y_train = y_train[shuffle_idx]


# In[168]:


nz_mask = np.any(X_train > 0, axis=0)
X_mask = X_train[:, nz_mask] / 255.0
d_mask = nz_mask.sum()
lam = 0.1
w_mask = np.linalg.inv(X_mask.T @ X_mask + lam * np.eye(d_mask)) @ X_mask.T @ y_train


# In[169]:


np.linalg.norm(np.round(X_mask @ w_mask) - y_train) / np.linalg.norm(y_train)


# In[170]:


w = np.zeros((d,))
w[nz_mask] = w_mask
fig, ax = plt.subplots(1, 1)
ax.imshow(w.reshape(28, 28))


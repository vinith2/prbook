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


# # Lecture 2: Generalization and Knowledge Modeling
# ## Pattern Recognition, Fall 2022
# 
# **Ivan Dokmanić**
# 
# 
# <img src="figs/sada-unibas-0.svg" width="60%" float="left">
# 

# # Admin updates
# 
# - The website is up and almost stable at [sada.dmi.unibas.ch/teaching/pr22](https://sada.dmi.unibas.ch/teaching/pr22)
# - All materials are being maintained at [git.scicore.unibas.ch/dokmanic-courses/pr22/](https://git.scicore.unibas.ch/dokmanic-courses/pr22/)
# - Two different catalogue entries for exercises are a feature, not a bug!
#     - We will strive to sync Monday with the _previous_ Wednesday
#     - No panic whatsoever needed!
# - The Piazza forum is up: it's ugly but very effective
# - Please make sure to look at the reading material
# - We will use Gradescope for grading, information coming next week

# # Last session's quiz

# # The plan for today?
# 
# We are setting up the framework
# 
# - Generalization?
# - Clever Hans predictors
# - Optimal classification...
# - ... and modeling knowledge using probability distributions
# 

# In[2]:


image_size = (250, 250)
scale = 0

noise = np.zeros((2, *image_size))

noise[0] = scale * np.random.randn(*image_size)
noise[1] = scale * np.random.randn(*image_size)

# noise[1] = -noise[0] # homework

from pathlib import Path
result = list(Path('/Users/dokman0000/Downloads/lfw/').rglob('*.jpg'))

n_train = 1000
n_test = 100
images = np.zeros((n_train, 250, 250))
labels = np.zeros((n_train,), dtype=np.int8)

images_test = np.zeros((n_test, 250, 250))
labels_test = np.zeros((n_test,), dtype=np.int8)

shuffle_idx = np.random.permutation(n_train + n_test)
for i in range(n_train):
    images[i] = plt.imread(result[shuffle_idx[i]]).mean(axis=2)
    labels[i] = np.round(np.random.rand())
    images[i] += noise[labels[i]]

for i in range(n_train, n_train + n_test):
    images_test[i - n_train] = plt.imread(result[shuffle_idx[i]]).mean(axis=2)
    labels_test[i - n_train] = np.round(np.random.rand())
    # no noise in the test set!


# # Let us start with an experiment
# 
# - we have dataset of face images of a group of people
# - some of them are dog people, some of them, evidently, cat people
# - our task is to classify them as such by looking at their face

# In[3]:


n_plots = 3
fig, axs = plt.subplots(n_plots, n_plots, figsize=(6, 6))

print(images.shape)
 
text_label = ['dog', 'cat']
for i in range(n_plots):
    for j in range(n_plots):
        axs[i, j].imshow(images[i*n_plots + j], cmap='gray');
        axs[i, j].axis('off')
        axs[i, j].set_title(text_label[labels[i*n_plots + j]])


# # Let's try the simple perceptron

# In[6]:


from perceptron import train

labsym = labels*2 - 1
w = train(images.reshape(n_train, -1), labsym)


# In[8]:


labsym_est = np.sign(images.reshape(n_train, -1) @ w)
labels_est = np.int8((labsym_est + 1) / 2)
n_correct = np.sum(labsym_est == labsym)
print('The perceptron correctly classifies %d out of %d training images' % (n_correct, n_train))


# In[9]:


fig, axs = plt.subplots(n_plots, n_plots, figsize=(6, 6))

for i in range(n_plots):
    for j in range(n_plots):
        axs[i, j].imshow(images[i*n_plots + j], cmap='gray');
        axs[i, j].axis('off')
        axs[i, j].set_title('T:' + text_label[labels[i*n_plots + j]] \
                            + ' P:' + text_label[labels_est[i*n_plots + j]])


# # Phenomenal and strange!
# 
# ## We also have a test set...
# 

# In[10]:


labsym_test = labels_test*2 - 1
labsym_test_est = np.sign(images_test.reshape(n_test, -1) @ w)

n_correct_test = np.sum(labsym_test_est == labsym_test)
print('The perceptron correctly classifies %d out of %d test images' % (n_correct_test, n_test))


# # What went wrong?
# 
# We built a classifier which does not _generalize_ well.
# 
# But why? How can it work so well on the training data??

# The answer is that I tricked you. I doctored the images in an impercetible way. Click here if you want to know how.

# First, I generated two images of pure noise.

# In[11]:


scale = 2
noise = np.zeros((2, *image_size))
noise[0] = scale * np.random.randn(*image_size)
noise[1] = scale * np.random.randn(*image_size)

fig, axs = plt.subplots(1, 2, figsize=(6, 3))
axs[0].imshow(noise[0], cmap='gray')
axs[0].axis('off')
axs[1].imshow(noise[1], cmap='gray')
axs[1].axis('off');


# Then I 
# - randomly assigned labels to training images
# - to every random dog person I added noise[0]
# - to every random cat person I added noise[1]

# In[12]:


for i in range(n_train):
    images[i] = plt.imread(result[shuffle_idx[i]]).mean(axis=2)
    labels[i] = np.round(np.random.rand())
    images[i] += noise[labels[i]]


# # To really understand why it works
# 
# - The key is in angles! High-dimensional Gaussian random vectors are near-orthogonal...
# - This can be understood already by the law of large numbers!

# In[13]:


def angle(x, y):
    return 180 / np.pi * np.arccos(np.dot(x.flatten(), y.flatten()) / \
                                   np.linalg.norm(x.flatten()) / \
                                   np.linalg.norm(y.flatten())
                                  )

angle(noise[0], images[0])


# but $\langle \mathbf{x} + \mathbf{e_0}, \mathbf{e_0} \rangle = \langle \mathbf{x}, \mathbf{e_0} \rangle + \langle \mathbf{e_0}, \mathbf{e_0} \rangle \approx 0 + \| \mathbf{e_0} \|^2$
# 
# while $\langle \mathbf{x} + \mathbf{e_1}, \mathbf{e_0} \rangle = \langle \mathbf{x}, \mathbf{e_0} \rangle + \langle \mathbf{e_1}, \mathbf{e_0} \rangle \approx 0 + 0$
# 

# # Clever Hans (der Kluge Hans)
# <br>
# <center>
#     <img src="https://upload.wikimedia.org/wikipedia/commons/e/e3/CleverHans.jpg" width="50%" />
# </center>
# 
# I very much like the Wikipedia description:
# 
#     Hans was a horse owned by Wilhelm von Osten, who was a gymnasium mathematics teacher, an 
#     amateur horse trainer, phrenologist, and considered a mystic.

# # Generalization
# 
# - We say that our classifier or predictor **fails to generalize** (from training to test cases)
# - Since we have great train performance, this is an example of **overfitting**: we achieve extremely good performance on the train set but it does not say anything about the test set (in fact, it is detrimental)
# - Unlike classical overfitting to independent noise, our dataset was additionally doctored!
# 
# _Generalization is the ability of a pattern recognition system to do well on data that it hasn't seen in the training phase._

# # Structure of a PR dataset
# 
# Train, validate, test!
# 
# (Not that it can help with doctored data...)

# Show kNN errors on training _and_ test (or validation) sets at the same time
# 
# Also: perceptron bias!

# # Part 2: Modeling Knowledge

# # Predicting Snail Sex from Number of Rings
# 
# A gorgeous example of an abalone snail
# 
# <img src="https://static01.nyt.com/images/2022/03/08/science/24sci-abalone/24sci-abalone-videoSixteenByNine3000.jpg" width="70%"/>
# 
# <font size="3">Picture from New York Times</font>

# In[14]:


import pandas as pd

snails = pd.read_csv('/Users/dokman0000/Downloads/abalone.data')
snails.columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 
                  'Shucked weight', 'Viscera weight', 'Shell weight', 
                  'Rings']
sex = np.array(snails['Sex'])
number_of_rings = np.array(snails['Rings'])
infant_idx = sex == 'I'
sex = sex[~infant_idx]
number_of_rings = number_of_rings[~infant_idx]
sex[sex == 'M'], sex[sex == 'F'] = -1, 1
sex = np.int8(sex)

males = np.zeros((30))
females = np.zeros((30))
for i in range(30):
    males[i] = sum(sex[number_of_rings == i] == -1)    
    females[i] = sum(sex[number_of_rings == i] == 1)
    
print(males > females)


# In[15]:


males[3] -= 2
males[4] -= 3
males[5] -= 5
males[6] -= 5
males[7] -= 15
males[8] -= 45
males[9] -= 36
males[10] -= 40
males[11] -= 15
females[6] += 5
females[7] += 10
females[13] += 10
females[14] += 5
females[15] += 12
females[16] += 10
females[26] += 2
print(males > females)
print(sum(males))
print(sum(females))


# In[16]:


fig, ax = plt.subplots(1, 1, figsize=(8, 5))

ax.bar(np.arange(30) - 0.2, males, width=0.4)
ax.bar(np.arange(30) + 0.2, females, width=0.4)

ax.legend(['Male', 'Female'], frameon=False)
ax.set_xlabel('Number of rings')
ax.set_ylabel('Number of snails');


# # What is the "best" classifier?
# 
# - We have a binary prediction problem
# - We _define_ "best predictor" as the one that makes the fewest mistakes
#     - This is called _the minimum error rule_
# - There are two kinds of mistakes:
#     - Predicting a male when the true snail is female
#     - Predicting a female when the true snail is male

# In[17]:


fig, ax = plt.subplots(1, 1, figsize=(6, 3))
ax.plot(-np.array([-1, 1])[np.int8(males > females)[3:]])


# # A good but impractical rule
# 
# - a good rule with compelling intuition
# - alas, to apply it, we need to measure the entire population of snails
# - impractical and largely defeats the purpose
# - if we only measure a subset, then what do we believe when we apply the thing in the real world?
# - we need (domain or otherwise) knowledge about the problem / population
#     - this make prediction possible
# 

# # Modeling knowledge
# 
# (cf. PPA book)
# 
# - Knowledge about the population is what makes prediction without enumeration possible
# - There are different ways to represent "knowledge" (or model populations)
# - In machine learning we model populations by probability distributions
#     - We assume that both the training data and the data we will apply our algorithms to are independent samples from a joint distribution of objects and labels
# 
# 
# - You could simply flip to the next slide but I'd like you to contemplate the meaning of this. This is really a semi-religious assumption which seems to be extremely useful in practice

# In[18]:


from scipy.stats import skewnorm

# btw, do I need to explain what a probability distribution / density function is?

l1, s1, a1 = 7.40, 4.48, 3.12
l2, s2, a2 = 7.63, 4.67, 4.34

x = np.linspace(skewnorm.ppf(0.001, a1, loc=l1, scale=s1),
                skewnorm.ppf(0.999, a1, loc=l1, scale=s1), 
                100)

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
line1, = axs[0].plot(x, skewnorm.pdf(x, a1, loc=l1, scale=s1),
       'b-', lw=4, alpha=0.6, label='skewnorm pdf')
line2, = axs[0].plot(x, skewnorm.pdf(x, a2, loc=l2, scale=s2),
       'r-', lw=4, alpha=0.6, label='skewnorm pdf')
text = axs[0].text(15, 0.12, '0.000')

axs[0].set_xlabel('Number of rings')
axs[0].set_ylabel('Probability density')
axs[0].set_ylim(0, 0.154)

thr0 = 15
thrline, = axs[0].plot([thr0, thr0], [0, 0.20])

def update(thr=thr0):
    err1 = skewnorm.cdf(thr, a2, loc=l2, scale=s2)
    err2 = 1 - skewnorm.cdf(thr, a1, loc=l1, scale=s1)
    
    p_error = (err1 + err2) / 2
    
    thrline.set_xdata([thr, thr])
    axs[1].plot(err1, 1 - err2, 'b.')
    text.set_text('$\mathbb{P}_{\mathrm{err}}$ = %0.3f' % (p_error,))
    fig.canvas.draw_idle()
    


# In[19]:


fig, axs = plt.subplots(1, 2, figsize=(10, 4))
line1, = axs[0].plot(x, skewnorm.pdf(x, a1, loc=l1, scale=s1),
       'b-', lw=4, alpha=0.6, label='skewnorm pdf')
line2, = axs[0].plot(x, skewnorm.pdf(x, a2, loc=l2, scale=s2),
       'r-', lw=4, alpha=0.6, label='skewnorm pdf')
text = axs[0].text(15, 0.12, '0.000')

axs[0].set_xlabel('Number of rings')
axs[0].set_ylabel('Probability density')
axs[0].set_ylim(0, 0.154)

thrline, = axs[0].plot([thr0, thr0], [0, 0.20])
interact(update, thr=(5.0, 22.5, 0.1));


# # The Dogma
# 
# - We assume that there is some joint probability distribution over patterns and labels, $p_{X, Y}(x, y)$
# - (When all is clear we write simply $p(x, y)$ for simplicity)
# 
# - In a week or two we will talk about supervised learning where the training set $\{ (x_i, y_i) \}_{i = 1}^n$ consists of independent samples from $p(x, y)$, but the true $p$ is unknown 
# - Here we assume that $p$ is known—even so there are many important questions to answer
# 

# # Prediction from statistical models
# 
# - Let's further formalize the minimum error rule for binary classification
# - PPA: It is convenient both mathematically and conceptually to specify the joint distribution via class-conditional probabilities
# 
# - We assume that $Y$ has _a priori_ probabilities
# 
# $$
#     p_0 = \mathbb{P}[Y = 0] \quad \quad p_1 = \mathbb{P}[Y = 1]
# $$
# 
# - $p_0$ and $p_1$ are proportions of the two classes in the population
#     - if we draw a large number $n$ of samples from $p$ there will be approximatialy $p_0 n$ labels $0$ and $p_1 n$ labels $1$
# - In the snail example we had $p_0 = p_1 = \frac{1}{2}$ ("balanced" classes)
# 

# # Prediction from statistical models (cont'd)
# 
# - The patterns are modeled by a random vector $X$ (jointly distributed with labels $Y$)
# - The meaning of "jointly": the distribution of $X$ depends on whether $Y$ is 0 or 1
# - $\Rightarrow$ there are two different statistical models for the data, one for $Y = 1$ and another for $Y = -1$
# - These statistical models are the conditional probabilities
# 
# 
# $$
#     p(x | Y = 0) \quad \quad \text{and} \quad \quad p(x | Y = 1)
# $$
# 
# - Generalization to different kinds of labels is straightforward
# - $p(x | Y = y)$ called _generative models_ or _likelihood functions_
# - the joint probability is then
# 
# $$
#     p(x, y) = p(x | Y = y) p(Y = y)
# $$

# # Example: signal vs noise
# 
# - Suppose that 
#     - when $Y = 0$ we observe $\omega$, where $\omega \sim \mathcal{N}(0, 1)$
#     - when $Y = 1$ we observe $\omega + s$ for a deterministic scalar $s$
# - Then
# $$
# \begin{align}
#     p(x \,|\, Y = 0) &= \mathcal{N}(0, 1) = \tfrac{1}{\sqrt{2\pi}} e^{-\frac{x^2}{2}} \\
#     p(x \,|\, Y = 1) &= \mathcal{N}(s, 1) = \tfrac{1}{\sqrt{2\pi}} e^{-\frac{(x - s)^2}{2}}
# \end{align}
# $$
# - The shift $s$ determines how hard it is to predict $Y$
# - Expand a bit and add a figure

# # Prediction via optimization
# 
# We want to compute _optimal_ predictions, and the way to do it is by _optimization_.
# 
# - a predictor is a rule, a formula, or most generally, an algorithm
# - $\Rightarrow$ the quest for an optimal predictor involves optimization over algorihtms
# - algorithms (or functions) are maps from the set of patterns, $\mathcal{X}$, to the set of labels, $\mathcal{Y}$

# What we did earlier is
# 
# $$
#   \mathcal{A} = \text{the set of algorithms} = \left\{ f(x) = \begin{cases} 0 & \text{if}~ x \leq \eta \\ 1 & \text{if}~ x > \eta \end{cases} \ \bigg| \ \eta \in \mathbb{R} \right \}
# $$
# 

# and the optimization problem was
# 
# $$
# \begin{align}
#       \text{find}~f~\in~\mathcal{A} ~\text{which minimizes} ~
#       \mathbb{P}(\text{mistake})
#       &=
#       \mathbb{P}(f(X) = 0 \, |\, Y = 1) \mathbb{P}(Y = 1)
#       +
#       \mathbb{P}(f(X) = 1 \, |\, Y = 0) \mathbb{P}(Y = 0) \\
#       &=
#       \mathbb{P}(X \leq \eta \, |\, Y = 1) \mathbb{P}(Y = 1)
#       +
#       \mathbb{P}(X > \eta \, |\, Y = 1) \mathbb{P}(Y = 0) \\
# \end{align}
# $$ 

# # Our prediction is random
# 
# - We obtain our prediction by applying an function / algorithm $f$ to $X$
# - Recall that $X$ is modeled as a random draw from $p(x) = \int p(x, y) dy$
# - The common notation for an estimate of some quantity is with a hat. We will thus write
# 
# $$
# \hat{Y} \equiv \hat{Y}(X) \equiv f(X)
# $$
# 
# - most commonly we will simply write $\hat{Y}$
# - it is understood that $\hat{Y}$ is an _estimate_ or a _prediction_ calculated from the observed pattern $X$
# - we will often look at binary classification where $\mathcal{Y} = \{0, 1\}$ or $\mathcal{Y} = \{-1, 1\}$
# 

# # Quiz
# 
# - How would $k$-means perform on catness and dogness? Could it be fooled as well?
# - Recall our set of threshold-based clasifiers:
# 
# $$
#   \mathcal{A} = \text{the set of algorithms} = \left\{ f(x) = \begin{cases} 0 & \text{if}~ x \leq \eta \\ 1 & \text{if}~ x > \eta \end{cases} \ \bigg| \ \eta \in \mathbb{R} \right \}
# $$
# - Write down in a similar form the set of all perceptrons that work on 100-dimensional vectors
# 
# 

# # Next time
# 
# - Still towards optimal pattern recognition: the loss function 
# - The amazing effectiveness of likelihood ratios
# - A couple of curious computational examples

# In[ ]:





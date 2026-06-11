# ML From Scratch — Complete Reference

A from-scratch implementation of 47 machine learning algorithms in pure Python (NumPy only).
Built in dependency order, every module follows a consistent interface, and every implementation
is verified against a known-good reference.

---

## Why This Exists

Reading about algorithms and implementing them are two different things. This project forces
mechanical understanding — you cannot fake a working backward pass. By the end, every algorithm
in this library is something you can debug, extend, and explain at the level of the math, not
just the API.

The implementations are also directly mapped to a production project roadmap (P1–P15), so each
algorithm you build serves a real system you will eventually ship.

---

## Stack

- **Language:** Python 3.10+
- **Only allowed dependency:** NumPy
- **Testing:** pytest, verified against sklearn / PyTorch where applicable
- **Interface:** every model exposes `fit`, `predict`, `transform`, `save`, `load`, `summary`

---

## Project Structure

```
ml_scratch/
├── core/
├── classical/
├── unsupervised/
├── probabilistic/
├── deep/
├── nlp/
├── rl/
├── self_supervised/
├── utils/
└── tests/
```

---

## Module Reference

### `core/` — Foundation Everything Else Depends On

| File | What It Does |
|------|-------------|
| `autograd.py` | Builds a computation graph during the forward pass and runs backpropagation through it to compute gradients |
| `tensor.py` | NumPy array wrapper that tracks every operation so gradients can flow backward automatically |
| `activations.py` | Nonlinear functions applied after each layer — relu, sigmoid, tanh, gelu, softmax — plus their derivatives |
| `losses.py` | Measures how wrong the model's predictions are — mse, cross-entropy, kl divergence, huber, hinge |
| `optimizers.py` | Updates weights using gradients to minimize loss — SGD with momentum, Adam, RMSProp, AdaGrad |
| `initializers.py` | Sets starting weight values using Glorot, He, or orthogonal schemes so training does not die immediately |
| `metrics.py` | Measures model quality after training — accuracy, f1, rmse, r2, roc-auc, sharpe ratio, VaR |

**Build order:** `autograd.py` → `tensor.py` → `activations.py` → `losses.py` → `optimizers.py` → `initializers.py` → `metrics.py`

Everything else in the library depends on this layer. Do not move to any other module until
the autograd engine passes numerical gradient checking.

---

### `classical/` — Supervised Learning

#### Linear Models

| File | What It Does |
|------|-------------|
| `linear_regression.py` | Fits a line through data to predict continuous values — normal equation for small data, gradient descent for large |
| `logistic_regression.py` | Predicts probability of a binary outcome by passing a linear combination through sigmoid |
| `ridge.py` | Linear regression with L2 penalty on weights — closed form via `(X^T X + λI)^-1 X^T y` |
| `lasso.py` | Linear regression with L1 penalty — solved via coordinate descent, drives irrelevant weights to exactly zero |

#### Tree Models

| File | What It Does |
|------|-------------|
| `decision_tree.py` | Recursively splits data on the feature and threshold that maximizes information gain until leaves are pure |
| `random_forest.py` | Trains 100+ decision trees on bootstrap samples with random feature subsets, aggregates by majority vote |
| `gradient_boosting.py` | Builds trees sequentially where each new tree fits the negative gradient (residuals) of the previous ensemble |

#### Other Classical

| File | What It Does |
|------|-------------|
| `knn.py` | Classifies a point by majority vote of its k nearest neighbors — lazy learner, no training step |
| `naive_bayes.py` | Classifies using Bayes theorem assuming all features are conditionally independent given the class |
| `svm.py` | Finds the maximum-margin hyperplane between classes — solved via SMO algorithm, supports rbf and polynomial kernels |

**Project mapping:** P3 (End-to-End ML System) — implement Decision Tree, Random Forest, Gradient Boosting before building P3.

---

### `unsupervised/` — Learning Without Labels

#### Clustering

| File | What It Does |
|------|-------------|
| `kmeans.py` | Groups n points into k clusters by iteratively assigning points to nearest centroid and recomputing centroids |
| `dbscan.py` | Groups points by density reachability — no need to specify k, outliers automatically become noise |
| `gmm.py` | Fits k Gaussian distributions to data via EM algorithm, allowing soft probabilistic cluster membership |

#### Dimensionality Reduction

| File | What It Does |
|------|-------------|
| `pca.py` | Rotates data to the axes of maximum variance via eigendecomposition then drops the weakest components |
| `umap.py` | Compresses high-dimensional data to 2-3 dimensions by preserving the topological structure of the manifold |

#### Anomaly Detection

| File | What It Does |
|------|-------------|
| `isolation_forest.py` | Flags anomalies as points that get isolated fastest when random splits are applied — anomalies are few and different |

**Project mapping:** P6 (Streaming + Real-Time ML) uses Isolation Forest. P8 (Signal Detection) uses PCA and GMM.

---

### `probabilistic/` — Uncertainty, Sequences, and Time

| File | What It Does |
|------|-------------|
| `hmm.py` | Models sequences where hidden states generate observable outputs — forward-backward for learning, Viterbi for decoding |
| `kalman_filter.py` | Estimates the true signal from noisy measurements by alternating between prediction and correction steps |
| `gaussian_process.py` | Predicts outputs as a full probability distribution with uncertainty bounds, not just a point estimate |
| `bayesian_network.py` | Encodes conditional independence relationships between variables in a directed acyclic graph, supports variable elimination |
| `crf.py` | Labels sequences by considering the whole output sequence jointly — neighboring labels influence each other |
| `arima.py` | Forecasts time series using a combination of past values, differencing for stationarity, and past forecast errors |
| `garch.py` | Models financial volatility where large price moves cluster together — estimates time-varying variance |
| `monte_carlo.py` | Estimates probabilities, expectations, and integrals by sampling random scenarios at scale |

**Project mapping:** P8 (Signal Detection) uses HMM, Kalman Filter, Gaussian Process, ARIMA, GARCH. P14 (Risk Management) uses Monte Carlo and Kalman Filter. P15 (Multi-Agent) uses Bayesian Network.

---

### `deep/layers/` — Neural Network Primitives

| File | What It Does |
|------|-------------|
| `base.py` | Abstract interface every layer must implement — forward pass, backward pass, parameter access |
| `dense.py` | Multiplies input by weight matrix, adds bias, applies activation — the fundamental building block |
| `conv2d.py` | Slides learnable filters across spatial input to detect local patterns — implemented via im2col for efficiency |
| `pooling.py` | Shrinks spatial dimensions by taking the max or average over a local window |
| `recurrent.py` | Processes sequences by maintaining a hidden state passed from each timestep to the next |
| `lstm.py` | Recurrent layer with forget, input, and output gates that control what information to retain across long sequences |
| `gru.py` | Simplified LSTM with reset and update gates — fewer parameters, comparable long-range memory |
| `attention.py` | Computes scaled dot-product attention — each position attends to every other position with learned weights |
| `normalization.py` | Rescales layer activations to have zero mean and unit variance so deep networks train stably |
| `dropout.py` | Randomly zeros a fraction of activations during training — forces the network to not rely on any single neuron |
| `embedding.py` | Maps integer token indices to dense learnable vectors — the entry point for all sequence models |

---

### `deep/models/` — Full Architectures

| File | What It Does |
|------|-------------|
| `mlp.py` | Stack of dense layers — the universal function approximator, baseline for all tabular deep learning |
| `cnn.py` | Alternating conv and pooling layers that extract hierarchical spatial features from images |
| `rnn.py` | Stacked recurrent layers for modeling temporal sequences — baseline before LSTM |
| `autoencoder.py` | Encoder compresses input to a bottleneck, decoder reconstructs it — learns compact representations |
| `vae.py` | Autoencoder that learns a smooth, continuous latent space via reparameterization and KL divergence loss |
| `gan.py` | Generator and discriminator trained adversarially — generator fools the discriminator, discriminator catches the generator |
| `diffusion.py` | Learns to reverse a noise-adding Markov chain — generates data by iteratively denoising from Gaussian noise |

**Project mapping:** P4 (ML Inference Service) uses MLP/CNN. P5 (RAG System) uses the Transformer. P11 (Multi-Modal) uses CNN + Transformer + VAE.

---

### `nlp/` — Language and Sequence Understanding

| File | What It Does |
|------|-------------|
| `word2vec.py` | Trains word embeddings via skip-gram or CBOW so semantically similar words are close in vector space |
| `transformer.py` | Full encoder-decoder architecture built entirely on multi-head attention — no recurrence, fully parallelizable |
| `bert.py` | Transformer encoder pretrained by masking 15% of tokens and predicting them — bidirectional context |
| `gpt.py` | Transformer decoder pretrained by predicting the next token autoregressively — causal, left-to-right only |

**Project mapping:** P5 (RAG System) — implement attention and transformer before building the retrieval pipeline. Understanding these from scratch lets you debug retrieval quality at the embedding level.

---

### `rl/` — Reinforcement Learning

#### Agents

| File | What It Does |
|------|-------------|
| `q_learning.py` | Tabular agent that learns action-value function via TD updates — works only for small discrete state spaces |
| `dqn.py` | Q-learning with a neural network function approximator, experience replay buffer, and target network for stability |
| `ppo.py` | Policy gradient method that clips the surrogate objective to prevent catastrophically large updates |
| `sac.py` | Off-policy actor-critic that maximizes reward plus a entropy bonus — naturally explores without epsilon-greedy |
| `mcts.py` | Plans by simulating future trajectories from the current state and backing up value estimates via tree search |

#### Memory

| File | What It Does |
|------|-------------|
| `replay_buffer.py` | Ring buffer storing (state, action, reward, next_state, done) tuples — sampled randomly during DQN training |

**Project mapping:** P15 (Autonomous Multi-Agent) uses PPO or SAC for agent policy learning and MCTS for planning.

---

### `self_supervised/` — Learning Without Labels at Scale

| File | What It Does |
|------|-------------|
| `simclr.py` | Trains an encoder by pulling two augmented views of the same image together in latent space and pushing others apart |
| `mae.py` | Trains an encoder by masking 75% of input patches and learning to reconstruct the missing ones |

**Project mapping:** P11 (Multi-Modal Intelligence) — self-supervised pretraining on unlabeled imagery before fine-tuning.

---

### `utils/` — Supporting Infrastructure

| File | What It Does |
|------|-------------|
| `data.py` | Splits data into train/val/test sets, creates stratified folds, and generates shuffled mini-batches |
| `preprocessing.py` | Normalizes features, one-hot encodes categoricals, imputes missing values, applies log transforms |
| `validators.py` | Checks input shapes and dtypes at every model entry point so errors are caught early with clear messages |
| `serialization.py` | Saves trained weights and hyperparameters to disk as numpy archives, loads them back for inference |

---

### `tests/` — Verification

| File | What It Verifies |
|------|----------------|
| `test_classical.py` | Classical models match sklearn output on identical data within numerical tolerance |
| `test_unsupervised.py` | Clusters are geometrically correct and PCA reconstructions minimize MSE |
| `test_probabilistic.py` | Sequence models recover known parameters from synthetic data generated with known ground truth |
| `test_deep.py` | Gradients are correct via numerical gradient checking — perturb each weight by 1e-5, compare to analytical grad |
| `test_nlp.py` | Attention scores sum to 1, embeddings have correct shape, transformer output matches expected dimensions |
| `test_rl.py` | Agents improve cumulative reward over training episodes on a simple reference environment |
| `test_self_supervised.py` | Learned representations cluster semantically similar inputs closer than random representations |

---

## Build Order

Build strictly in this sequence. Each phase depends on the previous one being correct and tested.

```
Phase 0 — Core (Week 1-2)
  autograd → tensor → activations → losses → optimizers → initializers → metrics

Phase 1 — Classical ML (Week 3-6)
  linear_regression → logistic_regression → ridge → lasso
  decision_tree → random_forest → gradient_boosting
  knn → naive_bayes → svm

Phase 2 — Unsupervised (Week 7-9)
  kmeans → dbscan → gmm
  pca → umap
  isolation_forest

Phase 3 — Probabilistic (Week 10-14)
  arima → garch → kalman_filter
  hmm → crf
  gaussian_process → bayesian_network → monte_carlo

Phase 4 — Deep Layers (Week 15-20)
  base → dense → normalization → dropout
  conv2d → pooling
  recurrent → lstm → gru
  embedding → attention

Phase 5 — Deep Models (Week 21-26)
  mlp → cnn → rnn
  autoencoder → vae → gan → diffusion

Phase 6 — NLP (Week 27-32)
  word2vec → transformer → bert → gpt

Phase 7 — RL (Week 33-38)
  q_learning → replay_buffer → dqn → ppo → sac → mcts

Phase 8 — Self-Supervised (Week 39-42)
  simclr → mae
```

---

## Interface Contract

Every class in this library follows the same interface so models are interchangeable in pipelines.

```python
class BaseModel:
    def fit(self, X, y=None) -> self
        # Train the model. y is None for unsupervised.

    def predict(self, X) -> np.ndarray
        # Return predictions. For classifiers returns class labels.
        # For regressors returns continuous values.

    def predict_proba(self, X) -> np.ndarray
        # For probabilistic classifiers only. Returns class probabilities.

    def transform(self, X) -> np.ndarray
        # For unsupervised and dimensionality reduction only.

    def fit_transform(self, X) -> np.ndarray
        # Fit then transform in one call.

    def save(self, path: str) -> None
        # Serialize weights and hyperparameters to disk as .npz

    def load(self, path: str) -> self
        # Deserialize weights from disk.

    def summary(self) -> None
        # Print architecture, parameter count, and hyperparameters.
```

---

## Project Mapping

| Project | Algorithms Needed |
|---------|------------------|
| P1 — alphastats | linear_regression, basic stats, hypothesis testing |
| P3 — End-to-End ML | decision_tree, random_forest, gradient_boosting |
| P4 — ML Inference Service | autograd, mlp, cnn |
| P5 — RAG System | word2vec, attention, transformer |
| P6 — Streaming + Real-Time ML | isolation_forest, online linear regression |
| P8 — Signal Detection | hmm, kalman_filter, gaussian_process, arima, garch, pca, gmm |
| P11 — Multi-Modal Intelligence | cnn, transformer, crf, umap, bayesian_network, vae |
| P14 — Risk Management | kalman_filter, monte_carlo, garch, gaussian_process, gmm |
| P15 — Multi-Agent System | ppo, sac, mcts, bayesian_network, transformer |

---
# ML From Scratch — Complete Learning Resources

Every book, paper, course, and reference needed to implement all 47 algorithms.
Organized by module. Free resources marked with (FREE).

---

## Foundational Books — Read These First

These four books cover everything. Read them in this order alongside implementation.

| Book | Author | What It Covers | When To Read |
|------|--------|---------------|--------------|
| Mathematics for Machine Learning | Deisenroth, Faisal, Ong | Linear algebra, calculus, probability, optimization, PCA, GMM | Before Phase 1 |
| Machine Learning: An Algorithmic Perspective | Marsland | Every classical algorithm derived from scratch with Python code | Phase 1–3 |
| Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow | Géron | Practical implementation, full ML pipeline, deep learning intro | Phase 1–5 |
| Deep Learning | Goodfellow, Bengio, Courville | Full theory of deep learning, backprop, CNNs, RNNs, optimization | Phase 4–8 |

All four are available free:
- MML: https://mml-book.github.io (FREE)
- Deep Learning: https://www.deeplearningbook.org (FREE)

---

## Core — Autograd and Tensor

### Primary Resources

| Resource | Type | Link |
|----------|------|------|
| Micrograd by Andrej Karpathy | Video + Code | https://github.com/karpathy/micrograd |
| The spelled-out intro to neural networks and backpropagation | YouTube | https://youtu.be/VMj-3S1tku0 |
| Automatic Differentiation in Machine Learning: a Survey | Paper | https://arxiv.org/abs/1502.05767 |
| Backpropagation Applied to Handwritten Zip Code Recognition | Paper (LeCun 1989) | http://yann.lecun.com/exdb/publis/pdf/lecun-89e.pdf |
| Calculus on Computational Graphs: Backpropagation | Blog | https://colah.github.io/posts/2015-08-Backprop |

### For Optimizers

| Resource | Type | Link |
|----------|------|------|
| Adam: A Method for Stochastic Optimization | Paper | https://arxiv.org/abs/1412.6980 |
| An Overview of Gradient Descent Optimization Algorithms | Blog | https://www.ruder.io/optimizing-gradient-descent |
| Why Momentum Really Works | Blog | https://distill.pub/2017/momentum |

---

## Classical ML

### Linear and Logistic Regression

| Resource | Type | Link |
|----------|------|------|
| The Elements of Statistical Learning | Book (FREE) | https://hastie.su.domains/ElemStatLearn |
| An Introduction to Statistical Learning | Book (FREE) | https://www.statlearning.com |
| CS229 Lecture Notes — Linear Regression | Notes (FREE) | https://cs229.stanford.edu/notes2022fall/main_notes.pdf |
| CS229 Lecture Notes — Logistic Regression | Notes (FREE) | Same as above |
| Regression Shrinkage and Selection via the Lasso | Paper (Tibshirani 1996) | https://www.jstor.org/stable/2346178 |
| Ridge Regression: Biased Estimation for Nonorthogonal Problems | Paper (Hoerl & Kennard) | Classic — available via Google Scholar |

### Decision Trees and Ensembles

| Resource | Type | Link |
|----------|------|------|
| Classification and Regression Trees | Book | Breiman, Friedman, Olshen, Stone 1984 — CART original |
| Random Forests | Paper (Breiman 2001) | https://link.springer.com/article/10.1023/A:1010933404324 |
| Greedy Function Approximation: A Gradient Boosting Machine | Paper (Friedman 2001) | https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-function-approximation-a-gradient-boosting-machine/10.1214/aos/1013203451.full |
| XGBoost: A Scalable Tree Boosting System | Paper (Chen & Guestrin 2016) | https://arxiv.org/abs/1603.02754 |
| A Decision-Theoretic Generalization of On-Line Learning | Paper (AdaBoost — Freund & Schapire) | https://www.sciencedirect.com/science/article/pii/S002200009791504X |
| StatQuest: Decision Trees | YouTube (FREE) | https://youtu.be/_L39rN6gz7Y |
| StatQuest: Random Forests | YouTube (FREE) | https://youtu.be/J4Wdy0Wc_xQ |
| StatQuest: Gradient Boosting | YouTube (FREE) | https://youtu.be/3CC4N4z3GJc |

### SVM

| Resource | Type | Link |
|----------|------|------|
| A Training Algorithm for Optimal Margin Classifiers | Paper (Boser, Guyon, Vapnik 1992) | Original SVM paper — search via Google Scholar |
| A Tutorial on Support Vector Machines | Paper (Burges 1998) | https://link.springer.com/article/10.1023/A:1022627411411 |
| Sequential Minimal Optimization | Paper (Platt 1998) | https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-98-14.pdf |
| CS229 SVM Notes | Notes (FREE) | https://cs229.stanford.edu/notes2022fall/main_notes.pdf |

### KNN and Naive Bayes

| Resource | Type | Link |
|----------|------|------|
| Nearest Neighbor Pattern Classification | Paper (Cover & Hart 1967) | IEEE Transactions — search via Google Scholar |
| A Probabilistic Theory of Pattern Recognition | Book | Devroye, Gyorfi, Lugosi |
| CS229 Generative Learning Algorithms | Notes (FREE) | https://cs229.stanford.edu/notes2022fall/main_notes.pdf |

---

## Unsupervised Learning

### Clustering

| Resource | Type | Link |
|----------|------|------|
| Algorithm AS 136: A K-Means Clustering Algorithm | Paper (Hartigan & Wong 1979) | Original K-Means paper |
| A Density-Based Algorithm for Discovering Clusters (DBSCAN) | Paper (Ester et al. 1996) | https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf |
| Maximum Likelihood from Incomplete Data via the EM Algorithm | Paper (Dempster, Laird, Rubin 1977) | https://www.jstor.org/stable/2984875 — foundational EM paper |
| Pattern Recognition and Machine Learning | Book | Bishop 2006 — Chapter 9 covers GMM and EM exhaustively |
| StatQuest: K-Means Clustering | YouTube (FREE) | https://youtu.be/4b5d3muPQmA |
| StatQuest: GMM and EM | YouTube (FREE) | https://youtu.be/REypj2sy_5U |

### Dimensionality Reduction

| Resource | Type | Link |
|----------|------|------|
| On Lines and Planes of Closest Fit to Systems of Points | Paper (Pearson 1901) | Original PCA paper |
| A Tutorial on Principal Component Analysis | Paper (Shlens) | https://arxiv.org/abs/1404.1100 |
| Visualizing Data using t-SNE | Paper (van der Maaten & Hinton 2008) | https://jmlr.org/papers/v9/vandermaaten08a.html |
| UMAP: Uniform Manifold Approximation and Projection | Paper (McInnes et al. 2018) | https://arxiv.org/abs/1802.03426 |
| StatQuest: PCA Step by Step | YouTube (FREE) | https://youtu.be/FgakZw6K1QQ |

### Anomaly Detection

| Resource | Type | Link |
|----------|------|------|
| Isolation Forest | Paper (Liu, Ting, Zhou 2008) | https://ieeexplore.ieee.org/document/4781136 |
| Anomaly Detection: A Survey | Paper (Chandola et al. 2009) | https://dl.acm.org/doi/10.1145/1541880.1541882 |

---

## Probabilistic Models

### Hidden Markov Models

| Resource | Type | Link |
|----------|------|------|
| A Tutorial on Hidden Markov Models | Paper (Rabiner 1989) | https://ieeexplore.ieee.org/document/18626 — the definitive HMM tutorial |
| Speech and Language Processing | Book (FREE) | Jurafsky & Martin — Chapter 8 covers HMM for NLP https://web.stanford.edu/~jurafsky/slp3 |
| Pattern Recognition and Machine Learning | Book | Bishop — Chapter 13 covers HMM |

### Kalman Filter

| Resource | Type | Link |
|----------|------|------|
| A New Approach to Linear Filtering and Prediction Problems | Paper (Kalman 1960) | https://asmedigitalcollection.asme.org/fluidsengineering/article/82/1/35/397706 |
| How a Kalman Filter Works in Pictures | Blog (FREE) | https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures |
| Kalman and Bayesian Filters in Python | Book (FREE) | https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python |
| Understanding the Basis of the Kalman Filter | Paper | https://ieeexplore.ieee.org/document/6279585 |

### Gaussian Processes

| Resource | Type | Link |
|----------|------|------|
| Gaussian Processes for Machine Learning | Book (FREE) | Rasmussen & Williams — http://www.gaussianprocess.org/gpml |
| A Visual Exploration of Gaussian Processes | Blog (FREE) | https://distill.pub/2019/visual-exploration-gaussian-processes |
| CS229 Gaussian Processes | Notes (FREE) | https://cs229.stanford.edu/notes2022fall/main_notes.pdf |

### Bayesian Networks

| Resource | Type | Link |
|----------|------|------|
| Probabilistic Graphical Models | Book | Koller & Friedman 2009 — the definitive PGM textbook |
| Probabilistic Graphical Models (Coursera) | Course (FREE audit) | https://www.coursera.org/specializations/probabilistic-graphical-models |
| Bayesian Reasoning and Machine Learning | Book (FREE) | Barber — http://www.cs.ucl.ac.uk/staff/d.barber/brml |

### CRF

| Resource | Type | Link |
|----------|------|------|
| Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data | Paper (Lafferty et al. 2001) | Original CRF paper — search via Google Scholar |
| An Introduction to Conditional Random Fields | Paper (Sutton & McCallum) | https://arxiv.org/abs/1011.4088 |

### Time Series — ARIMA and GARCH

| Resource | Type | Link |
|----------|------|------|
| Time Series Analysis: Forecasting and Control | Book | Box, Jenkins, Reinsel, Ljung — the ARIMA bible |
| Autoregressive Conditional Heteroscedasticity | Paper (Engle 1982) | Original ARCH paper — Journal of Econometrica |
| Generalized Autoregressive Conditional Heteroscedasticity | Paper (Bollerslev 1986) | Original GARCH paper |
| Forecasting: Principles and Practice | Book (FREE) | Hyndman & Athanasopoulos — https://otexts.com/fpp3 |
| StatQuest: ARIMA | YouTube (FREE) | https://youtu.be/Aw77aMLj9uM |

### Monte Carlo

| Resource | Type | Link |
|----------|------|------|
| Monte Carlo Statistical Methods | Book | Robert & Casella |
| An Introduction to MCMC for Machine Learning | Paper | https://link.springer.com/article/10.1023/A:1020281327116 |
| CS228 Probabilistic Graphical Models Notes | Notes (FREE) | https://ermongroup.github.io/cs228-notes |

---

## Deep Learning

### Backpropagation and MLP

| Resource | Type | Link |
|----------|------|------|
| Deep Learning (Goodfellow et al.) | Book (FREE) | https://www.deeplearningbook.org — Chapters 6-8 |
| Learning Representations by Backpropagating Errors | Paper (Rumelhart, Hinton, Williams 1986) | The backprop paper |
| Neural Networks and Deep Learning | Book (FREE) | Michael Nielsen — http://neuralnetworksanddeeplearning.com |
| Andrej Karpathy — Neural Networks: Zero to Hero | YouTube (FREE) | https://youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ |

### CNN

| Resource | Type | Link |
|----------|------|------|
| Gradient-Based Learning Applied to Document Recognition | Paper (LeCun et al. 1998) | http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf |
| ImageNet Classification with Deep CNNs (AlexNet) | Paper (Krizhevsky et al. 2012) | https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html |
| Very Deep Convolutional Networks (VGGNet) | Paper (Simonyan & Zisserman 2014) | https://arxiv.org/abs/1409.1556 |
| Deep Residual Learning for Image Recognition (ResNet) | Paper (He et al. 2015) | https://arxiv.org/abs/1512.03385 |
| CS231n: Convolutional Neural Networks for Visual Recognition | Course (FREE) | https://cs231n.github.io |
| Why does im2col work? | Blog (FREE) | Search "im2col convolution explanation" — Pete Warden's blog |

### RNN, LSTM, GRU

| Resource | Type | Link |
|----------|------|------|
| Learning Long-Term Dependencies with Gradient Descent is Difficult | Paper (Bengio et al. 1994) | The vanishing gradient paper |
| Long Short-Term Memory | Paper (Hochreiter & Schmidhuber 1997) | Original LSTM paper |
| Empirical Evaluation of Gated Recurrent Neural Networks | Paper (Chung et al. 2014) | https://arxiv.org/abs/1412.3555 — GRU paper |
| Understanding LSTM Networks | Blog (FREE) | https://colah.github.io/posts/2015-08-Understanding-LSTMs |
| The Unreasonable Effectiveness of Recurrent Neural Networks | Blog (FREE) | https://karpathy.github.io/2015/05/21/rnn-effectiveness |

### Normalization and Regularization

| Resource | Type | Link |
|----------|------|------|
| Batch Normalization | Paper (Ioffe & Szegedy 2015) | https://arxiv.org/abs/1502.03167 |
| Layer Normalization | Paper (Ba et al. 2016) | https://arxiv.org/abs/1607.06450 |
| Dropout: A Simple Way to Prevent Neural Networks from Overfitting | Paper (Srivastava et al. 2014) | https://jmlr.org/papers/v15/srivastava14a.html |

### Autoencoder, VAE, GAN

| Resource | Type | Link |
|----------|------|------|
| Auto-Encoding Variational Bayes | Paper (Kingma & Welling 2013) | https://arxiv.org/abs/1312.6114 — VAE original |
| Generative Adversarial Networks | Paper (Goodfellow et al. 2014) | https://arxiv.org/abs/1406.2661 — GAN original |
| Tutorial on Variational Autoencoders | Paper (Doersch 2016) | https://arxiv.org/abs/1606.05908 |
| From Autoencoder to Beta-VAE | Blog (FREE) | https://lilianweng.github.io/posts/2018-08-12-vae |
| GAN — A Beginner's Guide | Blog (FREE) | https://lilianweng.github.io/posts/2017-08-20-gan |

### Diffusion Models

| Resource | Type | Link |
|----------|------|------|
| Denoising Diffusion Probabilistic Models | Paper (Ho et al. 2020) | https://arxiv.org/abs/2006.11239 — DDPM original |
| Improved Denoising Diffusion Probabilistic Models | Paper (Nichol & Dhariwal 2021) | https://arxiv.org/abs/2102.09672 |
| What are Diffusion Models? | Blog (FREE) | https://lilianweng.github.io/posts/2021-07-11-diffusion-models |
| Diffusion Models from Scratch | YouTube (FREE) | https://youtu.be/a4Yfz2FxXiY |

---

## NLP

### Word Embeddings

| Resource | Type | Link |
|----------|------|------|
| Efficient Estimation of Word Representations in Vector Space (Word2Vec) | Paper (Mikolov et al. 2013) | https://arxiv.org/abs/1301.3781 |
| GloVe: Global Vectors for Word Representation | Paper (Pennington et al. 2014) | https://aclanthology.org/D14-1162 |
| The Illustrated Word2Vec | Blog (FREE) | https://jalammar.github.io/illustrated-word2vec |

### Transformer

| Resource | Type | Link |
|----------|------|------|
| Attention Is All You Need | Paper (Vaswani et al. 2017) | https://arxiv.org/abs/1706.03762 — the transformer paper |
| The Illustrated Transformer | Blog (FREE) | https://jalammar.github.io/illustrated-transformer |
| The Annotated Transformer | Blog + Code (FREE) | https://nlp.seas.harvard.edu/annotated-transformer |
| Let's build GPT from scratch | YouTube (FREE) | https://youtu.be/kCc8FmEb1nY — Karpathy |
| CS224N: Natural Language Processing with Deep Learning | Course (FREE) | https://web.stanford.edu/class/cs224n |

### BERT and GPT

| Resource | Type | Link |
|----------|------|------|
| BERT: Pre-training of Deep Bidirectional Transformers | Paper (Devlin et al. 2018) | https://arxiv.org/abs/1810.04805 |
| Language Models are Unsupervised Multitask Learners (GPT-2) | Paper (Radford et al. 2019) | https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf |
| The Illustrated BERT | Blog (FREE) | https://jalammar.github.io/illustrated-bert |
| The Illustrated GPT-2 | Blog (FREE) | https://jalammar.github.io/illustrated-gpt2 |
| nanoGPT | Code (FREE) | https://github.com/karpathy/nanoGPT |

---

## Reinforcement Learning

### Foundational

| Resource | Type | Link |
|----------|------|------|
| Reinforcement Learning: An Introduction | Book (FREE) | Sutton & Barto — http://incompleteideas.net/book/the-book-2nd.html — the RL bible |
| CS285: Deep Reinforcement Learning | Course (FREE) | https://rail.eecs.berkeley.edu/deeprlcourse |
| David Silver RL Lectures | YouTube (FREE) | https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ |

### Q-Learning and DQN

| Resource | Type | Link |
|----------|------|------|
| Q-Learning | Paper (Watkins & Dayan 1992) | Original Q-learning paper — Machine Learning journal |
| Human-level Control Through Deep Reinforcement Learning (DQN) | Paper (Mnih et al. 2015) | https://www.nature.com/articles/nature14236 |
| Playing Atari with Deep Reinforcement Learning | Paper (Mnih et al. 2013) | https://arxiv.org/abs/1312.5602 |

### Policy Gradient and PPO

| Resource | Type | Link |
|----------|------|------|
| Policy Gradient Methods for Reinforcement Learning | Paper (Sutton et al. 1999) | Original policy gradient paper |
| Proximal Policy Optimization Algorithms | Paper (Schulman et al. 2017) | https://arxiv.org/abs/1707.06347 — PPO original |
| High-Dimensional Continuous Control Using Generalized Advantage Estimation | Paper (Schulman et al. 2015) | https://arxiv.org/abs/1506.02438 — GAE paper |
| An Introduction to Deep Reinforcement Learning | Paper (FREE) | https://arxiv.org/abs/1811.12560 |

### SAC

| Resource | Type | Link |
|----------|------|------|
| Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL | Paper (Haarnoja et al. 2018) | https://arxiv.org/abs/1801.01290 |
| Soft Actor-Critic Algorithms and Applications | Paper (Haarnoja et al. 2019) | https://arxiv.org/abs/1812.05905 |

### MCTS

| Resource | Type | Link |
|----------|------|------|
| Mastering the Game of Go with Deep Neural Networks (AlphaGo) | Paper (Silver et al. 2016) | https://www.nature.com/articles/nature16961 |
| Mastering Chess and Shogi by Self-Play (AlphaZero) | Paper (Silver et al. 2017) | https://arxiv.org/abs/1712.01815 |
| A Survey of Monte Carlo Tree Search Methods | Paper | https://ieeexplore.ieee.org/document/6145622 |

---

## Self-Supervised Learning

| Resource | Type | Link |
|----------|------|------|
| A Simple Framework for Contrastive Learning (SimCLR) | Paper (Chen et al. 2020) | https://arxiv.org/abs/2002.05709 |
| Masked Autoencoders Are Scalable Vision Learners (MAE) | Paper (He et al. 2021) | https://arxiv.org/abs/2111.06377 |
| Bootstrap Your Own Latent (BYOL) | Paper (Grill et al. 2020) | https://arxiv.org/abs/2006.07733 |
| Self-Supervised Representation Learning | Blog (FREE) | https://lilianweng.github.io/posts/2019-11-10-self-supervised |
| The Illustrated Self-Supervised Learning | Blog (FREE) | https://amitness.com/2020/02/illustrated-self-supervised-learning |

---

## Reference Blogs — Bookmark These

These blogs publish deep technical content consistently. Treat them as living textbooks.

| Blog | Author | Best For |
|------|--------|----------|
| https://lilianweng.github.io | Lilian Weng (OpenAI) | RL, GANs, VAEs, Diffusion, Attention — comprehensive deep dives |
| https://colah.github.io | Chris Olah | LSTMs, Attention, Neural Net visualization — best intuition writing |
| https://jalammar.github.io | Jay Alammar | Transformers, BERT, GPT — illustrated explanations |
| https://karpathy.github.io | Andrej Karpathy | RNNs, Training tricks, practical deep learning |
| https://distill.pub | Distill | Research-grade interactive explanations — momentum, attention, features |
| https://ruder.io | Sebastian Ruder | NLP, optimization, transfer learning |
| https://www.offconvex.org | Off the Convex Path | Optimization theory, deep learning theory |
| https://ermongroup.github.io/cs228-notes | Stanford PGM | Probabilistic graphical models complete notes |

---

## YouTube Channels — For Visual Learners

| Channel | Best For |
|---------|----------|
| Andrej Karpathy | Building everything from scratch — the best channel for this project |
| StatQuest with Josh Starmer | Classical ML with clear visual intuitions |
| 3Blue1Brown | Linear algebra, calculus, neural networks — mathematical intuition |
| Yannic Kilcher | Paper walkthroughs — transformers, RL, diffusion |
| Two Minute Papers | Paper summaries — staying current |
| Lex Fridman | Long-form interviews — context and perspective |
| Deepmind | RL, AlphaFold, research papers explained |

---

## Courses — Structured Learning Paths

| Course | Platform | Cost | Best For |
|--------|----------|------|----------|
| CS229 Machine Learning (Andrew Ng) | Stanford / YouTube | FREE | Classical ML theory — lectures + notes |
| CS231n CNNs for Visual Recognition | Stanford / YouTube | FREE | CNN theory and implementation |
| CS224N NLP with Deep Learning | Stanford / YouTube | FREE | Transformers, BERT, GPT in depth |
| CS285 Deep Reinforcement Learning | Berkeley / YouTube | FREE | PPO, SAC, MCTS, model-based RL |
| Fast.ai Practical Deep Learning | fast.ai | FREE | Top-down practical implementation |
| Deep Learning Specialization | Coursera (Ng) | Paid / audit free | Structured path through deep learning |
| Full Stack Deep Learning | https://fullstackdeeplearning.com | FREE | MLOps, deployment, production ML |

---

## Mathematics Prerequisites

If any mathematical concept blocks you while implementing, these are the targeted resources.

### Linear Algebra

| Resource | Type | Link |
|----------|------|------|
| Linear Algebra — Gilbert Strang | Book + YouTube (FREE) | https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010 |
| Essence of Linear Algebra — 3Blue1Brown | YouTube (FREE) | https://youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab |
| MML Book Chapter 2 | Book (FREE) | https://mml-book.github.io |

### Calculus and Optimization

| Resource | Type | Link |
|----------|------|------|
| Essence of Calculus — 3Blue1Brown | YouTube (FREE) | https://youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr |
| Convex Optimization — Boyd & Vandenberghe | Book (FREE) | https://web.stanford.edu/~boyd/cvxbook |
| MML Book Chapter 5 | Book (FREE) | https://mml-book.github.io |

### Probability and Statistics

| Resource | Type | Link |
|----------|------|------|
| Probability Theory: The Logic of Science | Book | Jaynes — the Bayesian probability bible |
| Think Stats | Book (FREE) | https://greenteapress.com/thinkstats2 |
| CS229 Probability Review | Notes (FREE) | https://cs229.stanford.edu/section/cs229-prob.pdf |
| MML Book Chapters 6-9 | Book (FREE) | https://mml-book.github.io |

---

## Paper Reading Order

Read papers in this order — each one builds on the previous.

```
Week 1-6 (Classical):
  Tibshirani 1996 — Lasso
  Breiman 2001 — Random Forests
  Friedman 2001 — Gradient Boosting
  Freund & Schapire — AdaBoost
  Platt 1998 — SMO for SVM

Week 7-9 (Unsupervised):
  Pearson 1901 — PCA
  Ester et al. 1996 — DBSCAN
  Dempster et al. 1977 — EM Algorithm
  Liu et al. 2008 — Isolation Forest
  van der Maaten & Hinton 2008 — t-SNE
  McInnes et al. 2018 — UMAP

Week 10-14 (Probabilistic):
  Kalman 1960 — Kalman Filter
  Rabiner 1989 — HMM Tutorial
  Lafferty et al. 2001 — CRF
  Engle 1982 — ARCH
  Bollerslev 1986 — GARCH

Week 15-20 (Deep Layers):
  Rumelhart et al. 1986 — Backpropagation
  Hochreiter & Schmidhuber 1997 — LSTM
  Ioffe & Szegedy 2015 — Batch Normalization
  Srivastava et al. 2014 — Dropout
  LeCun et al. 1998 — CNN

Week 21-26 (Deep Models):
  Kingma & Welling 2013 — VAE
  Goodfellow et al. 2014 — GAN
  He et al. 2020 — DDPM Diffusion

Week 27-32 (NLP):
  Mikolov et al. 2013 — Word2Vec
  Vaswani et al. 2017 — Attention Is All You Need
  Devlin et al. 2018 — BERT
  Radford et al. 2019 — GPT-2

Week 33-38 (RL):
  Watkins & Dayan 1992 — Q-Learning
  Mnih et al. 2015 — DQN
  Schulman et al. 2015 — GAE
  Schulman et al. 2017 — PPO
  Haarnoja et al. 2018 — SAC
  Silver et al. 2017 — AlphaZero

Week 39-42 (Self-Supervised):
  Chen et al. 2020 — SimCLR
  He et al. 2021 — MAE
```

---

## How To Read a Paper

Most people read papers wrong. Use this method:

**Pass 1 (10 minutes):** Read title, abstract, introduction, conclusion, and look at all figures. Decide if the paper is worth reading fully.

**Pass 2 (1 hour):** Read everything except the math. Understand what problem they're solving, what their key idea is, and what results they show. Ignore proofs.

**Pass 3 (2-4 hours):** Implement the algorithm described. Only now read the math carefully. Every equation should become a line of code.

A paper you've implemented is worth ten papers you've only read.

---

## Notes

- Prioritize papers over secondary explanations wherever possible. Blogs and videos build intuition but the paper is the ground truth.
- For every algorithm, the implementation should come before or alongside the paper reading — not after.
- Lilian Weng's blog covers almost every algorithm in this library. When stuck on theory, check her post on that topic first.
- Karpathy's YouTube series is the single best resource for building the autograd engine and transformer from scratch. Watch all of it before Phase 4.

---

## Verification Strategy

For every algorithm implemented, verify against a known reference before moving on.

**Classical models** — generate a toy dataset, run your implementation and sklearn side by side,
assert predictions match within 1e-4.

**Gradient-based models** — numerical gradient check: perturb each weight by epsilon=1e-5,
compute (loss_plus - loss_minus) / (2 * epsilon), compare to analytical gradient. Should match
within 1e-5.

**Probabilistic models** — generate synthetic data from a known distribution with known parameters,
fit your model, assert recovered parameters are close to ground truth.

**RL agents** — run on CartPole-v1 (simple custom implementation), assert mean episode reward
exceeds 195 over 100 episodes within 500 training episodes.

---

## Notes

- Never use sklearn, PyTorch, or TensorFlow inside any implementation file. Utils and tests may
  use them for verification only.
- Every file should be self-contained and importable independently.
- Comments explain the math, not the code. A line of code that implements a formula should cite
  the formula, not describe what the line does syntactically.
- Numerical stability matters. Use log-sum-exp tricks for softmax and cross-entropy. Use
  eigendecomposition via `np.linalg.eigh` not `eig` for symmetric matrices. Clip values before
  taking logs.

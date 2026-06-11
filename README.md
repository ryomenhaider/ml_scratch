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

## Foundational Books — Read Before Phase 1

| Book | Author | What It Covers |
|------|--------|----------------|
| Mathematics for Machine Learning | Deisenroth, Faisal, Ong | Linear algebra, calculus, probability, optimization, PCA, GMM — free at mml-book.github.io |
| Machine Learning: An Algorithmic Perspective | Marsland | Every classical algorithm derived from scratch with Python code — Phases 1–3 |
| Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow | Géron | Practical implementation, full ML pipeline, deep learning intro — Phases 1–5 |
| Deep Learning | Goodfellow, Bengio, Courville | Full theory of deep learning, backprop, CNNs, RNNs, optimization — free at deeplearningbook.org — Phases 4–8 |

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

---

### Phase 0 — `core/` — Foundation Everything Else Depends On

**Build order:** `autograd.py` → `tensor.py` → `activations.py` → `losses.py` → `optimizers.py` → `initializers.py` → `metrics.py`

Everything else in the library depends on this layer. Do not move to any other module until
the autograd engine passes numerical gradient checking.

| File | What It Does |
|------|-------------|
| `autograd.py` | Builds a computation graph during the forward pass and runs backpropagation through it to compute gradients |
| `tensor.py` | NumPy array wrapper that tracks every operation so gradients can flow backward automatically |
| `activations.py` | Nonlinear functions applied after each layer — relu, sigmoid, tanh, gelu, softmax — plus their derivatives |
| `losses.py` | Measures how wrong the model's predictions are — mse, cross-entropy, kl divergence, huber, hinge |
| `optimizers.py` | Updates weights using gradients to minimize loss — SGD with momentum, Adam, RMSProp, AdaGrad |
| `initializers.py` | Sets starting weight values using Glorot, He, or orthogonal schemes so training does not die immediately |
| `metrics.py` | Measures model quality after training — accuracy, f1, rmse, r2, roc-auc, sharpe ratio, VaR |

#### Learning — Core / Autograd

| Title | Type | Link |
|-------|------|------|
| Automatic Differentiation in Machine Learning: a Survey | Paper | https://arxiv.org/abs/1502.05767 |
| Backpropagation Applied to Handwritten Zip Code Recognition | Paper (LeCun 1989) | http://yann.lecun.com/exdb/publis/pdf/lecun-89e.pdf |
| Calculus on Computational Graphs: Backpropagation | Blog (Colah) | https://colah.github.io/posts/2015-08-Backprop |
| Adam: A Method for Stochastic Optimization | Paper (Kingma & Ba 2014) | https://arxiv.org/abs/1412.6980 |
| An Overview of Gradient Descent Optimization Algorithms | Blog (Ruder) | https://www.ruder.io/optimizing-gradient-descent |
| Why Momentum Really Works | Blog (Distill) | https://distill.pub/2017/momentum |

---

### Phase 1 — `classical/` — Supervised Learning

**Project mapping:** P3 (End-to-End ML System) — implement Decision Tree, Random Forest, Gradient Boosting before building P3.

#### Linear Models

| File | What It Does |
|------|-------------|
| `linear_regression.py` | Fits a line through data to predict continuous values — normal equation for small data, gradient descent for large |
| `logistic_regression.py` | Predicts probability of a binary outcome by passing a linear combination through sigmoid |
| `ridge.py` | Linear regression with L2 penalty on weights — closed form via `(X^T X + λI)^-1 X^T y` |
| `lasso.py` | Linear regression with L1 penalty — solved via coordinate descent, drives irrelevant weights to exactly zero |

##### Learning — Linear and Logistic Regression

| Title | Type | Link |
|-------|------|------|
| CS229 Lecture Notes — Linear and Logistic Regression | Notes (Stanford FREE) | https://cs229.stanford.edu/notes2022fall/main_notes.pdf |
| Regression Shrinkage and Selection via the Lasso | Paper (Tibshirani 1996) | https://www.jstor.org/stable/2346178 |
| Ridge Regression: Biased Estimation for Nonorthogonal Problems | Paper (Hoerl & Kennard 1970) | Search via Google Scholar |
| The Elements of Statistical Learning — Chapters 3–4 | Book (FREE) | https://hastie.su.domains/ElemStatLearn |

#### Tree Models

| File | What It Does |
|------|-------------|
| `decision_tree.py` | Recursively splits data on the feature and threshold that maximizes information gain until leaves are pure |
| `random_forest.py` | Trains 100+ decision trees on bootstrap samples with random feature subsets, aggregates by majority vote |
| `gradient_boosting.py` | Builds trees sequentially where each new tree fits the negative gradient (residuals) of the previous ensemble |

##### Learning — Decision Trees and Ensembles

| Title | Type | Link |
|-------|------|------|
| Random Forests | Paper (Breiman 2001) | https://link.springer.com/article/10.1023/A:1010933404324 |
| Greedy Function Approximation: A Gradient Boosting Machine | Paper (Friedman 2001) | https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-function-approximation-a-gradient-boosting-machine/10.1214/aos/1013203451.full |
| XGBoost: A Scalable Tree Boosting System | Paper (Chen & Guestrin 2016) | https://arxiv.org/abs/1603.02754 |
| A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting (AdaBoost) | Paper (Freund & Schapire 1997) | https://www.sciencedirect.com/science/article/pii/S002200009791504X |

#### Other Classical

| File | What It Does |
|------|-------------|
| `knn.py` | Classifies a point by majority vote of its k nearest neighbors — lazy learner, no training step |
| `naive_bayes.py` | Classifies using Bayes theorem assuming all features are conditionally independent given the class |
| `svm.py` | Finds the maximum-margin hyperplane between classes — solved via SMO algorithm, supports rbf and polynomial kernels |

##### Learning — SVM

| Title | Type | Link |
|-------|------|------|
| A Tutorial on Support Vector Machines for Pattern Recognition | Paper (Burges 1998) | https://link.springer.com/article/10.1023/A:1022627411411 |
| Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines | Paper (Platt 1998) | https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-98-14.pdf |
| CS229 Lecture Notes — Support Vector Machines | Notes (Stanford FREE) | https://cs229.stanford.edu/notes2022fall/main_notes.pdf |

##### Learning — KNN and Naive Bayes

| Title | Type | Link |
|-------|------|------|
| Nearest Neighbor Pattern Classification | Paper (Cover & Hart 1967) | IEEE Transactions — search via Google Scholar |
| CS229 Lecture Notes — Generative Learning Algorithms | Notes (Stanford FREE) | https://cs229.stanford.edu/notes2022fall/main_notes.pdf |

---

### Phase 2 — `unsupervised/` — Learning Without Labels

**Project mapping:** P6 (Streaming + Real-Time ML) uses Isolation Forest. P8 (Signal Detection) uses PCA and GMM.

#### Clustering

| File | What It Does |
|------|-------------|
| `kmeans.py` | Groups n points into k clusters by iteratively assigning points to nearest centroid and recomputing centroids |
| `dbscan.py` | Groups points by density reachability — no need to specify k, outliers automatically become noise |
| `gmm.py` | Fits k Gaussian distributions to data via EM algorithm, allowing soft probabilistic cluster membership |

##### Learning — Clustering

| Title | Type | Link |
|-------|------|------|
| A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise (DBSCAN) | Paper (Ester et al. 1996) | https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf |
| Maximum Likelihood from Incomplete Data via the EM Algorithm | Paper (Dempster, Laird, Rubin 1977) | https://www.jstor.org/stable/2984875 |
| From EM to Gaussian Mixture Models | Blog (Towards Data Science) | https://towardsdatascience.com/gaussian-mixture-models-explained-6986aaf5a95 |

#### Dimensionality Reduction

| File | What It Does |
|------|-------------|
| `pca.py` | Rotates data to the axes of maximum variance via eigendecomposition then drops the weakest components |
| `umap.py` | Compresses high-dimensional data to 2-3 dimensions by preserving the topological structure of the manifold |

##### Learning — Dimensionality Reduction

| Title | Type | Link |
|-------|------|------|
| A Tutorial on Principal Component Analysis | Paper (Shlens 2014) | https://arxiv.org/abs/1404.1100 |
| Visualizing Data using t-SNE | Paper (van der Maaten & Hinton 2008) | https://jmlr.org/papers/v9/vandermaaten08a.html |
| UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction | Paper (McInnes et al. 2018) | https://arxiv.org/abs/1802.03426 |
| Understanding UMAP | Blog (Pair Code) | https://pair-code.github.io/understanding-umap |

#### Anomaly Detection

| File | What It Does |
|------|-------------|
| `isolation_forest.py` | Flags anomalies as points that get isolated fastest when random splits are applied — anomalies are few and different |

##### Learning — Anomaly Detection

| Title | Type | Link |
|-------|------|------|
| Isolation Forest | Paper (Liu, Ting, Zhou 2008) | https://ieeexplore.ieee.org/document/4781136 |
| Anomaly Detection: A Survey | Paper (Chandola et al. 2009) | https://dl.acm.org/doi/10.1145/1541880.1541882 |

---

### Phase 3 — `probabilistic/` — Uncertainty, Sequences, and Time

**Project mapping:** P8 (Signal Detection) uses HMM, Kalman Filter, Gaussian Process, ARIMA, GARCH. P14 (Risk Management) uses Monte Carlo and Kalman Filter. P15 (Multi-Agent) uses Bayesian Network.

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

#### Learning — Hidden Markov Models

| Title | Type | Link |
|-------|------|------|
| A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition | Paper (Rabiner 1989) | https://ieeexplore.ieee.org/document/18626 |
| Speech and Language Processing — Chapter 8 | Book (Jurafsky & Martin FREE) | https://web.stanford.edu/~jurafsky/slp3 |

#### Learning — Kalman Filter

| Title | Type | Link |
|-------|------|------|
| A New Approach to Linear Filtering and Prediction Problems | Paper (Kalman 1960) | https://asmedigitalcollection.asme.org/fluidsengineering/article/82/1/35/397706 |
| How a Kalman Filter Works, in Pictures | Blog | https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures |
| Kalman and Bayesian Filters in Python | Free Online Book | https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python |

#### Learning — Gaussian Processes

| Title | Type | Link |
|-------|------|------|
| Gaussian Processes for Machine Learning | Book (Rasmussen & Williams FREE) | http://www.gaussianprocess.org/gpml |
| A Visual Exploration of Gaussian Processes | Blog (Distill) | https://distill.pub/2019/visual-exploration-gaussian-processes |

#### Learning — Bayesian Networks

| Title | Type | Link |
|-------|------|------|
| An Introduction to Bayesian Networks | Paper (Jensen 1996) | Search via Google Scholar |
| Probabilistic Graphical Models — CS228 Notes | Notes (Stanford FREE) | https://ermongroup.github.io/cs228-notes |
| Bayesian Reasoning and Machine Learning — Chapters 1–5 | Book (Barber FREE) | http://www.cs.ucl.ac.uk/staff/d.barber/brml |

#### Learning — CRF

| Title | Type | Link |
|-------|------|------|
| Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data | Paper (Lafferty et al. 2001) | Search via Google Scholar |
| An Introduction to Conditional Random Fields | Paper (Sutton & McCallum 2010) | https://arxiv.org/abs/1011.4088 |

#### Learning — ARIMA and GARCH

| Title | Type | Link |
|-------|------|------|
| Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation | Paper (Engle 1982) | Journal of Econometrica — search via Google Scholar |
| Generalized Autoregressive Conditional Heteroscedasticity | Paper (Bollerslev 1986) | Journal of Econometrics — search via Google Scholar |
| Forecasting: Principles and Practice — Chapters 8–9 | Book (Hyndman & Athanasopoulos FREE) | https://otexts.com/fpp3 |

#### Learning — Monte Carlo

| Title | Type | Link |
|-------|------|------|
| An Introduction to MCMC for Machine Learning | Paper (Andrieu et al. 2003) | https://link.springer.com/article/10.1023/A:1020281327116 |
| CS228 Probabilistic Graphical Models — Sampling Notes | Notes (Stanford FREE) | https://ermongroup.github.io/cs228-notes |

---

### Phase 4 — `deep/layers/` — Neural Network Primitives

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

#### Learning — Backpropagation and MLP

| Title | Type | Link |
|-------|------|------|
| Learning Representations by Backpropagating Errors | Paper (Rumelhart, Hinton, Williams 1986) | Search via Google Scholar |
| Neural Networks and Deep Learning | Free Online Book (Nielsen) | http://neuralnetworksanddeeplearning.com |

#### Learning — CNN

| Title | Type | Link |
|-------|------|------|
| Gradient-Based Learning Applied to Document Recognition | Paper (LeCun et al. 1998) | http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf |
| ImageNet Classification with Deep Convolutional Neural Networks (AlexNet) | Paper (Krizhevsky et al. 2012) | https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html |
| Deep Residual Learning for Image Recognition (ResNet) | Paper (He et al. 2015) | https://arxiv.org/abs/1512.03385 |
| CS231n — Convolutional Neural Networks for Visual Recognition | Notes (Stanford FREE) | https://cs231n.github.io |

#### Learning — RNN, LSTM, GRU

| Title | Type | Link |
|-------|------|------|
| Learning Long-Term Dependencies with Gradient Descent is Difficult | Paper (Bengio et al. 1994) | Search via Google Scholar |
| Long Short-Term Memory | Paper (Hochreiter & Schmidhuber 1997) | Search via Google Scholar |
| Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling | Paper (Chung et al. 2014) | https://arxiv.org/abs/1412.3555 |
| Understanding LSTM Networks | Blog (Colah) | https://colah.github.io/posts/2015-08-Understanding-LSTMs |

#### Learning — Normalization and Regularization

| Title | Type | Link |
|-------|------|------|
| Batch Normalization: Accelerating Deep Network Training | Paper (Ioffe & Szegedy 2015) | https://arxiv.org/abs/1502.03167 |
| Layer Normalization | Paper (Ba et al. 2016) | https://arxiv.org/abs/1607.06450 |
| Dropout: A Simple Way to Prevent Neural Networks from Overfitting | Paper (Srivastava et al. 2014) | https://jmlr.org/papers/v15/srivastava14a.html |

---

### Phase 5 — `deep/models/` — Full Architectures

**Project mapping:** P4 (ML Inference Service) uses MLP/CNN. P5 (RAG System) uses the Transformer. P11 (Multi-Modal) uses CNN + Transformer + VAE.

| File | What It Does |
|------|-------------|
| `mlp.py` | Stack of dense layers — the universal function approximator, baseline for all tabular deep learning |
| `cnn.py` | Alternating conv and pooling layers that extract hierarchical spatial features from images |
| `rnn.py` | Stacked recurrent layers for modeling temporal sequences — baseline before LSTM |
| `autoencoder.py` | Encoder compresses input to a bottleneck, decoder reconstructs it — learns compact representations |
| `vae.py` | Autoencoder that learns a smooth, continuous latent space via reparameterization and KL divergence loss |
| `gan.py` | Generator and discriminator trained adversarially — generator fools the discriminator, discriminator catches the generator |
| `diffusion.py` | Learns to reverse a noise-adding Markov chain — generates data by iteratively denoising from Gaussian noise |

#### Learning — Autoencoder, VAE, GAN

| Title | Type | Link |
|-------|------|------|
| Auto-Encoding Variational Bayes | Paper (Kingma & Welling 2013) | https://arxiv.org/abs/1312.6114 |
| Generative Adversarial Networks | Paper (Goodfellow et al. 2014) | https://arxiv.org/abs/1406.2661 |
| Tutorial on Variational Autoencoders | Paper (Doersch 2016) | https://arxiv.org/abs/1606.05908 |
| From Autoencoder to Beta-VAE | Blog (Lilian Weng) | https://lilianweng.github.io/posts/2018-08-12-vae |
| From GAN to WGAN | Blog (Lilian Weng) | https://lilianweng.github.io/posts/2017-08-20-gan |

#### Learning — Diffusion Models

| Title | Type | Link |
|-------|------|------|
| Denoising Diffusion Probabilistic Models | Paper (Ho et al. 2020) | https://arxiv.org/abs/2006.11239 |
| Improved Denoising Diffusion Probabilistic Models | Paper (Nichol & Dhariwal 2021) | https://arxiv.org/abs/2102.09672 |
| What are Diffusion Models? | Blog (Lilian Weng) | https://lilianweng.github.io/posts/2021-07-11-diffusion-models |

---

### Phase 6 — `nlp/` — Language and Sequence Understanding

**Project mapping:** P5 (RAG System) — implement attention and transformer before building the retrieval pipeline. Understanding these from scratch lets you debug retrieval quality at the embedding level.

| File | What It Does |
|------|-------------|
| `word2vec.py` | Trains word embeddings via skip-gram or CBOW so semantically similar words are close in vector space |
| `transformer.py` | Full encoder-decoder architecture built entirely on multi-head attention — no recurrence, fully parallelizable |
| `bert.py` | Transformer encoder pretrained by masking 15% of tokens and predicting them — bidirectional context |
| `gpt.py` | Transformer decoder pretrained by predicting the next token autoregressively — causal, left-to-right only |

#### Learning — Word Embeddings

| Title | Type | Link |
|-------|------|------|
| Efficient Estimation of Word Representations in Vector Space (Word2Vec) | Paper (Mikolov et al. 2013) | https://arxiv.org/abs/1301.3781 |
| GloVe: Global Vectors for Word Representation | Paper (Pennington et al. 2014) | https://aclanthology.org/D14-1162 |
| The Illustrated Word2Vec | Blog (Jay Alammar) | https://jalammar.github.io/illustrated-word2vec |

#### Learning — Transformer

| Title | Type | Link |
|-------|------|------|
| Attention Is All You Need | Paper (Vaswani et al. 2017) | https://arxiv.org/abs/1706.03762 |
| The Illustrated Transformer | Blog (Jay Alammar) | https://jalammar.github.io/illustrated-transformer |
| The Annotated Transformer | Blog + Code (Harvard NLP) | https://nlp.seas.harvard.edu/annotated-transformer |

#### Learning — BERT and GPT

| Title | Type | Link |
|-------|------|------|
| BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding | Paper (Devlin et al. 2018) | https://arxiv.org/abs/1810.04805 |
| Language Models are Unsupervised Multitask Learners (GPT-2) | Paper (Radford et al. 2019) | https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf |
| The Illustrated BERT, ELMo, and co. | Blog (Jay Alammar) | https://jalammar.github.io/illustrated-bert |
| The Illustrated GPT-2 | Blog (Jay Alammar) | https://jalammar.github.io/illustrated-gpt2 |

---

### Phase 7 — `rl/` — Reinforcement Learning

**Project mapping:** P15 (Autonomous Multi-Agent) uses PPO or SAC for agent policy learning and MCTS for planning.

#### Agents

| File | What It Does |
|------|-------------|
| `q_learning.py` | Tabular agent that learns action-value function via TD updates — works only for small discrete state spaces |
| `dqn.py` | Q-learning with a neural network function approximator, experience replay buffer, and target network for stability |
| `ppo.py` | Policy gradient method that clips the surrogate objective to prevent catastrophically large updates |
| `sac.py` | Off-policy actor-critic that maximizes reward plus an entropy bonus — naturally explores without epsilon-greedy |
| `mcts.py` | Plans by simulating future trajectories from the current state and backing up value estimates via tree search |

#### Memory

| File | What It Does |
|------|-------------|
| `replay_buffer.py` | Ring buffer storing (state, action, reward, next_state, done) tuples — sampled randomly during DQN training |

#### Learning — Foundational RL

| Title | Type | Link |
|-------|------|------|
| Reinforcement Learning: An Introduction | Book (Sutton & Barto FREE) | http://incompleteideas.net/book/the-book-2nd.html |
| An Introduction to Deep Reinforcement Learning | Paper (Francois-Lavet et al. 2018) | https://arxiv.org/abs/1811.12560 |

#### Learning — Q-Learning and DQN

| Title | Type | Link |
|-------|------|------|
| Q-Learning | Paper (Watkins & Dayan 1992) | Search via Google Scholar — Machine Learning journal |
| Playing Atari with Deep Reinforcement Learning | Paper (Mnih et al. 2013) | https://arxiv.org/abs/1312.5602 |
| Human-level Control Through Deep Reinforcement Learning (DQN) | Paper (Mnih et al. 2015) | https://www.nature.com/articles/nature14236 |

#### Learning — PPO and Policy Gradient

| Title | Type | Link |
|-------|------|------|
| Policy Gradient Methods for Reinforcement Learning with Function Approximation | Paper (Sutton et al. 1999) | Search via Google Scholar |
| High-Dimensional Continuous Control Using Generalized Advantage Estimation | Paper (Schulman et al. 2015) | https://arxiv.org/abs/1506.02438 |
| Proximal Policy Optimization Algorithms | Paper (Schulman et al. 2017) | https://arxiv.org/abs/1707.06347 |
| Proximal Policy Optimization | Blog (OpenAI) | https://openai.com/research/openai-baselines-ppo |

#### Learning — SAC

| Title | Type | Link |
|-------|------|------|
| Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor | Paper (Haarnoja et al. 2018) | https://arxiv.org/abs/1801.01290 |
| Soft Actor-Critic Algorithms and Applications | Paper (Haarnoja et al. 2019) | https://arxiv.org/abs/1812.05905 |

#### Learning — MCTS

| Title | Type | Link |
|-------|------|------|
| Mastering the Game of Go with Deep Neural Networks and Tree Search (AlphaGo) | Paper (Silver et al. 2016) | https://www.nature.com/articles/nature16961 |
| Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm (AlphaZero) | Paper (Silver et al. 2017) | https://arxiv.org/abs/1712.01815 |
| A Survey of Monte Carlo Tree Search Methods | Paper (Browne et al. 2012) | https://ieeexplore.ieee.org/document/6145622 |

---

### Phase 8 — `self_supervised/` — Learning Without Labels at Scale

**Project mapping:** P11 (Multi-Modal Intelligence) — self-supervised pretraining on unlabeled imagery before fine-tuning.

| File | What It Does |
|------|-------------|
| `simclr.py` | Trains an encoder by pulling two augmented views of the same image together in latent space and pushing others apart |
| `mae.py` | Trains an encoder by masking 75% of input patches and learning to reconstruct the missing ones |

#### Learning — Self-Supervised

| Title | Type | Link |
|-------|------|------|
| A Simple Framework for Contrastive Learning of Visual Representations (SimCLR) | Paper (Chen et al. 2020) | https://arxiv.org/abs/2002.05709 |
| Masked Autoencoders Are Scalable Vision Learners (MAE) | Paper (He et al. 2021) | https://arxiv.org/abs/2111.06377 |
| Self-Supervised Representation Learning | Blog (Lilian Weng) | https://lilianweng.github.io/posts/2019-11-10-self-supervised |

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

## Reference Blogs — Bookmark These

| Blog | Author | Best For |
|------|--------|----------|
| https://lilianweng.github.io | Lilian Weng | RL, GANs, VAEs, Diffusion, Attention — most comprehensive technical blog in ML |
| https://colah.github.io | Chris Olah | LSTMs, Attention, Neural Net internals — best intuition writing anywhere |
| https://jalammar.github.io | Jay Alammar | Transformers, BERT, GPT — illustrated walkthroughs |
| https://karpathy.github.io | Andrej Karpathy | RNNs, training tricks, practical deep learning |
| https://distill.pub | Distill | Research-grade interactive explanations — momentum, attention, features |
| https://ruder.io | Sebastian Ruder | NLP, optimization, transfer learning |
| https://ermongroup.github.io/cs228-notes | Stanford CS228 | Probabilistic graphical models — complete course notes |

---

## Videos — Lowest Priority

Only watch if reading the paper and blog did not build sufficient intuition.

| Title | Channel | Link |
|-------|---------|------|
| The spelled-out intro to neural networks and backpropagation | Andrej Karpathy | https://youtu.be/VMj-3S1tku0 |
| Let's build GPT from scratch | Andrej Karpathy | https://youtu.be/kCc8FmEb1nY |
| Essence of Linear Algebra | 3Blue1Brown | https://youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab |
| Essence of Calculus | 3Blue1Brown | https://youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr |
| Neural Networks (full series) | 3Blue1Brown | https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi |
| David Silver RL Lecture Series | DeepMind | https://youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ |
| CS231n Lecture Series | Stanford | https://youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv |

---

## How To Read a Paper

**Pass 1 (10 minutes):** Read title, abstract, introduction, conclusion, and look at all figures. Decide if the paper is worth a full read.

**Pass 2 (1 hour):** Read everything except proofs. Understand the problem, the key idea, and the results. Skip the math.

**Pass 3 (2-4 hours):** Implement the algorithm. Only now read the math carefully. Every equation becomes a line of code.

A paper you have implemented is worth ten papers you have only read.

---

## Notes

- Never use sklearn, PyTorch, or TensorFlow inside any implementation file. Utils and tests may use them for verification only.
- Every file should be self-contained and importable independently.
- Comments explain the math, not the code. A line of code that implements a formula should cite the formula, not describe what the line does syntactically.
- Numerical stability matters. Use log-sum-exp tricks for softmax and cross-entropy. Use eigendecomposition via `np.linalg.eigh` not `eig` for symmetric matrices. Clip values before taking logs.
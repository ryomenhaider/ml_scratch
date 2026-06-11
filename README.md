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

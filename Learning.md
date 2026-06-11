# ML From Scratch — Learning Resources

Books only for foundations. Everything else is papers and blogs. Videos listed last as lowest priority.

---

## Foundational Books — Read These First

| Book | Author | What It Covers | When To Read |
|------|--------|---------------|--------------|
| Mathematics for Machine Learning | Deisenroth, Faisal, Ong | Linear algebra, calculus, probability, optimization, PCA, GMM | Before Phase 1 — free at mml-book.github.io |
| Machine Learning: An Algorithmic Perspective | Marsland | Every classical algorithm derived from scratch with Python code | Phase 1–3 |
| Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow | Géron | Practical implementation, full ML pipeline, deep learning intro | Phase 1–5 |
| Deep Learning | Goodfellow, Bengio, Courville | Full theory of deep learning, backprop, CNNs, RNNs, optimization | Phase 4–8 — free at deeplearningbook.org |

---

## Core — Autograd and Tensor

| Title | Type | Link |
|-------|------|------|
| Automatic Differentiation in Machine Learning: a Survey | Paper | https://arxiv.org/abs/1502.05767 |
| Backpropagation Applied to Handwritten Zip Code Recognition | Paper (LeCun 1989) | http://yann.lecun.com/exdb/publis/pdf/lecun-89e.pdf |
| Calculus on Computational Graphs: Backpropagation | Blog (Colah) | https://colah.github.io/posts/2015-08-Backprop |
| Adam: A Method for Stochastic Optimization | Paper (Kingma & Ba 2014) | https://arxiv.org/abs/1412.6980 |
| An Overview of Gradient Descent Optimization Algorithms | Blog (Ruder) | https://www.ruder.io/optimizing-gradient-descent |
| Why Momentum Really Works | Blog (Distill) | https://distill.pub/2017/momentum |

---

## Classical ML

### Linear and Logistic Regression

| Title | Type | Link |
|-------|------|------|
| CS229 Lecture Notes — Linear and Logistic Regression | Notes (Stanford FREE) | https://cs229.stanford.edu/notes2022fall/main_notes.pdf |
| Regression Shrinkage and Selection via the Lasso | Paper (Tibshirani 1996) | https://www.jstor.org/stable/2346178 |
| Ridge Regression: Biased Estimation for Nonorthogonal Problems | Paper (Hoerl & Kennard 1970) | Search via Google Scholar |
| The Elements of Statistical Learning — Chapters 3-4 | Book (FREE) | https://hastie.su.domains/ElemStatLearn |

### Decision Trees and Ensembles

| Title | Type | Link |
|-------|------|------|
| Random Forests | Paper (Breiman 2001) | https://link.springer.com/article/10.1023/A:1010933404324 |
| Greedy Function Approximation: A Gradient Boosting Machine | Paper (Friedman 2001) | https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-function-approximation-a-gradient-boosting-machine/10.1214/aos/1013203451.full |
| XGBoost: A Scalable Tree Boosting System | Paper (Chen & Guestrin 2016) | https://arxiv.org/abs/1603.02754 |
| A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting (AdaBoost) | Paper (Freund & Schapire 1997) | https://www.sciencedirect.com/science/article/pii/S002200009791504X |

### SVM

| Title | Type | Link |
|-------|------|------|
| A Tutorial on Support Vector Machines for Pattern Recognition | Paper (Burges 1998) | https://link.springer.com/article/10.1023/A:1022627411411 |
| Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines | Paper (Platt 1998) | https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-98-14.pdf |
| CS229 Lecture Notes — Support Vector Machines | Notes (Stanford FREE) | https://cs229.stanford.edu/notes2022fall/main_notes.pdf |

### KNN and Naive Bayes

| Title | Type | Link |
|-------|------|------|
| Nearest Neighbor Pattern Classification | Paper (Cover & Hart 1967) | IEEE Transactions — search via Google Scholar |
| CS229 Lecture Notes — Generative Learning Algorithms | Notes (Stanford FREE) | https://cs229.stanford.edu/notes2022fall/main_notes.pdf |

---

## Unsupervised Learning

### Clustering

| Title | Type | Link |
|-------|------|------|
| A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise (DBSCAN) | Paper (Ester et al. 1996) | https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf |
| Maximum Likelihood from Incomplete Data via the EM Algorithm | Paper (Dempster, Laird, Rubin 1977) | https://www.jstor.org/stable/2984875 |
| From EM to Gaussian Mixture Models | Blog (Towards Data Science) | https://towardsdatascience.com/gaussian-mixture-models-explained-6986aaf5a95 |

### Dimensionality Reduction

| Title | Type | Link |
|-------|------|------|
| A Tutorial on Principal Component Analysis | Paper (Shlens 2014) | https://arxiv.org/abs/1404.1100 |
| Visualizing Data using t-SNE | Paper (van der Maaten & Hinton 2008) | https://jmlr.org/papers/v9/vandermaaten08a.html |
| UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction | Paper (McInnes et al. 2018) | https://arxiv.org/abs/1802.03426 |
| Understanding UMAP | Blog (Pair Code) | https://pair-code.github.io/understanding-umap |

### Anomaly Detection

| Title | Type | Link |
|-------|------|------|
| Isolation Forest | Paper (Liu, Ting, Zhou 2008) | https://ieeexplore.ieee.org/document/4781136 |
| Anomaly Detection: A Survey | Paper (Chandola et al. 2009) | https://dl.acm.org/doi/10.1145/1541880.1541882 |

---

## Probabilistic Models

### Hidden Markov Models

| Title | Type | Link |
|-------|------|------|
| A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition | Paper (Rabiner 1989) | https://ieeexplore.ieee.org/document/18626 |
| Speech and Language Processing — Chapter 8 | Book (Jurafsky & Martin FREE) | https://web.stanford.edu/\~jurafsky/slp3 |

### Kalman Filter

| Title | Type | Link |
|-------|------|------|
| A New Approach to Linear Filtering and Prediction Problems | Paper (Kalman 1960) | https://asmedigitalcollection.asme.org/fluidsengineering/article/82/1/35/397706 |
| How a Kalman Filter Works, in Pictures | Blog | https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures |
| Kalman and Bayesian Filters in Python | Free Online Book | https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python |

### Gaussian Processes

| Title | Type | Link |
|-------|------|------|
| Gaussian Processes for Machine Learning | Book (Rasmussen & Williams FREE) | http://www.gaussianprocess.org/gpml |
| A Visual Exploration of Gaussian Processes | Blog (Distill) | https://distill.pub/2019/visual-exploration-gaussian-processes |

### Bayesian Networks

| Title | Type | Link |
|-------|------|------|
| An Introduction to Bayesian Networks | Paper (Jensen 1996) | Search via Google Scholar |
| Probabilistic Graphical Models — CS228 Notes | Notes (Stanford FREE) | https://ermongroup.github.io/cs228-notes |
| Bayesian Reasoning and Machine Learning — Chapters 1-5 | Book (Barber FREE) | http://www.cs.ucl.ac.uk/staff/d.barber/brml |

### CRF

| Title | Type | Link |
|-------|------|------|
| Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data | Paper (Lafferty et al. 2001) | Search via Google Scholar |
| An Introduction to Conditional Random Fields | Paper (Sutton & McCallum 2010) | https://arxiv.org/abs/1011.4088 |

### ARIMA and GARCH

| Title | Type | Link |
|-------|------|------|
| Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation | Paper (Engle 1982) | Journal of Econometrica — search via Google Scholar |
| Generalized Autoregressive Conditional Heteroscedasticity | Paper (Bollerslev 1986) | Journal of Econometrics — search via Google Scholar |
| Forecasting: Principles and Practice — Chapters 8-9 | Book (Hyndman & Athanasopoulos FREE) | https://otexts.com/fpp3 |

### Monte Carlo

| Title | Type | Link |
|-------|------|------|
| An Introduction to MCMC for Machine Learning | Paper (Andrieu et al. 2003) | https://link.springer.com/article/10.1023/A:1020281327116 |
| CS228 Probabilistic Graphical Models — Sampling Notes | Notes (Stanford FREE) | https://ermongroup.github.io/cs228-notes |

---

## Deep Learning

### Backpropagation and MLP

| Title | Type | Link |
|-------|------|------|
| Learning Representations by Backpropagating Errors | Paper (Rumelhart, Hinton, Williams 1986) | Search via Google Scholar |
| Neural Networks and Deep Learning | Free Online Book (Nielsen) | http://neuralnetworksanddeeplearning.com |

### CNN

| Title | Type | Link |
|-------|------|------|
| Gradient-Based Learning Applied to Document Recognition | Paper (LeCun et al. 1998) | http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf |
| ImageNet Classification with Deep Convolutional Neural Networks (AlexNet) | Paper (Krizhevsky et al. 2012) | https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html |
| Deep Residual Learning for Image Recognition (ResNet) | Paper (He et al. 2015) | https://arxiv.org/abs/1512.03385 |
| CS231n — Convolutional Neural Networks for Visual Recognition (Course Notes) | Notes (Stanford FREE) | https://cs231n.github.io |

### RNN, LSTM, GRU

| Title | Type | Link |
|-------|------|------|
| Learning Long-Term Dependencies with Gradient Descent is Difficult | Paper (Bengio et al. 1994) | Search via Google Scholar |
| Long Short-Term Memory | Paper (Hochreiter & Schmidhuber 1997) | Search via Google Scholar |
| Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling | Paper (Chung et al. 2014) | https://arxiv.org/abs/1412.3555 |
| Understanding LSTM Networks | Blog (Colah) | https://colah.github.io/posts/2015-08-Understanding-LSTMs |

### Normalization and Regularization

| Title | Type | Link |
|-------|------|------|
| Batch Normalization: Accelerating Deep Network Training | Paper (Ioffe & Szegedy 2015) | https://arxiv.org/abs/1502.03167 |
| Layer Normalization | Paper (Ba et al. 2016) | https://arxiv.org/abs/1607.06450 |
| Dropout: A Simple Way to Prevent Neural Networks from Overfitting | Paper (Srivastava et al. 2014) | https://jmlr.org/papers/v15/srivastava14a.html |

### Autoencoder, VAE, GAN

| Title | Type | Link |
|-------|------|------|
| Auto-Encoding Variational Bayes | Paper (Kingma & Welling 2013) | https://arxiv.org/abs/1312.6114 |
| Generative Adversarial Networks | Paper (Goodfellow et al. 2014) | https://arxiv.org/abs/1406.2661 |
| Tutorial on Variational Autoencoders | Paper (Doersch 2016) | https://arxiv.org/abs/1606.05908 |
| From Autoencoder to Beta-VAE | Blog (Lilian Weng) | https://lilianweng.github.io/posts/2018-08-12-vae |
| From GAN to WGAN | Blog (Lilian Weng) | https://lilianweng.github.io/posts/2017-08-20-gan |

### Diffusion Models

| Title | Type | Link |
|-------|------|------|
| Denoising Diffusion Probabilistic Models | Paper (Ho et al. 2020) | https://arxiv.org/abs/2006.11239 |
| Improved Denoising Diffusion Probabilistic Models | Paper (Nichol & Dhariwal 2021) | https://arxiv.org/abs/2102.09672 |
| What are Diffusion Models? | Blog (Lilian Weng) | https://lilianweng.github.io/posts/2021-07-11-diffusion-models |

---

## NLP

### Word Embeddings

| Title | Type | Link |
|-------|------|------|
| Efficient Estimation of Word Representations in Vector Space (Word2Vec) | Paper (Mikolov et al. 2013) | https://arxiv.org/abs/1301.3781 |
| GloVe: Global Vectors for Word Representation | Paper (Pennington et al. 2014) | https://aclanthology.org/D14-1162 |
| The Illustrated Word2Vec | Blog (Jay Alammar) | https://jalammar.github.io/illustrated-word2vec |

### Transformer

| Title | Type | Link |
|-------|------|------|
| Attention Is All You Need | Paper (Vaswani et al. 2017) | https://arxiv.org/abs/1706.03762 |
| The Illustrated Transformer | Blog (Jay Alammar) | https://jalammar.github.io/illustrated-transformer |
| The Annotated Transformer | Blog + Code (Harvard NLP) | https://nlp.seas.harvard.edu/annotated-transformer |

### BERT and GPT

| Title | Type | Link |
|-------|------|------|
| BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding | Paper (Devlin et al. 2018) | https://arxiv.org/abs/1810.04805 |
| Language Models are Unsupervised Multitask Learners (GPT-2) | Paper (Radford et al. 2019) | https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf |
| The Illustrated BERT, ELMo, and co. | Blog (Jay Alammar) | https://jalammar.github.io/illustrated-bert |
| The Illustrated GPT-2 | Blog (Jay Alammar) | https://jalammar.github.io/illustrated-gpt2 |

---

## Reinforcement Learning

### Foundational

| Title | Type | Link |
|-------|------|------|
| Reinforcement Learning: An Introduction | Book (Sutton & Barto FREE) | http://incompleteideas.net/book/the-book-2nd.html |
| An Introduction to Deep Reinforcement Learning | Paper (Francois-Lavet et al. 2018) | https://arxiv.org/abs/1811.12560 |

### Q-Learning and DQN

| Title | Type | Link |
|-------|------|------|
| Q-Learning | Paper (Watkins & Dayan 1992) | Search via Google Scholar — Machine Learning journal |
| Playing Atari with Deep Reinforcement Learning | Paper (Mnih et al. 2013) | https://arxiv.org/abs/1312.5602 |
| Human-level Control Through Deep Reinforcement Learning (DQN) | Paper (Mnih et al. 2015) | https://www.nature.com/articles/nature14236 |

### PPO and Policy Gradient

| Title | Type | Link |
|-------|------|------|
| Policy Gradient Methods for Reinforcement Learning with Function Approximation | Paper (Sutton et al. 1999) | Search via Google Scholar |
| High-Dimensional Continuous Control Using Generalized Advantage Estimation | Paper (Schulman et al. 2015) | https://arxiv.org/abs/1506.02438 |
| Proximal Policy Optimization Algorithms | Paper (Schulman et al. 2017) | https://arxiv.org/abs/1707.06347 |
| Proximal Policy Optimization | Blog (OpenAI) | https://openai.com/research/openai-baselines-ppo |

### SAC

| Title | Type | Link |
|-------|------|------|
| Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor | Paper (Haarnoja et al. 2018) | https://arxiv.org/abs/1801.01290 |
| Soft Actor-Critic Algorithms and Applications | Paper (Haarnoja et al. 2019) | https://arxiv.org/abs/1812.05905 |

### MCTS

| Title | Type | Link |
|-------|------|------|
| Mastering the Game of Go with Deep Neural Networks and Tree Search (AlphaGo) | Paper (Silver et al. 2016) | https://www.nature.com/articles/nature16961 |
| Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm (AlphaZero) | Paper (Silver et al. 2017) | https://arxiv.org/abs/1712.01815 |
| A Survey of Monte Carlo Tree Search Methods | Paper (Browne et al. 2012) | https://ieeexplore.ieee.org/document/6145622 |

---

## Self-Supervised Learning

| Title | Type | Link |
|-------|------|------|
| A Simple Framework for Contrastive Learning of Visual Representations (SimCLR) | Paper (Chen et al. 2020) | https://arxiv.org/abs/2002.05709 |
| Masked Autoencoders Are Scalable Vision Learners (MAE) | Paper (He et al. 2021) | https://arxiv.org/abs/2111.06377 |
| Self-Supervised Representation Learning | Blog (Lilian Weng) | https://lilianweng.github.io/posts/2019-11-10-self-supervised |

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

## Paper Reading Order

Read alongside the build phases, not before them.

```
Phase 1 — Classical (Week 1-6)
  Tibshirani 1996 — Regression Shrinkage and Selection via the Lasso
  Breiman 2001 — Random Forests
  Friedman 2001 — Greedy Function Approximation: A Gradient Boosting Machine
  Freund & Schapire 1997 — A Decision-Theoretic Generalization (AdaBoost)
  Platt 1998 — Sequential Minimal Optimization

Phase 2 — Unsupervised (Week 7-9)
  Shlens 2014 — A Tutorial on Principal Component Analysis
  Ester et al. 1996 — A Density-Based Algorithm for Discovering Clusters (DBSCAN)
  Dempster et al. 1977 — Maximum Likelihood from Incomplete Data via the EM Algorithm
  Liu et al. 2008 — Isolation Forest
  van der Maaten & Hinton 2008 — Visualizing Data using t-SNE
  McInnes et al. 2018 — UMAP

Phase 3 — Probabilistic (Week 10-14)
  Kalman 1960 — A New Approach to Linear Filtering and Prediction Problems
  Rabiner 1989 — A Tutorial on Hidden Markov Models
  Lafferty et al. 2001 — Conditional Random Fields
  Engle 1982 — Autoregressive Conditional Heteroscedasticity
  Bollerslev 1986 — Generalized Autoregressive Conditional Heteroscedasticity

Phase 4-5 — Deep Learning (Week 15-26)
  Rumelhart et al. 1986 — Learning Representations by Backpropagating Errors
  LeCun et al. 1998 — Gradient-Based Learning Applied to Document Recognition
  Hochreiter & Schmidhuber 1997 — Long Short-Term Memory
  Ioffe & Szegedy 2015 — Batch Normalization
  Srivastava et al. 2014 — Dropout
  Ba et al. 2016 — Layer Normalization
  He et al. 2015 — Deep Residual Learning for Image Recognition
  Kingma & Welling 2013 — Auto-Encoding Variational Bayes
  Goodfellow et al. 2014 — Generative Adversarial Networks
  Ho et al. 2020 — Denoising Diffusion Probabilistic Models

Phase 6 — NLP (Week 27-32)
  Mikolov et al. 2013 — Efficient Estimation of Word Representations in Vector Space
  Vaswani et al. 2017 — Attention Is All You Need
  Devlin et al. 2018 — BERT
  Radford et al. 2019 — Language Models are Unsupervised Multitask Learners

Phase 7 — RL (Week 33-38)
  Watkins & Dayan 1992 — Q-Learning
  Mnih et al. 2013 — Playing Atari with Deep Reinforcement Learning
  Mnih et al. 2015 — Human-level Control Through Deep Reinforcement Learning
  Schulman et al. 2015 — High-Dimensional Continuous Control Using GAE
  Schulman et al. 2017 — Proximal Policy Optimization Algorithms
  Haarnoja et al. 2018 — Soft Actor-Critic
  Silver et al. 2017 — Mastering Chess and Shogi by Self-Play (AlphaZero)

Phase 8 — Self-Supervised (Week 39-42)
  Chen et al. 2020 — A Simple Framework for Contrastive Learning (SimCLR)
  He et al. 2021 — Masked Autoencoders Are Scalable Vision Learners
```

---

## How To Read a Paper

**Pass 1 (10 minutes):** Read title, abstract, introduction, conclusion, and look at all figures. Decide if the paper is worth a full read.

**Pass 2 (1 hour):** Read everything except proofs. Understand the problem, the key idea, and the results. Skip the math.

**Pass 3 (2-4 hours):** Implement the algorithm. Only now read the math carefully. Every equation becomes a line of code.

A paper you have implemented is worth ten papers you have only read.

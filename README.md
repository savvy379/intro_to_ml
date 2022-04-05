# Intro to Machine Learning
This course is primarily intended for non-computer science students who want to understand the foundations of building and testing an ML pipeline, different model types, important considerations in data and model design, and the role ML plays in research and society. To get the most out of the course you should have some experience coding (ideally in Python) and remember some general concepts from your calculus class, though you can certainly get a lot out of the material even without these pre-reqs!

Each day has a lecture and a corresponding notebook that implements the topics discussed and provides some short exercises (student notebook) and solutions (answer notebook). All coding exercises use python, and most models are built in Keras.

This course was originally designed as a Princeton University Winterssion Intensive. 

## Content

#### Day 1: Basics
**Topics**
- Conceptual foundations of ML
- Supervised and Unsupervised learning
- Training a model (optimization, data splitting, over and under fitting, loss functions, and evaluation metrics)
- Quick review of linear algebra
- Intro to python datascience ecosystem
- Linear and logistic regression as an ML algorithm

**Exercises**
- Exploration of key libraries (numpy, pandas, matplotlib)
- Implementing and tuning a linear regression model

[Day 1 Lecture](https://princeton.zoom.us/rec/share/LE9QgwUhH-i8OaWl_uyIhFd18HwD5eI9d_0AW1stDWco31Sg0TAdDBP_bly0SiGm.pEz6BA7R5kSHvBVF?startTime=1611598176000)

#### Day 2: Neural Networks 
**Topics**
- Intermediate step: Decision Trees and Forests
- Why NNs are useful
- Conceptual description of NNs
- Mathematical foundations of NNs (including gradient descent and back propagation)
- Optimizing NNs (hyperparameters, over and under fitting, regularization)

**Exercises**
- Coding a basic NN by hand (using Numpy)
- Building a NN in Keras to classify the Fashion MNIST dataset
- Optimizing NN hyperparameters and architecture

[Day 2 Lecture](https://princeton.zoom.us/rec/share/FdhEWbAZoP3uHFmsYqgG0uXbVnunJSMlLyRskrWc3vFZGE2X5Av0CSZGMfNds_gY.v-TB-H4Njtt2c22q?startTime=1611684755000)

#### Day 3: Convolutional and Recurrent Neural Networks
**Topics**
- Introduction to computer vision
- Conceptual and mathematical foundations of CNNs (filters, convolutions, pooling, network architectures)
- Demo of different interpretable filters
- Introduction to sequence data in ML
- Conceptual and mathematical foundations of RNNs (hidden states, training)
- LSTMs
- Demo using RNNs to generate text in a specific style

**Exercises**
- Image classification CNN using Keras on the CIFAR10 dataset
- Text classification RNN using Keras on the IMBD movie review dataset

[Day 3 Lecture](https://princeton.zoom.us/rec/share/YHBO38k-bvakNvdJziauK2Boq3nCbVw3RUueu0YlPYgkTyOyVWtQ5tWjWfY7aE0q.e4q2Y_NeXWJD5c8M?startTime=1611771197000)

#### Day 4: Unsupervised Learning and Generative Models
**Topics**
- Intro to unsupervised learning
- Clustering algorithms (kMeans, DBScan, Hieararchical) 
- Dimensionality Reduction (PCA, SVD, tSNE)
- Autoencoders
- Anomaly detection
- Demo using hierarchical clustering for customer segmentation
- Demo using autoencoders for health insurance fraud detection
- GANs
- Conditional GANs
- Adversarial examples and Fast Gradient Sign
- Gaussian Mixture Models
- Demo using GANs for image generation and style transfer
- Demo of SketchRNN and MusicVAE

**Exercises**
- Building an autoencoder for image segmentation (Keras)
- Comparing clustering algorithms (Sci-kit Learn)
- PCA in Sci-kit learn
- Implementing DCGAN in Keras

[Day 4 Lecture](https://princeton.zoom.us/rec/share/wE0yywHh4FkLj7q1dbsXZrnkyzx8_HPGPhXb5Ykp3Zjem1rW8wANj8C1Nsoo0lQR.EbUh0yeBdIiP8yNX?startTime=1611857502000)

#### Day 5: Other Topics
- Discuss examples of ML in daily life
- A scientific approach to ML model building
- Examples of ML in research (particle physics, disaster response, literature, and genomics)
- A primer on AI ethics concerns (data bias, algorithmic auditing, predictive policing, inequitable utilization of algorithms, proposed regulation)

[Day 5 Lecture](https://princeton.zoom.us/rec/share/BAu6ZhSPv8YLf8NuuzMTgafrvYy_qz8rE36D6DFl-eUM2kRvpIybOP6YQ5C1bG8l.OaXuQk_tE1LYB1Tg?startTime=1611944256000)

## Material Use
Please feel free to use these materials for your own learning or teaching (with credit attribution). If you have any suggestions please feel free to open a pull request or contact me at sthais@princeton.edu. 

**Under revision throughout week of 01/17/2022**. See individual days for lectures and notebooks.

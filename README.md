# Intro to Machine Learning
This course is primarily intended for non-computer science students who want to understand the foundations of building and testing an ML pipeline, different model types, important considerations in data and model design, and the role ML plays in research and society. To get the most out of the course you should have some experience coding (ideally in Python) and remember some general concepts from your calculus class, though you can certainly get a lot out of the material even without these pre-reqs!

Each day has a lecture and a corresponding notebook that implements the topics discussed and provides some shor exercises (student notebook) and solutions (answer notebook). All coding exercises use python, and most models are built in Keras.

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

#### Day 5: Other Topics
- Discuss examples of ML in daily life
- A scientific approach to ML model building
- Examples of ML in research (particle physics, disaster response, literature, and genomics)
- A primer on AI ethics concerns (data bias, algorithmic auditing, predictive policing, inequitable utilization of algorithms, proposed regulation)

## Material Use
Please feel free to use these materials for your own learning or teaching (with credit attribution). If you have any suggestions please feel free to open a pull request or contact me at sthais@princeton.edu. 

**Under construction throughout week of 01/25/2021**. See individual days for lectures and notebooks.

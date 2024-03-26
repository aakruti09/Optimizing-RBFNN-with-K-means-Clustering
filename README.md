# Optimizing RBFNN with KMeans Clustering

## Overview
This project aims to explore the use of Radial Basis Function Neural Networks (RBFNN) for data classification on non-linear patterns. In this, the focus will be on designing RBFNNs with different center selection strategies, optimizing hyperparameters, and comparing performance metrics to understand the impact on model accuracy and generalization.

## Background and Motivation
RBFNNs are a type of artificial neural network known for their ability to approximate complex non-linear functions. They consist of radial basis functions as activation functions, making them suitable for various machine learning tasks, especially when dealing with non-linear patterns. 

The motivation behind this project is to gain a deeper understanding of RBFNNs, explore different techniques for center selection, optimize model hyperparameters, and analyze their performance on non-linear datasets.

## Datasets
For this project, synthetic non-linear data is generated using mathematical functions.

Each data point is represented as **$x=(x_i,x_j)$**, where:
$$x_i = -2 + 0.2i \quad \quad i = 0,1,.. , 20$$
$$x_j = -2 + 0.2j \quad \quad j = 0,1,.. , 20$$

The mapping is defined by : 

$$f(x_1, x_2) = 
    \begin{cases}
        +1, & \quad \text{if } x_1^2 + x_2^2 \leq 1 \\
        -1, & \quad \text{if } x_1^2 + x_2^2 > 1
    \end{cases}
$$

$$\text{over region} -2 < x_1 < 2 \text{ and} -2 < x_2 < 2$$

## Practical Applications
- Pattern recognition and image classification
- Financial forecasting and stock market analysis
- Medical diagnosis and healthcare analytics
- Control systems and robotics for adaptive learning

## Milestones
1. Develop **Python code** to build **RBFNN models** using **TensorFlow, Keras, NumPy, Scikit-Learn** libraries.
2. Implement different **center selection strategies** (all data points, random selection, **KMeans clustering**) and compare their effects.
3. Conduct experiments to **optimize hyperparameters** such as spread parameter, learning rate, and number of centers.
4. **Evaluate model performance** using appropriate metrics like Mean Square Error (MSE) and visualize results.
5. **Final Analysis and Reporting:** Summarize findings, compare different approaches, and provide recommendations for future enhancements or applications.

## References
- [Most Effective Way To Implement Radial Basis Function Neural Network for Classification Problem](https://towardsdatascience.com/most-effective-way-to-implement-radial-basis-function-neural-network-for-classification-problem-33c467803319)
- Documentation and tutorials from scikit-learn for RBFNNs and KMeans clustering

<!-- ![question](https://user-images.githubusercontent.com/104097868/194796291-1a961149-80ca-4101-a88a-7ef0893977c4.png) -->

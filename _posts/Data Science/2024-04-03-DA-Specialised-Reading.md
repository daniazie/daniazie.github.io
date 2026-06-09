---
layout: post
title: "Data Analysis Specialised Reading: Kernel Trick"
published: true
date: 2024-04-03
math: true
categories: 
    - Study
    - Data Science
tags: KHUDA ML DA
---

# Kernel Trick
- Mathematical technique that maps data from one space to another space → uses Kernel function ⇒ takes as vectors in the original space as its inputs and returns the dot product of the vectors in the feature space
	- If we have data $\textbf{x}, \textbf{z} \in X$ and a map $\phi: X → \mathbb{R}^N$ then
			$k(\textbf{x}, \textbf{z}) = \left<\phi(\textbf{x}), \phi(\textbf{z})\right>$
			is a kernel function
		- consider each coordinate of the transformed vector $\phi(\textbf{x})$ is just some function oft he coordinates in the corresponding lower dimensional vector $\textbf{x}$ 
- Work with non-linearly separable data → map data points to a higher dimensional space where they become linearly separable
	- ex: mapping data from the 2nd dimension to a 3rd dimensional space so that they can be linearly separated by a plane (rather than a line)
- Generally used in SVMs → find the hyperplane that best separates the data points of different classes

## How It Works
1. Data Preprocessing: Data cleaning & normalisation
2. Kernel Function Selection: Selecting the best kernel function
3. Kernel Matrix Computation: Compute the kernel matrix that measures the similarity between the data pointers in the input space
4. Feature Space Mapping: Map the data into higher-dimensional space
5. Model Training: Train model on mapped data

## Types of Kernel Methods in SVM
1. Linear Kernel: used when the data is linearly separable
	$k(\textbf{x}, \textbf{z}) = \textbf{x} \cdot \textbf{z}$ 
2. Polynomial Kernel: used when the data is not linearly separable
	$k(\textbf{x}, \textbf{z}) = (\textbf{x} \cdot \textbf{z}+1)d$
3. Radial-basis Function (RBF) Kernel: maps input data to an infinite-dimensional space
	$k(\textbf{x}, \textbf{z}) = e^{-\gamma\left\Vert \textbf{x}-\textbf{z} \right\Vert^2}$
	- The Gaussian Kernel is a type of RBF kernel
4. Sigmoid Kernel: transforms input data using the Sigmoid kernel 
	$k(\textbf{x}, \textbf{z}) = \tanh{(\alpha \textbf{x}^T\textbf{z} + \beta)}$
5. Laplacian Kernel: similar to RBF but has a sharper peak and faster decay
	$k(\textbf{x}, \textbf{z}) = e^{-\left\Vert \textbf{x}-\textbf{z} \right\Vert}$ 
6. ANOVA Radial Basis Kernel: multiple-input kernel function → can be used for feature selection
	$k(\textbf{x}, \textbf{z}) = 1ne^{-(\textbf{x}k-\textbf{z}k)2}d$ 
7. Exponential Kernel: similar to RBF but decays much faster
	$k(\textbf{x}, \textbf{z}) = e^{-\gamma\left\Vert \textbf{x}-\textbf{z} \right\Vert^2_2}$
8. Wavelet Kernel: non-stationary kernel function that can be used for time-series analysis
	$k(\textbf{x}, \textbf{z}) = \sum{\phi(i,j)\psi(x(i), z(j))}$
9. Spectral Kernel: based on eigenvalues and eigenvectors of a similarity matrix
	$k(\textbf{x}, \textbf{z}) = \sum{\lambda_i\phi_i(\textbf{x})\phi_i(\textbf{z})}$
10. Mahalonibus Kernel: takes into account the covariance structure of the data
	$k(\textbf{x}, \textbf{z}) = e^{-\frac{1}{2}(\textbf{x}-\textbf{z})^TS^{-1}(\textbf{x}-\textbf{z})}$

## Choosing the Right Kernel Function
1. Understand the problem: Understand the type of data, features, and the complexity of the relationship between the features
2. Choose a simple kernel function: Start with the linear kernel function as a baseline to compare with the more complex ones
3. Test different functions: Play around with other kernel functions and compare their performance
4. Tune the parameters: Experiment with different parameter values and choose the values that results in the best performance
5. Use domain knowledge: Based on the type of data, use the domain knowledge and choose the appropriate type of kernel for the data
6. Consider computational complexity: Take into consideration the resources that would be required for larger data sets
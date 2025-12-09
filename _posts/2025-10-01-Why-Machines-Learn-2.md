---
layout: post
title: "[수학 공부] Why Machines Learn 2주차: Gradients"
published: true
date: 2025-10-01
math: true
categories: 
    - Study
    - 수학
tags: KHUDA
---

## 공지사항
* 이번 주부터 영어로 함

## The Basics of Calculus

Given a curve, we can use the tangent at any given point to find the slope of the curve. To find the tangent, we divide the change in $y$ by the change in $x$, 
$$\frac{\Delta y}{\Delta x}$$

where $\Delta$ represents a very small change in a given value. However, this would correspond to the movement _along the tangent_ as opposed to along the slope. However, as the changes in $x$, $\Delta x$, approaches 0, the tangent line also approaches th slope until the tangent and the slope are the same at $\Delta x = 0$. 

Here is where the problem comes in: ... how do we divide a value (in this case, $\Delta y$) by 0? This is where calculus comes in. Calculus allows us to calculate the ratio of the changes in two values even as the term approaches zero.

Specifically, $\frac{dy}{dx}$ is a little bit of $y$ divided by a little bit of $x$, and calculus allows us to calculate this ratio even when $dx \rightarrow 0$

## Gradient Descent

$$x_\text{new} = x_\text{old} - \eta\cdot\text{gradient} $$ 

$$ y_\text{new} = x_\text{new}^2$$

What we aim to do here is minimise $x$ in order to move down the gradient. $\eta$ is some fraction that represents how far along the gradient do we want to move (step size). That is, we want to find the global minimum.

#### Caution!!
* Some functions such as a hyperbolic paraboloid function ($z = y^2 - x^2$) do not have a minimum. These functions are often unstable due to its saddle point ~~세션 때 안 그려드리면 편하게 혼내세요~~, where by one false step might cause you to fall off the hyperplane. 

For a 3-D function, we need partial derivatives. Given $$z = x^2 + y^2,$$ we need find the partial derivatives of $z$. In this case, they are: $$\frac{\delta{z}}{\delta{x}} = 2x, \frac{\delta{z}}{\delta{y}} = 2y.$$

By calculating $2x$ and $2y$, you get a vector, which informs us the direction directly opposite to the steepest descent. The main takeaway is is that for a multidimensional function, the gradient is given by a vector. Given our elliptical paraboloid, its gradient would be written as: 

$$\begin{bmatrix}\delta z/ \delta x \\ \delta{z} /\delta{y}\end{bmatrix} = \begin{bmatrix}2x \\ 2y\end{bmatrix} or \begin{bmatrix}2x & 2y\end{bmatrix}$$


Reorganising the above equations, we get the following weight update rule: $$\mathbf{w}_\text{new} = \mathbf{w}_\text{old} + \mu (-\nabla), \text{where } \mu = \text{step size, } \nabla=\text{gradient}$$

However, with a large number of features, finding the gradient would be computatinally expensive, if not impossible. Thus, we estimate the gradient so that the update rule becomes: 

$$\mathbf{w}_\text{new} = \mathbf{w}_\text{old} + 2\mu\epsilon\mathbf{x} \text{ where, }$$ 

$$\mu = \text{step size}$$

$$ \epsilon = \text{error based on one data point}, $$

$$ \mathbf{x} = \text{the vector representing a single data point}$$

The error is given by:

$$\epsilon = d - \mathbf{w}\top\mathbf{x}, \text{where } d = \text{target}$$

And through this, we get the least mean squares algorithm.
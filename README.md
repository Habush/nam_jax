# Neural Additive Models in JAX

This repo contains JAX-based version of the model introduced in [Neural Additive Models: Interpretable Machine Learning with Neural Nets](https://arxiv.org/abs/2004.13912) by R. Agarwal et.al 2021. 

![NAM Architecture](https://camo.githubusercontent.com/8d26b1ac52a93281242b8e8b1dd4d15cbf2741977241fdc021a0e4e3b779b928/68747470733a2f2f692e696d6775722e636f6d2f487662377362322e6a7067)


### Dependencies

 - jax
 - optax
 - haiku # used for implementing NN model
 - torch # used for creating mini-batches
 - numpy
 - scikit-learn


### Examples

Checkout the `nam_regression_example.ipynb` notebook to see an example of using the model for the California housing Dataset
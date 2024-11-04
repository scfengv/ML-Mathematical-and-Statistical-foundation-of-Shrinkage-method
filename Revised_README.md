
# Mathematical and Statistical Foundation of Shrinkage Methods

## Contents
- Abstract
- Introduction
- Shrinkage Methods in Mathematics and Statistics
    - Bayesian Perspective
    - Gradient Descent Approach
- Lasso Feature Selection
- Logistic Regression
- References

# Abstract

This document explores the mathematical and statistical foundations behind shrinkage methods, focusing on understanding the fundamental differences between Lasso and Ridge regression through Bayesian and gradient descent perspectives. The wave dataset is utilized to demonstrate variable reduction using Lasso regression and stepwise selection, reducing the original 121 variables to 14 and 9, respectively. Classification using logistic regression yielded accuracy rates of 92.36% for Lasso and 91.96% for stepwise selection.

# Introduction

The dataset used in this study is the well-known artificial wave dataset (Breiman et al., 1984). Originally containing 21 variables, Rakotomalala (2005) introduced 100 noise variables, expanding the data to a shape of (33334, 121). Variables are labeled as v1, v2, …, v21 and alea1, alea2, …, alea100. It can be inferred that v1, v2, …, v21 (henceforth referred to as v-type variables) are the original set, while alea1 to alea100 (henceforth referred to as a-type variables) are noise. Two initial hypotheses are proposed:

1. "v-type variables are more significant."
2. "a-type variables are either non-contributory or detrimental to the model."

Before commencing variable selection, it is expected that the results would ideally include only v-type variables and exclude a-type variables.

Given the dataset's division according to the classification target, stratified sampling using `StratifiedKFold` is applied for cross-validation to maintain class proportions across all folds, ensuring unbiased results.

```python
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

# Shrinkage Methods in Mathematics and Statistics

Shrinkage methods include Lasso and Ridge regression, both of which can be seen as modifications of Ordinary Least Squares (OLS). OLS often fits the training data well (low bias) but tends to perform poorly on unseen data (high variance), indicating overfitting. Shrinkage methods address this by introducing a penalty term, denoted as $\lambda$, which adjusts variable weights to prevent overfitting and enhance generalizability. This balance between variance and bias is known as the variance-bias trade-off, and the optimal $\lambda$ can be determined through cross-validation.

### Bayesian Perspective

The OLS problem is defined as:

$$
y = X\beta + \varepsilon
$$

OLS estimates have high variance:

$$
\hat{\beta}_{OLS}
$$

The regularization objectives for Lasso and Ridge are:

- **Lasso (L1)**: 
$$
\hat{\beta}_{L1} = \arg \min_{\beta} \left[ ||y - X\beta||_2^2 + \lambda \|\beta\|_1 \right]
$$

- **Ridge (L2)**: 
$$
\hat{\beta}_{L2} = \arg \min_{\beta} \left[ ||y - X\beta||_2^2 + \lambda \|\beta\|_2^2 \right]
$$

The penalty term influences the coefficients, with $\lambda$ driving regularization strength. Under the Bayesian framework, the maximum a posteriori (MAP) estimate can be expressed as:

$$
\hat{\beta} = \arg \max_{\beta} P(\beta|y)
$$

$$
\hat{\beta} = \arg \max_{\beta} \left[ \log P(y|\beta) + \log P(\beta) \right]
$$

Assuming:

$$
y_i \sim N(\beta^T x_i, \sigma^2), \quad \beta_j \sim N(0, \tau^2)
$$

The results yield interpretations consistent with Lasso and Ridge.

### Gradient Descent Approach

For gradient descent, the update rule is:

$$
w_{new} = w - \gamma \frac{\partial L}{\partial w}
$$

where $\gamma$ is the learning rate. This approach shows how regularization terms influence the weight updates, preventing large coefficients and overfitting.

# Lasso Feature Selection

Using `sklearn.linear_model.Lasso`, variables were reduced, highlighting significant differences in variable selection compared to stepwise methods. Results indicated that at certain $\lambda$ thresholds, most a-type variables converge to zero, emphasizing their minimal importance.

```python
params = {
    'alpha': [10**i for i in range(-5, 5)],
    'selection': ['cyclic', 'random']
}

# Results
Best alpha: {'alpha': 0.001, 'selection': 'cyclic'}
Best scores: 0.6604
```

# Logistic Regression

Five feature groups were tested: without selection, Lasso, stepwise, all v-type, and all a-type. Results confirmed that Lasso feature selection outperformed others, while a-type variables introduced noise without significant model contribution.

| Method | w/o Selection | Lasso | Stepwise | All v-type | All a-type |
|--------|---------------|-------|----------|------------|------------|
| Accuracy | 92.14% | 92.36% | 91.96% | 92.43% | 50.35% |
| AUC | 0.98 | 0.98 | 0.98 | 0.98 | 0.50 |
| Most Important Variable | v17 | v17 | v17 | v17 | alea73 |

# References

1. Regularization in Machine Learning. [GeeksforGeeks](https://www.geeksforgeeks.org/regularization-in-machine-learning/)
2. Shrinkage Methods. [M0nads](https://m0nads.wordpress.com/2018/01/15/shrinkage-methods-ridge-regression-and-lasso/)
3. Bias-Variance Tradeoff. [Weebly](https://jason-chen-1992.weebly.com/home/-bias-variance-tradeoff)
4. Intuitions on L1 and L2 Regularisation. [Towards Data Science](https://towardsdatascience.com/intuitions-on-l1-and-l2-regularisation-235f2db4c261#dda9)
5. Regularization (Mathematics). [Wikipedia](https://zh.wikipedia.org/zh-tw/%E6%AD%A3%E5%88%99%E5%8C%96_(%E6%95%B0%E5%AD%A6))
6. Gradient Descent Optimization. [Medium](https://chih-sheng-huang821.medium.com/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E5%9F%BA%E7%A4%8E%E6%95%B8%E5%AD%B8-%E4%B8%89-%E6%A2%AF%E5%BA%A6%E6%9C%80%E4%BD%B3%E8%A7%A3%E7%9B%B8%E9%97%9C%E7%AE%97%E6%B3%95-gradient-descent-optimization-algorithms-b61ed1478bd7)
7. Gradient Descent. [Wikipedia](https://en.wikipedia.org/wiki/Gradient_descent)
8. GLMNET Python Example. [Notebook Community](https://notebook.community/ceholden/glmnet-python/examples/glmnet_demo)
9. Ridge Regression Differences. [Stats Stack Exchange](https://stats.stackexchange.com/questions/160096/what-are-the-differences-between-ridge-regression-using-rs-glmnet-and-pythons)
10. Bayesian Linear Regression. [YouTube](https://www.youtube.com/watch?v=Z6HGJMUakmc&ab_channel=ritvikmath)

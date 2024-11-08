# Mathematical and Statistical foundation of Shrinkage method

## Content
- Abstract
- Introduction
- Shrinkage method in Mathematic and Statistic
	- Bayesian
	- Gradient Descent
- Lasso Feature Selection
- Logistic Regression
- Reference

# Abstract

This article explores the mathematical and statistical foundations behind shrinkage methods and uses Bayesian and gradient descent perspectives to understand the fundamental differences between Lasso and Ridge regression. Finally, I employed the well-known artificial wave dataset to reduce the original 121 variables to 14 and 9 variables using Lasso regression and stepwise selection, respectively. I then perform classification comparisons using logistic regression, achieving classification accuracies of 92.36% and 91.96%, respectively.

# Introduction

I utilized the well-known artificial wave dataset (Breiman *et al.*, 1984) in this study. The original dataset contained 21 variables. In 2005, Rakotomalala added 100 noise variables to the dataset, resulting in a dataset with dimensions (33334, 121). The variable names are v1, v2, …, v21, alea1, alea2, …, alea100. While the names do not provide much information, I can infer that v1, v2, …, v21 (hereafter referred to as v-type variables) are the original variables in the dataset, and alea1, alea2, …, alea100 (hereafter referred to as a-type variables) are the noise variables added by Rakotomalala. Before commencing any analysis, I have two simple hypotheses regarding the results:

1. "V-type variables are more important"
2. "A-type variables do not help the model and may even harm it"
3. 
Therefore, I expect that the result of variable selection should ideally include only V-type variables and exclude A-type variables.

Additionally, upon examining the dataset, the data is roughly divided into upper and lower halves according to the classification target results. Therefore, when performing K-Fold Cross Validation,I need to use stratified sampling different from the usual simple K-Folds, specifically `StratifiedKFold`. This cross-validation method ensures that each fold maintains the same class proportion as the entire dataset, preventing class bias in each fold due to initial data ordering and other factors.

```python
kf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
```

# Shrinkage method in Mathematic and Statistic

Shrinkage methods include Lasso regression and Ridge regression, which can be seen as modifications of Ordinary Least Squares (OLS). OLS has a problem: the OLS model can fit the training data very well (low bias), but performs poorly on unseen data (validation and testing data), exhibiting high variance. This is often due to overfitting when fitting the training set. To address this issue, shrinkage methods introduce a $\lambda$ term (penalty term) that is independent of the model and data to adjust the weight of each variable in the model. By “not predicting so accurately” (increasing bias), we enhance the model’s predictability on unseen data (reducing variance). This trade-off between variance and bias is known as the Variance-Bias Trade-off, and the optimal balance point $\lambda$ can be found through cross-validation.

![Untitled](https://github.com/scfengv/Mathematical-and-Statistical-foundation-of-Shrinkage-method/assets/123567363/bb9513c4-3c60-4028-bb50-d7cfdfed78c5)

Fig. 1 [1]

Figure 2 is an excellent example. The blue line ($\lambda = 0$) represents OLS. In the right figure, we can clearly see that this line results in overfitting, being overly influenced by certain data points, making the model less applicable in general situations. After introducing $\lambda$, the fitting process tends to “penalize” variables when they acquire large weights, reducing the weights to avoid overfitting. However, $\lambda$ cannot be too large. For instance, when $\lambda = 100$ (red line) in the left figure, the regression line becomes almost flat ($y = 1.7$), meaning that regardless of the value of $x_1$, the model predicts $y = 1.7$. This is a clear case of underfitting. The adjustment of $\lambda$ can be determined through cross-validation.

<img width="500" src="https://github.com/scfengv/Mathematical-and-Statistical-foundation-of-Shrinkage-method/assets/123567363/e2b9dd5a-9789-4f6a-a273-7e6b24c90db8">
<img width="300" src="https://github.com/scfengv/Mathematical-and-Statistical-foundation-of-Shrinkage-method/assets/123567363/9f9a4eaa-6a63-4bef-bc05-d37c1e5e6a8d">
Fig. 2 [2] & Fig. 3 [3]

Below, I will attempt to explain how shrinkage methods use the $\lambda$ penalty term to adjust the OLS model from both Bayesian and Gradient Descent perspectives.

## Bayesian

$Ordinary\ Least\ square$

$$
y=X\beta+\varepsilon
$$

$$
Problem\ of\ OLS\ :\ \hat{\beta}_{OLS}\ have\ high\ variance
$$

$$
Small\ \Delta x \ would\ lead\ to \ Large\ \Delta \beta , which\ will\ lead\ to\ bad\ prediction\ for\ test\ set\ in\ ML
$$

***Varianve-Bias Trade-off***

$Regularization$

$$
Lasso\ :\ \hat{\beta}_{L1} = \arg \min _{\beta} [{\color{blue}{ |y-X \beta |}_2^2} + {\color{red} \lambda \| \beta \|_1}]
$$

$$
Ridge\ :\ \hat{\beta}_{L2} = \arg \min _{\beta} [{\color{blue}{ |y-X \beta |}_2^2} + {\color{red} \lambda \| \beta \|_2^2}]
$$


$$
\left \| \beta\right \|_1 = \sum _{i=1}^{N} \left | \beta_i \right |
$$

$$
\left \| \beta\right \|_2 = [\sum _{i=1}^{N} (\beta_i)^2]^{\frac{1}{2}}
$$

$$
\left \| \beta\right \|_p = [\sum _{i=1}^{N} (\beta_i)^p]^{\frac{1}{p}}
$$

The blue part represents the error, i.e., the Mean Squared Error (MSE) between the actual value ($y$) and the predicted value ($X\beta$), which is the original OLS model.

The red part is the regularization driven by $\lambda$, also known as the penalty term. The magnitude of $\lambda$ represents the strength of the penalty, which can shrink $\beta$ as much as possible, even setting $\beta = 0$ in Lasso regression, effectively removing the variable from the model.

How does the shrinkage method reduce the variance of OLS by adjusting the value of $\lambda$? We can understand this from a Bayesian perspective.


### Bayesian viewpoint


$$
\hat{\beta} = {\arg \max_{\beta}} P(\beta|y)
$$

$$
\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ = {\arg \max_{\beta}} ( \frac{P(y|\beta) * P(\beta)}{P(y)})
$$

$$
\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ =\arg \max_{\beta}({\color{blue}P(y|\beta)}*{\color{red}P(\beta)})
$$

$$
\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ =\arg \max_{\beta}[\ {\color{blue}log(P(y|\beta))}+{\color{red}log(P(\beta))}\ ]
$$

$$
{\color{blue}P(y|\beta)}\ :\ Likelihood
$$

$$
{\color{red}P(\beta)}\ :\ Prior
$$

$\hat{\beta}_{MAP}$ is the Maximum A Posteriori estimate, which seeks the value of $\beta$ that maximizes the posterior probability given the data $y$.

The blue part is the likelihood function, representing the probability of observing specific $y$ given $\beta$.

The red part is the prior, representing the probability of $\beta$ without any given conditions; it can be understood as the probability distribution or assumptions about $\beta$.

$Assume$

$$
y_i \sim N(\beta^Tx_i\ ,\  \sigma^2)
$$

$$
\beta_j \sim N(0,\ \tau^2)
$$

#### $\beta \ under\ a\ {\color{red}Gaussian\ prior}$

$$
P(y| \beta)=\prod_{i=1}^{N} \frac{1}{\sigma\sqrt{2\pi}} \exp (- \frac{(y_i - \beta^T x_i)^2}{2 \sigma^2})
$$

$$
Log(P(y|\beta))= \sum_{i=1}^N\ [\ log(\frac{1}{\sigma\sqrt{2\pi}})- \frac{(y_i - \beta^T x_i)^2}{2 \sigma^2} ]
$$

$$
{\arg \min_{\beta}} [|| y - X \beta||_2^2 + \frac{\sigma^2}{\tau^2} || \beta ||_2^2]
$$

$$
= {\arg \min_{\beta}} [|| y - X \beta||_2^2 + {\color{red} \lambda} || \beta ||_2^2] = \left \| \left \| \beta\right \| \right \|_2 \rightarrow \ {\color{red} Ridge}
$$

#### $\beta \ under\ a\ {\color{red}Laplacian\ prior}$

$$
P(y| \beta)=\prod_{i=1}^{N} \frac{1}{2b} \exp (- \frac{|y_i - \beta^T x_i|}{b})
$$

$$
Log(P(y|\beta))= \sum_{i=1}^N\ [\ log(\frac{1}{2b})- \frac{(y_i - \beta^T x_i)}{b} ]
$$

$$
{\arg \min_{\beta}} [|| y - X \beta||_2^2 + \frac{\sigma^2}{\tau^2} || \beta ||_1]
$$

$$
= {\arg \min_{\beta}} [|| y - X \beta||_2^2 + {\color{red} \lambda} || \beta ||_1] = \left \| \left \| \beta\right \| \right \|_1 \rightarrow \ {\color{red} Lasso}
$$

By introducing assumptions about $\beta$, regularization can, through different probability distributions, result in the same expressions for Lasso and Ridge as we had initially.

Knowing that $\lambda$ represents the strength of regularization, we can understand from the assumed probability distributions of $\beta$ that as $\lambda$ increases (i.e., as $\tau^2$ decreases), the probability distribution will make $\beta$ closer to 0 (Fig. 4, green → orange → blue). Under different priors, the Gaussian distribution (Ridge, Fig. 4-1) is a random distribution around 0, while the Laplacian distribution (Lasso, Fig. 4-2) specifies that most coefficients are 0, thus achieving the purpose of feature selection.

<img width="450" src="https://github.com/scfengv/Mathematical-and-Statistical-foundation-of-Shrinkage-method/assets/123567363/ff62c5a3-9236-4645-be4e-c8335f2cf270">
<img width="450" src="https://github.com/scfengv/Mathematical-and-Statistical-foundation-of-Shrinkage-method/assets/123567363/b996d417-b783-42d5-838c-711a9a1bf089">

Fig. 4-1 & Fig. 4-2

Additionally, the two $\arg \min$ expressions for Lasso and Ridge can also be represented as:

$$
Lasso\ :\ {\arg \min_{\beta}} [{\|y-X\beta\|}_2\ ^2+\lambda\|\beta\|_1]
$$

$$
Ridge\ :\ {\arg \min_{\beta}} [{\|y-X\beta\|}_2\ ^2+\lambda\|\beta\|_2^2]
$$

For every value of $\lambda$, there is an $s$ that satisfies:

$$
Lasso:min \lbrace \sum_{i=1}^N (y_i-\beta_0-\sum_{j=1}^p\beta_jx_{ij})^2\ \rbrace \ subject\ to\ \sum_{j=1}^p |\beta_j|\leq s
$$

$$
Ridge:min \lbrace \sum_{i=1}^N (y_i-\beta_0-\sum_{j=1}^p\beta_jx_{ij})^2 \rbrace \ subject\ to\ \sum_{j=1}^p \beta_j^2\leq s
$$

If we plot the constraints of Lasso and Ridge in 2D (Fig. 5), we can see that the $\beta_j$ in Lasso are constrained within a diamond-shaped region whose vertices lie on the axes ($|\beta_1| + |\beta_2| \leq s$), while the $\beta_j$ in Ridge are constrained within a circle ($\beta_1^2 + \beta_2^2 \leq s$). $\hat{\beta}$ is the solution from Least Squares, and the red lines are the contours of equal Residual Sum of Squares (RSS). As previously mentioned, introducing the penalty term $\lambda$ is a process of increasing bias to exchange for better adaptability to different data (reducing variance). Mathematically, this means finding the first intersection point that satisfies the boundary condition by increasing the RSS; this intersection point is the optimal solution under the constraint. Due to the boundary conditions, the contours of equal RSS often intersect at the vertices of the Lasso diamond (which lie on the axes due to the conditions), causing $\beta_j = 0$ for the axes not intersected, which is the feature selection process mentioned earlier. This idea can be extended to higher dimensions similarly.

![IMG_CDC01411653B-1](https://github.com/scfengv/Mathematical-and-Statistical-foundation-of-Shrinkage-method/assets/123567363/6f549137-3e2d-453f-b001-f08f1d32a927)

Fig. 5


## Gradient Descent

Alternatively, we can understand Lasso and Ridge through gradient descent. Here, I express linear regression using weights and biases, which is similar to the previous representations.

$$
Linear\ Regression\ : \hat{y} = w_1x_1+w_2x_2+...+w_Nx_N+b
$$

$$
w:weight,\ b:bias
$$

***Loss Function of each Regularization:***


$No\ Regularization$

$$
L=(\hat{y}-y)^2\ \ \ \ \ \ \ \ \ \ \ \ \ 
$$

$$
\ =(wx+b-y)^2
$$

$$
\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ =Error\ between\ \hat{y}\ (pred.\ value)\ and\ y\ (true\ value)
$$

$L1\ regularization$

$$
L_1=(wx+b-y)^2+\lambda|w|
$$

$L2\ regularization$

$$
L_2=(wx+b-y)^2+\lambda w^2
$$

By introducing gradient descent and considering the three scenarios, we observe the changes in $w$.

$$
w_{new}=w-\gamma \frac{\partial L}{\partial w}
$$

$$
\gamma:Learning\ rate
$$

$Assume$

$$
Learning\ rate\ (\gamma)=1
$$

$$
H=2x(wx+b-y)
$$


$No\ Regularization$

$$
w_{new}=w-\gamma\ [\ 2x(wx+b-y)\ ]
$$

$$
=w-H\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ 
$$

$L1\ Regularization$

$$
w_{new}=w-\gamma\ [\ 2x(wx+b-y)+\lambda \frac{d|w|}{w}\ ]
$$

$$
\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ =w-\gamma\ [\ 2x(wx+b-y)+\lambda\ ] \ \ \ (w>0)
$$

$$
\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ =w-\gamma\ [\ 2x(wx+b-y)-\lambda\ ] \ \ \ (w<0)
$$

$$
=(w-H)-\lambda \ \ \ (w>0)\ \ \ \ \ \ \ \ \ \ 
$$

$$
=(w-H)+\lambda \ \ \ (w<0)\ \ \ \ \ \ \ \ \ \ 
$$

$L2\ Regularization$

$$
w_{new}=w-\gamma\ [\ 2x(wx+b-y)+2\lambda w\ ]
$$

$$
=(w-H)-2\lambda w \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ 
$$

Here, we can first assume that the linear regression perfectly fits the training data. However, this also means that the model performs poorly on unfamiliar data (validation, test data), as previously mentioned (high variance due to overfitting). Introducing regularization to linear regression penalizes the model when it tends to overfit, since $\lambda$ is a constant independent of the model and data. From the expression for $w_{\text{new}}$ in L1 regularization, when $w > 0$, $-\lambda$ reduces $w$; similarly for $w < 0$. L2 regularization does something similar.

Combining the results from the Bayesian and gradient descent perspectives, as $\lambda$ increases, the $\beta_j$ distribution (both Lasso and Ridge) in the Bayesian result concentrates more around zero. In the gradient descent result, $|w_{\text{new}}|$ becomes smaller. Simply put, a stronger penalty reduces $w$ more, thus avoiding overfitting. However, note that $\lambda$ is not necessarily better when larger; the process of determining $\lambda$ involves a variance-bias trade-off behavior, so cross-validation is needed to find the optimal $\lambda$ for each model.

# Lasso Feature Selection

In this study, we use the Lasso function from the sklearn.linear_model module (which is similar to glmnet.lasso; differences can be found in these articles: [Generalized Linear Models and Elastic Nets (GLMNET)](https://notebook.community/ceholden/glmnet-python/examples/glmnet_demo), [What are the differences between Ridge regression using R's glmnet and Python's scikit-learn?](https://stats.stackexchange.com/questions/160096/what-are-the-differences-between-ridge-regression-using-rs-glmnet-and-pythons)). Note that in sklearn, $\lambda$ is referred to as alpha, but this does not affect the methodology; we will use $\lambda$ here.

After understanding the theory, we can apply Lasso regression for feature selection. Observing the Lasso Penalty vs. Coefficients graph below (Fig. 6-1), we can see that as $\lambda$ increases, the coefficients indeed converge. When $\lambda > 10^{-0.5}$, all coefficients shrink to zero, meaning the model contains no variables, which corresponds to the situation in the left plot of Fig. 2, where the model becomes a constant value. From the cross-validation results, the best performance is at $\lambda = 10^{-3}$, selecting 13 of the 21 V-type variables.

In variable selection, adjustments were made. As shown below, at $\lambda = 10^{-3}$, many variables have not yet converged to zero. If we set the threshold for selecting variables as coefficients > 0, we would include too many variables. Therefore, at $\lambda = 10^{-3}$, the threshold was set as coefficients > 0.01. However, at the already converged $\lambda = 10^{-2}$ (though not the best parameter selected by cross-validation), setting the threshold as coefficients > 0 yields a similar result, but with one additional variable `v20`.

<img width="600" src="https://github.com/scfengv/Mathematical-and-Statistical-foundation-of-Shrinkage-method/assets/123567363/33085962-37ff-4182-aab9-29c5f034fc87">

Fig. 6-1

<img width="450" src="https://github.com/scfengv/Mathematical-and-Statistical-foundation-of-Shrinkage-method/assets/123567363/8118174f-2587-4b71-8fda-e401e352fb85">
<img width="450" src="https://github.com/scfengv/Mathematical-and-Statistical-foundation-of-Shrinkage-method/assets/123567363/e4145a03-6ef0-4722-b95b-721860cea9d8">

Fig. 6-2 & Fig. 6-3

```python
params = {
    'alpha': [10**i for i in range(-5, 5)],
    'selection': ['cyclic', 'random']
}

Best alpha: {'alpha': 0.001, 'selection': 'cyclic'}
Best scores: 0.6603986868079547
```

```python
## alpha = 0.001
print(np.array(X.columns)[importance > 0.01])
Selected Feature: ['v4' 'v5' 'v8' 'v9' 'v10' 'v11' 'v12' 'v13' 'v15' 'v16' 'v17' 'v18' 'v19']

## alpha = 0.01
print(np.array(X.columns)[importance > 0])
Selected Feature: ['v4' 'v5' 'v8' 'v9' 'v10' 'v11' 'v12' 'v13' 'v15' 'v16' 'v17' 'v18' 'v19' 'v20']
```

## Compared Lasso with Stepwise Selection

Comparing the variables selected by Lasso and stepwise selection, we see some differences. Both methods only selected V-type variables, but from the performance of stepwise selection, we can see that accuracy does not significantly increase after about 9 variables.

```python
# Stepwise selection

Selected Feature: ['v9', 'v10', 'v11', 'v12', 'v13', 'v15', 'v16', 'v17', 'v18']
```

<img width="600" src="https://github.com/scfengv/Mathematical-and-Statistical-foundation-of-Shrinkage-method/assets/123567363/bcf72a76-d09f-49e1-bc57-7978f3e6f418">

Fig. 7

However, during the variable selection process, a very interesting phenomenon occurred. Looking into the stepwise variable selection process, we can see that contrary to our initial assumption that V-type variables are important and A-type variables are just noise, an A-type variable 'alea81' was selected as the 10th variable, entering the feature group earlier than the remaining 12 V-type variables. The 11th variable selected was 'v4', and the next V-type variable 'v8' was selected as the 13th variable. According to our assumption, the first 21 variables selected should all be V-type variables, but after the 13th variable 'v8', no more V-type variables were selected up to the 30th variable.

```python
# Stepwise selection

sfs.get_metric_dict()[9]
feature_names: ['v9','v10','v11','v12','v13','v15','v16','v17','v18']

sfs.get_metric_dict()[10]
feature_names: ['v9','v10','v11','v12','v13','v15','v16','v17','v18','alea81']

sfs.get_metric_dict()[11]
feature_names: ['v4','v9','v10','v11','v12','v13','v15','v16','v17','v18','alea81']

sfs.get_metric_dict()[21]
feature_names: ['v4','v8','v9','v10','v11','v12','v13','v15','v16','v17','v18',
		'alea81','alea7','alea20','alea24','alea26','alea29','alea32','alea48','alea71','alea81','alea85']

sfs.get_metric_dict()[30]
feature_names: ['v4','v8','v9','v10','v11','v12','v13','v15','v16','v17','v18',
		'alea2','alea7','alea9','alea18','alea24','alea26','alea29','alea32','alea35','alea42','alea48','alea51','alea58','alea71',
		'alea81','alea85''alea95','alea98','alea100']
```

Did a similar situation occur with Lasso? The answer is yes. Recall the issue around $\lambda$ at $10^{-3}$ and $10^{-2}$. The reason for ultimately choosing $\lambda = 10^{-2}$ is that at $\lambda = 10^{-3}$, most A-type variables’ coefficients had not yet converged. Looking more closely, we find that at $\lambda = 10^{-3}$, two v-type variables 'v1' and 'v7' had already converged to zero. If we raise the coefficient threshold to 0.05 (no substantial meaning), we see that the only remaining a-type variable is 'alea70', indicating that at $\lambda = 10^{-3}$, most a-type variables were about to converge. However, we also find that the v-type variable 'v2' disappeared, and at $\lambda = 10^{-2}$, 'v3', 'v4', 'v6', 'v14', and 'v21' also disappeared.

Summarizing the findings during feature selection, we see that our initial hypothesis “v-type variables are more important” is generally correct but fails in some cases, possibly due to setting the threshold too strictly or because Rakotomalala deliberately designed the added noise variables to not significantly affect the results but not deviate too much. As seen in Fig. 7, after a-type variables start to be added, the model performance does not significantly improve after about the 10th variable, and after the 13th variable 'v8', performance remains almost flat. This indicates that a-type variables do not contribute much to the model (neither helping nor harming). Comparing this to our initial hypotheses:
1.	“V-type variables are more important.”
2.	“A-type variables do not help the model and may even harm it.”

Both are generally correct but have exceptions. Whether a-type variables harm the model will be discussed later in the classification learning stage.

```python
# Lasso Selection

## alpha = 0.001
print(importance > 0)

feature_names: ['v2' 'v3' 'v4' 'v5' 'v6' 'v8' 'v9' 'v10' 'v11' 'v12' 'v13' 'v14' 'v15' 'v16' 'v17' 'v18' 'v19' 'v20' 'v21' 
		'alea1'  'alea3'  'alea5'  'alea6'  'alea7' 'alea8' 'alea10' 'alea11' 'alea12' 'alea13' 'alea14' 'alea15'
		'alea16' 'alea17' 'alea18' 'alea20' 'alea21' 'alea23' 'alea24' 'alea25' 'alea26' 'alea27' 'alea28' 'alea31' 
		'alea33' 'alea34' 'alea35' 'alea37' 'alea39' 'alea40' 'alea43' 'alea44' 'alea46' 'alea47' 'alea48' 'alea49'
		'alea50' 'alea52' 'alea54' 'alea55' 'alea56' 'alea58' 'alea60' 'alea61' 'alea62' 'alea64' 'alea65' 'alea66' 
		'alea67' 'alea68' 'alea69' 'alea70' 'alea72' 'alea73' 'alea74' 'alea75' 'alea76' 'alea77' 'alea78' 'alea79'
		'alea80' 'alea81' 'alea82' 'alea83' 'alea84' 'alea85' 'alea86' 'alea87' 'alea88' 'alea89' 'alea90' 'alea94' 
		'alea95' 'alea96' 'alea97' 'alea98' 'alea99' 'alea100']

## alpha = 0.001
print(importance > 0.05)

feature_names: ['v3' 'v4' 'v5' 'v6' 'v8' 'v9' 'v10' 'v11' 'v12' 'v13' 'v14' 'v15' 'v16' 'v17' 'v18' 'v19' 'v20' 'v21' 
								'alea70']

## alpha = 0.01
print(importance > 0)

feature_names: ['v4' 'v5' 'v8' 'v9' 'v10' 'v11' 'v12' 'v13' 'v15' 'v16' 'v17' 'v18' 'v19' 'v20']
```

# Logistic Regression

In the classification phase, I experimented with five feature groups:

	1.	Without feature selection
	2.	Lasso feature selection
	3.	Stepwise feature selection
	4.	All ‘v’ variables
	5.	All ‘a’ variables

Since the dataset is balanced, accuracy was used to compare the models. The results indicate that among the groups employing feature selection, using Lasso for feature selection achieved the best performance, followed by selecting all features, and then stepwise selection, although the differences were not substantial.

Additionally, when comparing the models without feature selection and with Lasso feature selection, a slight decrease in accuracy is observed in the model without feature selection; however, the decrease is minimal. Based on previous results, it can be inferred that the difference between these two feature groups is almost entirely due to the A-type variables. Referring back to the initial hypothesis 2: “The A-type variables do not help the model and may even harm it,” it can be concluded that the presence of A-type variables increases noise in the model, causing some impact, but the effect is limited. Therefore, A-type variables can be considered pure noise, which does not significantly harm the model but should be removed to reduce the model’s complexity.

Furthermore, we observed that `v17` consistently ranked first in feature importance across the three groups. Finally, when comparing models using all V-type variables and all A-type variables, we found that the performance of using all V- variables surpassed that of the Lasso-selected features. This is expected; as long as the model is not adversely affected (e.g., by extreme values), increasing the number of variables generally leads to better fitting. However, the purpose of feature selection is to reduce the model’s complexity and to avoid overfitting by reducing the number of variables.

|  | w/o Feature selection | Lasso | Stepwise | all ‘v’ | all ‘a’ |
| --- | --- | --- | --- | --- | --- |
| Accuracy | 92.14% | 92.36% | 91.96% | 92.43% | 50.35% |
| AUC | 0.98 | 0.98 | 0.98 | 0.98 | 0.50 |
| Most important variable | v17 | v17 | v17 | v17 | alea73 |

## Without Feature Selection

$$
Accuracy:92.1359\\%
$$

$$
Precision: 92.1531\\%
$$

$$
Recall: 92.1359\\%
$$

$$
f1 : 92.1358\\%
$$

|  | True | False |
| --- | --- | --- |
| Pred. True | 10729 | 1031 |
| Pred. False | 804 | 10770 |

<img width="450" src="https://github.com/scfengv/Mathematical-and-Statistical-foundation-of-Shrinkage-method/assets/123567363/58c3c9a3-9a95-445e-9c79-8dbff38b337e">
<img width="450" src="https://github.com/scfengv/Mathematical-and-Statistical-foundation-of-Shrinkage-method/assets/123567363/eb668cd7-121f-493d-869d-deae892af283">

Fig. 8-1 & Fig. 8-2

## Lasso Feature Selection

### Tuning Parameter

```python
params = {
    'C': [10**i for i in range(-4, 5)],
    'penalty': ['l2', 'elasticnet', 'None'],
    'solver': ['lbfgs', 'liblinear', 'saga']   
}

Best parameters: {'C': 0.1, 'penalty': 'l2', 'solver': 'liblinear'}
Best score: 0.9193
```

### Model Performance

$$
Accuracy: 92.3631 \\%
$$

$$
Precision: 92.3826 \\%
$$

$$
Recall: 92.3631 \\%
$$

$$
f1: 92.3629 \\%
$$

|  | True | False |
| --- | --- | --- |
| Pred. True | 10748 | 1012 |
| Pred. False | 770 | 10804 |

<img width="450" src="https://github.com/scfengv/Mathematical-and-Statistical-foundation-of-Shrinkage-method/assets/123567363/3811daf0-351e-42d0-bf3d-9860e97c11ef">
<img width="450" src="https://github.com/scfengv/Mathematical-and-Statistical-foundation-of-Shrinkage-method/assets/123567363/d01d9339-31f6-44f1-be70-544359aa6929">

Fig. 9-1 & Fig. 9-2

## With Stepwise Selection

### Tuning Parameter

```python
params = {
    'C': [10**i for i in range(-4, 5)],
    'penalty': ['l2', 'elasticnet', 'None'],
    'solver': ['lbfgs', 'liblinear', 'saga']   
}

Best parameters: {'C': 0.01, 'penalty': 'l2', 'solver': 'lbfgs'}
Best score: 0.9148
```

### Model Performance

$$
Accuracy: 91.9559 \\%
$$

$$
Precision: 91.9924 \\%
$$

$$
Recall: 91.9559 \\%
$$

$$
f1: 91.9552 \\%
$$

|  | True | False |
| --- | --- | --- |
| Pred. True | 10654 | 1106 |
| Pred. False | 771 | 10803 |

<img width="450" src="https://github.com/scfengv/Mathematical-and-Statistical-foundation-of-Shrinkage-method/assets/123567363/3bb01c38-bb2a-4dc8-92e8-9dfd230e9779">
<img width="450" src="https://github.com/scfengv/Mathematical-and-Statistical-foundation-of-Shrinkage-method/assets/123567363/e3f9dcea-7ef1-4382-a3c2-95c559988b83">

Fig. 10-1 & Fig. 10-2

## Model Complexity

In the model complexity tests, the performance of Lasso was generally better than that of Stepwise, which is expected and aligns with the previous results. Both methods achieved their lowest errors when Log(C) was around -2 or -3. However, an interesting phenomenon observed in both cases is that $Train\ Error > Test\ Error$. Assuming there are no issues with the code (attached below), this might simply be because the testing data fits better. I have yet to determine other possible reasons, and I will provide updates if I discover any new insights.

<img width="450" src="https://github.com/scfengv/Mathematical-and-Statistical-foundation-of-Shrinkage-method/assets/123567363/fa00f11e-ad0e-47d3-ab02-3dc60095c547">
<img width="450" src="https://github.com/scfengv/Mathematical-and-Statistical-foundation-of-Shrinkage-method/assets/123567363/48f038a2-73c0-4496-a65c-9341498aa28d">


### Reference:

[1] Regularization in Machine Learning. [https://www.geeksforgeeks.org/regularization-in-machine-learning/](https://www.geeksforgeeks.org/regularization-in-machine-learning/)

[2] Shrinkage methods. [https://m0nads.wordpress.com/2018/01/15/shrinkage-methods-ridge-regression-and-lasso/](https://m0nads.wordpress.com/2018/01/15/shrinkage-methods-ridge-regression-and-lasso/)

[3]【機器學習】偏差與方差之權衡 Bias-Variance Tradeoff. [https://jason-chen-1992.weebly.com/home/-bias-variance-tradeoff](https://jason-chen-1992.weebly.com/home/-bias-variance-tradeoff)

[4] Intuitions on L1 and L2 Regularisation. [https://towardsdatascience.com/intuitions-on-l1-and-l2-regularisation-235f2db4c261#dda9](https://towardsdatascience.com/intuitions-on-l1-and-l2-regularisation-235f2db4c261#dda9)

[5] 正則化 (數學) [https://zh.wikipedia.org/zh-tw/正则化_(数学)](https://zh.wikipedia.org/zh-tw/%E6%AD%A3%E5%88%99%E5%8C%96_(%E6%95%B0%E5%AD%A6))

[6] 機器/深度學習-基礎數學(三):梯度最佳解相關算法(gradient descent optimization algorithms) [https://chih-sheng-huang821.medium.com/機器學習-基礎數學-三-梯度最佳解相關算法-gradient-descent-optimization-algorithms-b61ed1478bd7](https://chih-sheng-huang821.medium.com/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E5%9F%BA%E7%A4%8E%E6%95%B8%E5%AD%B8-%E4%B8%89-%E6%A2%AF%E5%BA%A6%E6%9C%80%E4%BD%B3%E8%A7%A3%E7%9B%B8%E9%97%9C%E7%AE%97%E6%B3%95-gradient-descent-optimization-algorithms-b61ed1478bd7)

[7] Gradient descent. [https://en.wikipedia.org/wiki/Gradient_descent](https://en.wikipedia.org/wiki/Gradient_descent) 

[8] Generalized Linear Models and Elastic Nets (GLMNET). [https://notebook.community/ceholden/glmnet-python/examples/glmnet_demo](https://notebook.community/ceholden/glmnet-python/examples/glmnet_demo) 

[9] What are the differences between Ridge regression using R's glmnet and Python's scikit-learn? ****[https://stats.stackexchange.com/questions/160096/what-are-the-differences-between-ridge-regression-using-rs-glmnet-and-pythons](https://stats.stackexchange.com/questions/160096/what-are-the-differences-between-ridge-regression-using-rs-glmnet-and-pythons)

[10] Bayesian Linear Regression : Data Science Concepts. [https://www.youtube.com/watch?v=Z6HGJMUakmc&ab_channel=ritvikmath](https://www.youtube.com/watch?v=Z6HGJMUakmc&ab_channel=ritvikmath)

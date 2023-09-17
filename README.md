# Mathematical and Statistical foundation of Shrinkage method

# Abstract

本文將探討 Shrinkage method 背後的數學和統計學基礎，並透過 Bayesian 和 Gradient descent 兩種觀點去了解 Lasso 和 Ridge 兩者根本上的差異。最後使用著名的人工資料 wave dataset 嘗試將原始資料的 121 個變數透過 Lasso regression 和 Stepwise selection 簡化到分別剩 14 和 9 個變數，並用 Logistic regression 進行分類比較，分類準確度分別高達 92.36% 和 91.96%。

# Introduction

本文使用的是一個著名的人工資料 wave dataset (Breiman et al., 1984)，在原始資料中原有 21 個 variables，在 2005 年時，Rakotomalala 在資料集中加入了 100 個 noise variables，使其成為了一筆 (33334, 121) 的資料。變數名稱 v1, v2, …, v21, alea1, alea2, …, alea100，並沒有什麼資訊可以從名稱中獲得，但可以大致推斷 v1, v2, …, v21 (以下簡稱 v 類變數) 為原始資料集中的 21 個變數，而 alea1, alea2, …, alea100 (以下簡稱為 a 類變數) 為 Rakotomalala 後來加入的 noise variables，故在開始做任何學習前對結果有兩個簡單的猜測 
1. 「v 類變數比較重要」
2. 「a 類變數對於模型沒有幫助或甚至會傷害模型」
所以可以期望的是在做 Variable selection 的結果中最好只包含 v 類變數而沒有 a 類變數。

另外，在拿到資料集時，資料大致依分類目標結果分為上下半部，故在做 K-Folds Cross Validation 時，需要做不同於以往簡單 K-Folds 的 `StratifiedKFold` 分層抽樣，此 CV 方法讓各 Fold 保有和母體一樣類別比例，可以確保各 Fold 不會因為初始 data 的排序等因素而有類別上的偏差。

```python
kf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
```

# Shrinkage method in Mathematic and Statistic

Shrinkage method 包含了 Lasso regression 和 Ridge regression 兩種，可以視為 Ordinary Least square 的修正。Ordinary Least square 存在著一個問題，OLS model 可以很好的擬合一個 Training data (Low Bias)，但會在未知的 data (Validation, Testing data) 表現較差，稱為 High Variance，原因多半是因為在擬合 Training set 的時候有 Overfitting 發生。為了修正這個問題，引入了 Shrinkage method，藉由一個和 model 和 data 無關的 $\lambda$  項 (penalty term) 來調節每個變數對於模型的權重，藉由「不要預測那麼準」(Increase Bias) 的方式來提升模型對於未知的資料的可預測性 (Reduce Variance)。這個 Variance 和 Bias 的消長過程稱為 Variance-Bias Trade-off，而最佳的平衡點 $\lambda$ 可以透過 Cross Validation 的方式找到。

![Fig. 1 [1]](Mathematical%20and%20Statistical%20foundation%20of%20Shrinka%2087dc5f74e4e44e2f9ec9bc3946e23e1a/Untitled.png)

Fig. 1 [1]

Fig. 2 即是一個很好的例子，藍線 ($\lambda = 0$) 代表 OLS，在右圖可以很清楚地看到這條線產生了 Overfitting，太過度被某些的資料點影響導致模型不太可能可以應用在普遍的情況，在引入 $\lambda$ 後，會在擬合時傾向對某變數產生很大權重的時候對其做「懲罰」，而降低權重以此避免 Overfitting。但這項 $\lambda$ 也不可以太大，如左圖的 $\lambda = 100$ (紅線)，回歸線變為一條幾乎是平坦的直線 ( $y = 1.7$ )，也就是說不管 $x_1$ 為多少，模型都猜 $y=1.7$，這是一個明顯的 Underfitting 的現象。而 $\lambda$ 的調節也可以透過做 Cross Validation 的方式去判斷。

![Fig. 2 [2]](Mathematical%20and%20Statistical%20foundation%20of%20Shrinka%2087dc5f74e4e44e2f9ec9bc3946e23e1a/Untitled%201.png)

Fig. 2 [2]

![Fig. 3 [3]](Mathematical%20and%20Statistical%20foundation%20of%20Shrinka%2087dc5f74e4e44e2f9ec9bc3946e23e1a/Untitled%202.png)

Fig. 3 [3]

以下會分別透過 Bayesian 和 Gradient Descent 的方式去嘗試說明 Shrinkage method 是如何透過 $\lambda$ penalty term 去對 OLS model 做修正

## Bayesian

$$
Ordinary\ Least\ square:y=X\beta+\varepsilon\\
---------------\\
Problem\ of\ OLS:\hat{\beta}_{OLS}\ have\ high\ variance\\
 Small\ \Delta x \ would\ lead\ to \ Large\ \Delta \beta \\
In\ machine\ learning: Bad\ prediction\ for\ Test\ data\\
---------------\\
Varianve-Bias\ \ Trade-off:\\
Regularization\\
Lasso:\hat{\beta}_{L1}:\arg \min_{\beta}[{\color{blue}{\|y-X\beta\|}_2\ ^2}+{\color{red}\lambda\|\beta\|_1}]\\

Ridge:\hat{\beta}_{L2}:\arg \min_{\beta}[{\color{blue}{\|y-X\beta\|}_2\ ^2}+{\color{red}\lambda\|\beta\|_2\ ^2}]\\--------------\\
\|\beta\|_1=\sum_{i=1}^N |\beta_i|\\
\ \ \ \ \ \ \ \ \ \ \|\beta\|_2 =\Big[\ \sum_{i=1}^N (\beta_i)^2\ \Big]^{\frac{1}{2}}\\
\ \ \ \ \ \ \ \ \ \ \|\beta\|_p =\Big[\ \sum_{i=1}^N (\beta_i)^p\ \Big]^{\frac{1}{p}}
$$

藍色的部分為 Error，即實際值 ($true\ y$) 和預測值 ($pred.\ y,\ X \beta$) 之間的差值 (MSE)，即為原本的 OLS model

紅色部分即為由 $\lambda$ 所驅使的 Regularization，也可稱為 Penalty term。$\lambda$ 的大小即為懲罰的強弱，可以使 $\beta$ 盡量的縮小，甚至在 Lasso regression 中可以將 $\beta =0$，即將此 variable 從 model 中移除

至於 Shrinkage method 是如何透過調控 $\lambda$ 的大小去做到降低 OLS 的 variance，可以透過 Bayesian 的觀點去理解

$$
Bayesian\ viewpoint\\
\hat{\beta}_{MAP}=\arg \max_{\beta}P(\beta|y)\\
\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ =\arg \max_{\beta}(\frac{P(y|\beta)*P(\beta)}{P(y)})\\
\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ =\arg \max_{\beta}({\color{blue}P(y|\beta)}*{\color{red}P(\beta)})\\
\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ =\arg \max_{\beta}[\ {\color{blue}log(P(y|\beta))}+{\color{red}log(P(\beta))}\ ]\\-----------------\\
{\color{blue}P(y|\beta)}:Likelihood\\{\color{red}P(\beta)}:Prior\\
$$

$\hat{\beta}_{MAP}$ 是最大後驗機率 (Maximum a posterior)，是指找出在給定資料 $y$ 下，出現機率最大的 $\beta$

藍色的部分為 Likelihood function，指在給定 $\beta$ 的情況下，觀察到特定 $y$ 的機率

紅色的部分為這裡的重點，稱為 Prior，指在沒有給定條件下，觀察到 $\beta$ 的機率，可以理解為 $\beta$ 的機率分佈或對 $\beta$ 的假設

$$
Assume\ y_i \sim N(\beta^Tx_i\ ,\  \sigma^2)\\\ \ \ \ \ \ \ \ \beta_j \sim N(0,\ \tau^2)\\-----------------\\\beta \ under\ a\ {\color{red}Gaussian\ prior}:\\
P(y|\beta)=\prod_{i=1}^N\ \frac{1}{\sigma\sqrt{2\pi}}\ exp(-{(y_i-\beta^Tx_i)^2\over 2\sigma^2})\\

Log(P(y|\beta))= \sum_{i=1}^N\ [\ log(\frac{1}{\sigma\sqrt{2\pi}})-{(y_i-\beta^Tx_i)^2\over 2\sigma^2}\ ]\\

\arg \min_{\beta}[\ \|y-X\beta\|_2\ ^2+\frac{\sigma^2}{\tau^2}\|\beta\|_2\ ^2\ ]\\\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ =\arg \min_{\beta}[\ \|y-X\beta\|_2\ ^2+{\color{red}\lambda}\|\beta\|_2\ ^2\ ]=\hat{\beta}_{L2}\rightarrow {\color{red}Ridge}\\
------------------\\
\beta \ under\ a\ {\color{red}Laplacian\ prior}:\\
P(y|\beta)=\prod_{i=1}^N\ \frac{1}{2b}\ exp(-\frac{|y_i-\beta^Tx_i|}{b})\\

Log(P(y|\beta))= \sum_{i=1}^N\ [\ log(\frac{1}{2b})-{(y_i-\beta^Tx_i)\over b}\ ]\\

\arg \min_{\beta}[\ \|y-X\beta\|_2\ ^2+\frac{\sigma^2}{\tau^2}\|\beta\|_1\ ]\\\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ =\arg \min_{\beta}[\ \|y-X\beta\|_2+{\color{red}\lambda}\|\beta\|_1\ ]=\hat{\beta}_{L1}\rightarrow {\color{red}Lasso}
$$

可以看到 Regularization 在透過引入對 $\beta$ 的假設後可以透過不同的機率分佈假設在 Bayesian 下得到和一開始的 Lasso 和 Ridge 一樣的結果
已知 lambda 表示為 Regularization 的強度，可以透過對 $\beta$ 的假設機率分佈知道，當 $\lambda$ 越大 ($\tau^2$ 越小)，則機率分佈應該會使 $\beta$ 更靠近 0 (Fig. 4, 綠 → 橘 → 藍)，而在不同的 Prior 下，Gaussian distribution (Ridge, Fig. 4-1) 是圍繞在 0 的周圍隨機分佈，而 Laplacian distribution (Lasso, Fig. 4-2) 則是指定大部分的係數為 0，進而達到 Feature Selection 的目的

![Fig. 4-1](Mathematical%20and%20Statistical%20foundation%20of%20Shrinka%2087dc5f74e4e44e2f9ec9bc3946e23e1a/newplot.png)

Fig. 4-1

![Fig. 4-2](Mathematical%20and%20Statistical%20foundation%20of%20Shrinka%2087dc5f74e4e44e2f9ec9bc3946e23e1a/newplot%201.png)

Fig. 4-2

另外，Lasso and Ridge 的這兩條 $\arg \min$ 數學式也可以表示為

$$
Lasso:\arg \min_{\beta}[{\|y-X\beta\|}_2\ ^2+\lambda\|\beta\|_1]\\

Ridge:\arg \min_{\beta}[
{\|y-X\beta\|}_2\ ^2+\lambda\|\beta\|_2\ ^2]
$$

$$
For\ every\ value\ of\ \lambda, there\ is\ a\ ''s''\ that\ satisfy:\\
Lasso:min\Big\{\ \sum_{i=1}^N (y_i-\beta_0-\sum_{j=1}^p\beta_jx_{ij})^2\ \Big\}\ subject\ to\ \sum_{j=1}^p |\beta_j|\leq s\\
Ridge:min\Big\{\ \sum_{i=1}^N (y_i-\beta_0-\sum_{j=1}^p\beta_jx_{ij})^2\ \Big\}\ subject\ to\ \sum_{j=1}^p \beta_j^2\leq s
$$

若分別將 Lasso 和 Ridge 的條件在 2D 下畫出來 (Fig. 5)，即可發現 Lasso 的 $\beta_j$ 是被限制在一個頂點在兩軸上的平面四邊形內 ( $|\beta_1|+|\beta_2| \leq s$ )，而 Ridge 的 $\beta_j$ 是被限制在一個圓中 ( $\beta_1^2+\beta_2^2 \leq s$ )， $\hat{\beta}$ 為 Least square 的解，紅色的線為等 RSS 線。如前面所述，導入 penalty term $\lambda$ 是一個透過不要預測的太準 (Increase Bias) 去交換一個對不同資料的適應性 (Reduce Variance) 的過程，在數學上，即為透過增加 RSS，來找到滿足邊界條件的第一個交點，此交點即為滿足條件下的最佳解。而因為邊界條件設定的關係，等 RSS 線多會交在 Lasso 四邊形的頂點 (條件設定的關係，頂點會在軸上) ，會造成不在該軸上的 $\beta_j=0$，即為前面所提到的 Feature Selection 的過程。至於多維度的情況大致也可以遵照這個想法去擴張。

![Fig. 5](Mathematical%20and%20Statistical%20foundation%20of%20Shrinka%2087dc5f74e4e44e2f9ec9bc3946e23e1a/IMG_CDC01411653B-1.jpeg)

Fig. 5

## Gradient Descent

另外，也可以通過 Gradient Descent 的方式來理解 Lasso & Ridge
這裡將 Linear Regression 透過 weight 和 bias 的方式表示，但大致和上面的表示方法大同小異

$$
Linear\ Regression:\hat{y}=w_1x_1+w_2x_2+...+w_Nx_N+b\\---------------\\
w:weight,\ b:bias
$$

$$
Loss\ Function\ of \ each\ Regularization:\\No\ Regularization\\
L=(\hat{y}-y)^2\\\ \ \ \ \ \ \ \ \ \ \ \ \ =(wx+b-y)^2\\\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ =Error\ between\ \hat{y}\ (pred.\ value)\ and\ y\ (true\ value)
$$

$$
L1 \ regularization\\
L_1=(wx+b-y)^2+\lambda|w|
$$

$$
L2 \ regularization\\
L_2=(wx+b-y)^2+\lambda w^2
$$

引入 Gradient Descent 並分別以三種情境帶入以觀察 $w$ 的變化

$$
w_{new}=w-\gamma \frac{\partial L}{\partial w}\\---------------\\
\gamma:Learning\ rate\\---------------\\Assume:\\
Learning\ rate\ (\gamma)=1\\
H=2x(wx+b-y)

$$

$$
No\ Regularization:\\
w_{new}=w-\gamma\ [\ 2x(wx+b-y)\ ]\\=w-H\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ 

$$

$$
L1\ Regularization:\\
w_{new}=w-\gamma\ [\ 2x(wx+b-y)+\lambda \frac{d|w|}{w}\ ]
\\\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ =w-\gamma\ [\ 2x(wx+b-y)+\lambda\ ] \ \ \ (w>0)
\\\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ =w-\gamma\ [\ 2x(wx+b-y)-\lambda\ ] \ \ \ (w<0)\\=(w-H)-\lambda \ \ \ (w>0)\ \ \ \ \ \ \ \ \ \ \\=(w-H)+\lambda \ \ \ (w<0)\ \ \ \ \ \ \ \ \ \ 

$$

$$
L2\ Regularization:\\
w_{new}=w-\gamma\ [\ 2x(wx+b-y)+2\lambda w\ ]\\=(w-H)-2\lambda w \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ 

$$

在這裡可以先假設 Linear Regression 完美 fit train data，但這也表示這個 model 對於陌生 data (validation, test data) 等的表現會較差，即上面所提到的 High variance
(Overfitting)，所以在對 Linear Regression 引入 Regularization，可以對原始的 model 要 overfit 的時候做到懲罰，因為 $\lambda$ 是一個獨立於 model & data 的常數。從 L1 Regularization 的 $w_{new}$ 式中看到，當 $w>0$ 時，$-\lambda$ 會使 $w$ 不要那麼大，$w<0$ 時同理。而在 L2 Regularization 中也做了差不多的事。

所以綜合 Bayesian 和 Gradient descent 的結果，當 $\lambda$ 越大時，在 Bayesian 的結果中，$\beta_j$ distribution (both Lasso & Ridge) 會更集中在 0 左右，而在 Gradient descent的結果中，$|w_{new}|$ 會越小，簡單來說即為懲罰力道越強，可以使 $w$ 降低越多，進而達到避免 Overfitting 的結果，但也要注意 $\lambda$ 並非越小越好，而是在決定 $\lambda$ 的過程中會有 Variance-Bias Trade-off 的行為，所以需要透過 Cross Validation 來決定適合每個 model 的最佳 $\lambda$ 值。

# Lasso Feature Selection

本文所使用的是 `sklearn.linear_model` 模組裡面的 `lasso` (glmnet.lasso 和 sklearn.lasso 大致相同，兩者的差別可以參考這兩篇文章: [https://notebook.community/ceholden/glmnet-python/examples/glmnet_demo](https://notebook.community/ceholden/glmnet-python/examples/glmnet_demo), [https://stats.stackexchange.com/questions/160096/what-are-the-differences-between-ridge-regression-using-rs-glmnet-and-pythons](https://stats.stackexchange.com/questions/160096/what-are-the-differences-between-ridge-regression-using-rs-glmnet-and-pythons))，要注意的是在 `sklearn` 中 $\lambda$ 被稱為 `alpha` ，但不會影響太多，以下以 $\lambda$ 表示。

在理解了理論面後就可以實際應用 Lasso Regression 來做到 Feature Selection。觀察下面這張 Lasso Penalty vs. Coefficients 的圖 (Fig. 6-1)，可以看到隨著 $\lambda$ 值的提升，確實對 Coefficients 起到收斂的效果，而在 $\lambda > 10^{-0.5}$ 後所有係數都收縮到 0，也就是指模型中已經不含有任何變數，即為 Fig. 2 中左圖的狀況，模型已經變成一個常數值。可以從 Cross Validation 的結果中看到，表現最好的 $\lambda=10^{-3}$，選擇的變數為 21 個 v 類變數中的 13 個，在變數選擇上我有做了一些更動，由下圖可以發現，在 $\lambda = 10^{-3}$ 時，很多變數都還沒收斂到 0，故此時若挑選變數的門檻令為 Lasso Feature Selection 定義的 coef. > 0 的話，會將太多變數放入，故在 $\lambda = 10^{-3}$ 時，將門檻設為 coef. > 0.01。但在已經收斂的 $\lambda = 10^{-2}$ (但非 Cross Validation 選出的最高分參數)，即可將挑選變數門檻令為 coef. > 0，得出的結果差不多，但在 $\lambda = 10^{-2}$ 時被挑選的變數會多一個 `‘v20’`。

![Fig. 6-1](Mathematical%20and%20Statistical%20foundation%20of%20Shrinka%2087dc5f74e4e44e2f9ec9bc3946e23e1a/newplot%202.png)

Fig. 6-1

![Fig. 6-2](Mathematical%20and%20Statistical%20foundation%20of%20Shrinka%2087dc5f74e4e44e2f9ec9bc3946e23e1a/newplot%203.png)

Fig. 6-2

![Fig. 6-3](Mathematical%20and%20Statistical%20foundation%20of%20Shrinka%2087dc5f74e4e44e2f9ec9bc3946e23e1a/newplot%204.png)

Fig. 6-3

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

若將 Lasso 所選的變數和 Stepwise selection 做比較可以看到些微的不同。兩者都只有挑了 v 類變數，但從 Stepwise Selection 的 Performance 中可以看到，Accuracy 大致在 9 個變數以後就幾乎沒有再明顯增加。

```python
# Stepwise selection

Selected Feature: ['v9', 'v10', 'v11', 'v12', 'v13', 'v15', 'v16', 'v17', 'v18']
```

![Fig. 7](Mathematical%20and%20Statistical%20foundation%20of%20Shrinka%2087dc5f74e4e44e2f9ec9bc3946e23e1a/newplot%205.png)

Fig. 7

但在挑選變數的過程中有一個非常非常有趣的現象，翻開 Stepwise 變數選擇的過程，可以看到在我先前的假設中，我認為 v 類變數是重要的，a 類變數只是 noise variables。但在 Stepwise selection 的選擇過程中，卻有一個 a 類變數 `‘alea81’` 是第 10 個被選中的，比剩下 12 個 v 類變數還要早進到 Feature group，第 11 個被選中的變數為 `‘v4’` ，而下一個被選中的 v 類變數 `‘v8’` 為第13 個。依照我的假設，前 21 個被選入的變數應該都要是 v 類變數，但在第 13 個 `‘v8’` 後到第 30 個之間都再沒有 v 類變數被選中。

```python
# Stepwise selection

sfs.get_metric_dict()[9]
feature_names: ['v9','v10','v11','v12','v13','v15','v16','v17','v18']

sfs.get_metric_dict()[10]
feature_names: ['v9','v10','v11','v12','v13','v15','v16','v17','v18',
								'alea81']

sfs.get_metric_dict()[11]
feature_names: ['v4','v9','v10','v11','v12','v13','v15','v16','v17','v18',
								'alea81']

sfs.get_metric_dict()[21]
feature_names: ['v4','v8','v9','v10','v11','v12','v13','v15','v16','v17','v18',
								'alea81','alea7','alea20','alea24','alea26','alea29','alea32','alea48','alea71','alea81','alea85']

sfs.get_metric_dict()[30]
feature_names: ['v4','v8','v9','v10','v11','v12','v13','v15','v16','v17','v18',
								'alea2','alea7','alea9','alea18','alea24','alea26','alea29','alea32','alea35','alea42','alea48','alea51','alea58','alea71',
								'alea81','alea85''alea95','alea98','alea100']
```

而在 Lasso 有沒有發生類似的狀況呢？答案是有的，還記得我剛剛提到的  $\lambda$ 在 $10^{-3}$ 和 $10^{-2}$ 那邊的問題，之所以後來選擇 $\lambda=10^{-2}$ 的原因是因為在  $\lambda=10^{-3}$ 時大部分的 a 類變數係數都還沒有收斂，但如果看得更仔細一點會發現，在 $\lambda=10^{-3}$ 時，`’v1’, ‘v7’` 兩個 v 類變數已經收斂到 0 了，若將 coef. 門檻上調到 0.05 ( 沒有大太的實質意義 )，可以看到 a 類變數只剩 `‘alea70’` ，說明其實在  $\lambda=10^{-3}$ 時大部分的 a 類變數已經準備要收斂了，但可以發現 v 類變數中的 `‘v2’` 也消失了，而在 $\lambda=10^{-2}$ 時，`’v3’, ‘v4’, ‘v6’, ‘v14’, ‘v21’` 也都消失了。

綜合以上兩段關於 Feature selection 時的發現可以看到，在開頭時的第一個猜想「v 類變數比較重要」大部分時間都是對的，但會在小部分時候失效，可能是門檻設定過於嚴苛，也有可能是 Rakotomalala 在加入這個雜訊的時候是有經過設計使加入的變數不會對結果有太重要的影響卻也不至於偏差太多。因為從 Fig. 7 中可以看到，其實在開始有 a 類變數加入後的 10 個變數以後的 model performance 並沒有明顯的上升，甚至在第 13 個 `‘v8’` 加入後，後面的表現幾乎持平，表示 a 類變數對於模型其實沒有什麼作用 (沒有幫助也沒有傷害)，因次可以對比開頭所下的猜想得出結論
1.「v 類變數比較重要」
2.「a 類變數對於模型沒有幫助或甚至會傷害模型」
兩個想法都大致上是對的，但也都有特例的發生，至於在 a 類變數會傷害模型這點會留到後面進入分類學習的階段再做討論。

```python
# Lasso Selection

## alpha = 0.001
print(importance > 0)

feature_names: ['v2' 'v3' 'v4' 'v5' 'v6' 'v8' 'v9' 'v10' 'v11' 'v12' 'v13' 'v14' 'v15' 'v16' 'v17' 'v18' 'v19' 'v20' 'v21' 
								'alea1'  'alea3'  'alea5'  'alea6'  'alea7'   'alea8' 'alea10' 'alea11' 'alea12' 'alea13' 'alea14' 'alea15'
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

在分類的部分我嘗試了五組的 Feature Group，分別是 without feature selection / lasso feature selection / stepwise feature selection / all ‘v’ / all ‘a’。因為是均衡的資料集，故可以以 Accuracy 來做模型比較，可以看到在有做變數選擇的組別中，運用 Lasso 做 Feature selection 的成效是最好的，再來是全選，最後才是 Stepwise selection，但並沒有差到太多。另外可以比較 w/o Feature selection 和 Lasso 這兩組可以看到在 Accuracy 在 w/o Feature selection 的情況下有下降，但其實下降的量有限，由前面的結果可以大致認定這兩組 Feature group 之間的差幾乎都是 a 類變數，所以回歸到開頭的假設 2.「a 類變數對於模型沒有幫助或甚至會傷害模型」這點，可以說 a 類變數的存在會增加模型的 noise，會對模型造成一定的影響，但其實效果有限，因此可以認定 a 類變數即為純粹的雜訊，並不會對模型起到非常大的傷害，但仍需要被刪除以降低模型的複雜度。另外也可以看到，`’v17’` 在三組裡面的變數重要性都是排第一。最後在比較全部的 v 類變數和全部的 a 類變數會發現 all ‘v’ 的表現又優於 Lasso，這實際上也是可以預期的，只要在不傷害模型 (常有有極端值) 的情況下，增加模型中的變數多少會使他擬合得更好，但 Feature selection 的目的就在於降低模型的複雜度，在減少變數的同時避免掉發生 Overfitting 的可能性。

|  | w/o Feature selection | Lasso | Stepwise | all ‘v’ | all ‘a’ |
| --- | --- | --- | --- | --- | --- |
| Accuracy | 92.14% | 92.36% | 91.96% | 92.43% | 50.35% |
| AUC | 0.98 | 0.98 | 0.98 | 0.98 | 0.50 |
| Most important variable | v17 | v17 | v17 | v17 | alea73 |

## Without Feature Selection

$$
Accuracy:92.1359 \%\\
Precision: 92.1531 \%\\
Recall: 92.1359 \%\\
f1: 92.1358 \%\\
$$

|  | True | False |
| --- | --- | --- |
| Pred. True | 10729 | 1031 |
| Pred. False | 804 | 10770 |

![Fig. 8-1](Mathematical%20and%20Statistical%20foundation%20of%20Shrinka%2087dc5f74e4e44e2f9ec9bc3946e23e1a/Untitled%203.png)

Fig. 8-1

![Fig. 8-2](Mathematical%20and%20Statistical%20foundation%20of%20Shrinka%2087dc5f74e4e44e2f9ec9bc3946e23e1a/newplot%206.png)

Fig. 8-2

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
Accuracy: 92.3631 \%\\
Precision: 92.3826 \%\\
Recall: 92.3631 \%\\
f1: 92.3629 \%\\
$$

|  | True | False |
| --- | --- | --- |
| Pred. True | 10748 | 1012 |
| Pred. False | 770 | 10804 |

![Fig. 9-1](Mathematical%20and%20Statistical%20foundation%20of%20Shrinka%2087dc5f74e4e44e2f9ec9bc3946e23e1a/Untitled%204.png)

Fig. 9-1

![Fig. 9-2](Mathematical%20and%20Statistical%20foundation%20of%20Shrinka%2087dc5f74e4e44e2f9ec9bc3946e23e1a/newplot%207.png)

Fig. 9-2

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
Accuracy: 91.9559 \%\\
Precision: 91.9924 \%\\
Recall: 91.9559 \%\\
f1: 91.9552 \%\\
$$

|  | True | False |
| --- | --- | --- |
| Pred. True | 10654 | 1106 |
| Pred. False | 771 | 10803 |

![Fig. 10-1](Mathematical%20and%20Statistical%20foundation%20of%20Shrinka%2087dc5f74e4e44e2f9ec9bc3946e23e1a/Untitled%205.png)

Fig. 10-1

![Fig. 10-2](Mathematical%20and%20Statistical%20foundation%20of%20Shrinka%2087dc5f74e4e44e2f9ec9bc3946e23e1a/newplot%208.png)

Fig. 10-2

## Model Complexity

在模型複雜度測試中，Lasso 的表現大致比 Stepwise 要好，這點是可預期的，吻合上面的結果。兩組均在 Log(C) = -2, -3 左右時分別有最低的 Error。但兩者均呈現出一個現象就是 $Train\ Error > Test\ Error$，在撇除 code 有問題的情況下 (附在下面)，可能就是剛好 testing data 比較好 fit，至於其他原因目前還沒想到，若有更新會隨後補上。

![Fig. 11-1](Mathematical%20and%20Statistical%20foundation%20of%20Shrinka%2087dc5f74e4e44e2f9ec9bc3946e23e1a/newplot%209.png)

Fig. 11-1

![Fig. 11-2](Mathematical%20and%20Statistical%20foundation%20of%20Shrinka%2087dc5f74e4e44e2f9ec9bc3946e23e1a/newplot%2010.png)

Fig. 11-2

```python
C_range = [ 10**i for i in range(-4, 5) ]

train_errors = []
test_errors = []

for C in C_range:
    clf = LogisticRegression(
        max_iter = int(1e7), C = C, penalty = 'l2', solver = 'lbfgs', random_state = 42
    )
    clf.fit(x_train_scl, y_train)
    train_errors.append(1 - clf.score(x_train_scl, y_train))
    test_errors.append(1 - clf.score(x_test_scl, y_test))

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x = [ np.log10(C) for C in C_range ], y = train_errors,
        mode = 'lines+markers',
        name = 'Train Error'
    )
)

fig.add_trace(
    go.Scatter(
        x = [ np.log10(C) for C in C_range ], y = test_errors,
        mode = 'lines+markers',
        name = 'Test Error'
    )
)
fig.update_layout(
    title = {
        'text': 'Model Complexity',
        'font': {
            'size': 40, 'family': 'Gulliver'
        },
        'x': 0.5
    },
    width = 800, height = 600,
    xaxis_title = {
        'text': 'Log(C)',
        'font': {
            'size': 24, 'family': 'Gulliver'
        }
    },
    yaxis_title = {
        'text': 'Classification Error',
        'font': {
            'size': 24, 'family': 'Gulliver'
        }
    }
)

fig.update_yaxes(
    range = [0.075, 0.095]
)

fig.show()
```

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

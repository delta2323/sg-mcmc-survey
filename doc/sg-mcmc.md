$\newcommand{\b}[1]{\boldsymbol{#1}}$

# 最近のMCMCについて

## 目次

* MCMC基礎
  * Metropolis-Hasting
  * 今回話さない話
    * レプリカ交換法（parallel tempering）
    * Importance Sampling
* 問題設定
  * 今回は値域が連続の場合を考える
  * 事後分布からサンプリングする問題に限定
* 必要な道具
  * 確率的にする方法
  * SDEとその離散化の方法
* 各手法の詳細説明
  * HMC
    * 解析力学からの事前知識
    * Leapfrog法
    * Hamiltonianが保存している->提案を棄却する必要がない
  * SGLD
    * 1st orderのLangevin Dynamicsを離散化したもの（Brownian Dynamicsとも呼ばれたりする）
    * SDEを考えているの確率的にすることによるノイズが無視できないという話
  * SGHMC
    * 2nd orderのLangevin Dynamicsを離散化したもの
    * HMC, SGLDとの関係
      * HMCを確率的にし、確率的にしたことによるノイズを入れて、摩擦の項を入れるとSGHMC
      * SGHMCでのLangevin Dynamicsを質量を0にした極限がSGLDのLangevin Dynamics
  * (m)SGNHT
    * Thermostatに対応する変数を導入してエネルギーをコントロールする
    * SGNHTとmSGNHTの関係
    * SSIによる近似精度の向上
      * 本質的にはBaker–Campbell–Hausdorff formulaから来る
      * Leapfrog法と少し似ている
  * Santa
    * 一言で言うと、RMSprop + mSGNHT
      * mSGNHTにRMSpropのPreconditionerを入れる
      * さらにSimulated Annealingすることで、逆温度が低いときは事後分布からのサンプリングで、その逆温度が無限大の極限では事後分布のmodeを探索するようになる
    * SSIによる近似精度向上もできる
* 実験
  * 実験設定
  * 実験結果

## Notation

* 細字はスカラー、太字はベクトルを表す
* ベクトルは列ベクトル
* $\b{\theta}$ or $\b{q}$: モデルのパラメータ
* $\b{p}$: 運動量に対応するパラメータ
* $\b{\xi}$: サーモスタットに対応するパラメータ
* $U(\b{\theta})$, $U(\b{q}$): 位置エネルギー
* $K(\b{p})$: 運動エネルギー
* $H(\b{\theta}, \b{p})$, $H(\b{\theta}, \b{p})$, $H(\b{\theta}, \b{p}, \b{\xi})$: ハミルトニアン
* $\mathcal{N}(\b{\mu}, \Sigma)$: 平均$\b{\mu}$, 分散$\Sigma$のガウス分布
* $\b{x}$: 訓練データ
* $X = \{ \b{x}_1, \ldots, \b{x}_N \}$: 訓練データセット
* $t = 1, \ldots, T$: 時刻（ 離散・連続時刻両方で$t$ を使う）
* $\beta$, $\beta_t$: 逆温度
* $d\b W$: ワイナー過程
* $\b a \odot \b b$: $\b a$と$\b b$のアダマール積（要素ごとの掛け算）
* $\b 0$, $\b 1$: 全要素が0, 1のベクトル
* $I$: 単位行列
* $h, h_t$: ステップ幅
* $\hat{a}$: $a$の推定値（推定値にはハットをつける）

## 略語

* HMC: Hamiltonian Monte Carlo, もしくはHybrid Monte Carlo
* SGLD: Stochastic Gradient Langevin Dynamics
* SGHMC: Stochastic Gradient HMC
* SGNHT: Stochastic Gradient Nos\'e -Hoover Thermostat
* mSGNHT: multivariate SGNHT
* Santa: Stochastic AnNealing
Thermostats with Adaptive momentum


## PDE, SDE

### HMC

$$
\begin{align}
\frac{d\b\theta}{dt} &= \b p\\
\frac{d\b p}{dt} &= -\nabla_{\b\theta}U(\b\theta)
\end{align}
$$
もしくは、
$$
\begin{align}
d\b\theta &= \b pdt\\
d\b p &= -\nabla_{\b\theta}U(\b\theta) dt
\end{align}
$$


### SGLD

SGLDは1次のLangevin Dynamicsを離散化したものである。

$$
\begin{align}
d\b\theta =  \nabla_{\b \theta} U(\b\theta) dt + \sqrt{2} d\b W\end{align}
$$

### SGHMC

SGHMCは

$$
\begin{align}
d\b\theta &= \b p dt \\
d\b p &= \nabla_{\b \theta} U(\b\theta)dt - A\b pdt + \sqrt{2A}d \b W
\end{align}
$$

$t' = At$とリスケーリングして、$A\to \infty$の極限を取ると、SGLDに帰着する


### SGNHT, mSGNHT

* SGNHT

$$
\begin{align}
d\b\theta &= \b p dt \\
d\b p &= \left(\nabla_{\b \theta} U(\b \theta) - \xi \b p\right) dt + \sqrt{2A} d\b W\\
d\xi &= \left( \frac{1}{n} \b p^T \b p - 1\right) dt.
\end{align}
$$

* mSGNHT

$$
\begin{align}
d\b\theta &= \b p dt \\
d\b p &= \left(\nabla_{\b \theta} U(\b \theta) - \b\xi \b p\right) dt + \sqrt{2A} d\b W\\
d\b\xi &= \left( \b p \odot \b p - \b 1\right) dt.
\end{align}
$$

### Santa

$$
\begin{align}
d\b\theta &= G_1(\b\theta) \b pdt\\
d\b p &= \left(\nabla_{\b \theta} U(\b \theta) - \b \xi \b p\right) dt + \b F(\b\theta, \b\xi) dt + \left(\frac{2}{\beta} G_2(\b\theta)\right) d\b W\\
d\b\xi &= \left(\b p\odot \b p - \frac{\b 1}{\beta}\right) dt
\end{align}
$$
ここで、$\b F(\b\theta, \b\xi) = \frac{1}{\beta} \nabla_{\b \theta} G_1(\b\theta) + G_1(\b\theta)\left(\b\xi - G_2(\b\theta)\right) \nabla_{\b \theta} G_2(\b\theta).$

## アルゴリズム

### HMC

$$
\begin{align}
& \text{Initialize $\b \theta$}\\
& \text{For $i = 1$ to $\infty$}\\
& \qquad \b p \sim N(\b 0, I)\\
& \qquad \b p \leftarrow \b p - \frac{h}{2} \widehat{\nabla_{\b\theta}U}(\b\theta)\\
& \qquad \text{For $l = 1$ to $L$}\\
& \qquad \qquad \b\theta \leftarrow \b\theta + h \b p\\
& \qquad \qquad \text{If $l \not = L$}\\
& \qquad \qquad \qquad \b p \leftarrow \b p - h \widehat{\nabla_{\b\theta}U}(\b\theta)\\
& \qquad \b p \leftarrow \b p - \frac{h}{2} \widehat{\nabla_{\b\theta}U}(\b\theta)\\
& \qquad \text{Accept $\b\theta$}
\end{align}
$$

疑似コード
```python
def update(p, q, x):
    def update_p(p, q, x):
        d_q = estimate_grad(q, x)
        return p + d_q * args.eps / 2

    def update_q(q, p):
        return q + p * args.eps

    for l in six.moves.range(L):
        p = update_p(p, q, x)
        q = update_q(q, p)
        p = update_p(p, q, x)
    return p, q


theta = model.sample_from_prior()
for epoch in six.moves.range(EPOCH):
    p = numpy.random.randn(*theta.shape)
    for i in six.moves.range(0, args.N, args.batchsize):
        x = get_minibatch()
        p, theta = update(p, theta, x)
```

### SGLD


### SGHMC

### mSGNHT

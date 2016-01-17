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
      * さらにSimulated Annealingすることで、逆温度が低いときは事後分布からの
      サンプリングで、その逆温度が無限大の極限では事後分布のmodeを探索するようになる
    * SSIによる近似精度向上もできる
* 実験
  * 実験設定
  * 実験結果

## Notation

* 細字はスカラーもしくは行列、太字はベクトルを表す
* ベクトルは列ベクトル
* $\b{\theta}$ or $\b{q}$: モデルのパラメータ
* $\b{p}$: 運動量に対応するパラメータ
* $\b{\xi}$: サーモスタットに対応するパラメータ
* $U(\b{\theta})$, $U(\b{q}$): 位置エネルギー
* $K(\b{p})$: 運動エネルギー
* $H(\b{\theta}, \b{p})$, $H(\b{\theta}, \b{p})$,
$H(\b{\theta}, \b{p}, \b{\xi})$: ハミルトニアン
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
* $x_i \sim p(x | \theta)$:  $x_i$ を $p(x|\theta)$ からサンプリングする
* $|X|$: 集合$X$の要素数
* $A : B$: 行列 $A, B$ の要素ごとの積の和 $\mathrm{tr} (A^TB)$ (double dot productというらしい)
* $\mathrm{diag} (\b v)$: ベクトル$\b v$の成分を対角線上に並べた行列

## 略語

* HMC: Hamiltonian Monte Carlo, もしくはHybrid Monte Carlo
* SGLD: Stochastic Gradient Langevin Dynamics
* SGHMC: Stochastic Gradient HMC
* SGNHT: Stochastic Gradient Nos\'e -Hoover Thermostat
* mSGNHT: multivariate SGNHT
* Santa: Stochastic AnNealing
Thermostats with Adaptive momentum

## 問題設定

訓練データ $X=\{\b x_1, \ldots, \b x_N\}$ が与えられている。
これらは確率モデルから生成されたものとする: $X\sim p(X|\b\theta)$。
ここで、 $\b\theta$ は モデルのパラメータである。
$p(X|\b\theta)$ をモデル化する方法は問題によって色々と考えられるが、
今回の話では、その対数尤度 $\log p(X|\b\theta)$ とその微分
$\nabla_{\b\theta} \log p(X|\b\theta)$ が計算か、少なくとも推定ができれば良い。
例えば、 $p(X|\b\theta)$ はトピックモデルかもしれないし、ニューラルネットかもしれない。
この対数尤度の勾配の計算については後で詳しく考える。

この状況でパラメータの事後分布 $p(\b\theta|X)$ を推定したい。ベイズの定理より、
$$
p(\b\theta|X) = \frac{p(X|\b\theta)p(\b\theta)}{p(X)} =
\frac{p(X|\b\theta)p(\b\theta)}{\int p(X, \b\theta)d\b\theta}
$$
が成り立つことがわかる。パラメータの事前分布 $p(\b\theta)$
は自分で好きなように設定すれば良いとして、
問題は分母の同時分布が積分が伴うために解析的には計算できないことである。もちろん、
$$
\begin{align}
\int p(X, \b\theta)d\b\theta = \int p(X|\b\theta) p(\b\theta) d\b\theta
\approx \sum_{i=1}^{I} p(X|\b\theta_i)\quad \b\theta_i \sim p(\b\theta)
\end{align}
$$
とすれば推定値が得られるが、パラメータが高次元になるとこの方法による多重積分の推定は困難になる。

（ここまでテンプレ）

この困難を解消する方法としては、事後分布の推定の方法としては、少なくとも2つの方法がある
* サンプリング：事後分布 $p(\b\theta |X)$ からサンプリングをして、
その経験分布で事後分布を近似する
* 変分ベイズ：計算ができる確率分布族 $q(\b\varphi)$ を用意し、
$p(X|\b\theta)$ を近似する $\b\varphi$ を見つける

今回は前者のサンプリングの方法について解説を行う。
それについては以降の章で詳細を述べることにし、後者について補足のコメントを行う。
後者の方法としては平均場近似、確率伝搬法、Variational AutoEncoder(VAE)などが知られている。
VAEは、 $p(X|\b\theta)$, $q(\b\varphi)$ としてニューラルネットを用いて、
それらを誤差逆伝播法とReparametrizationTrickを用いて同時に最適化する手法であり、
近年よく研究されている。
例えばVAEを用いて大きな画像を生成したり、RNNと組み合わせて、
可変長データの生成などに応用されている。
また、これら2つは相反する物ではなく、実際両者を組み合わせて、
分布族としてより複雑な確率分布族を用いて事後分布を近似する方法が提案されている。

## パラメータ空間上での運動

今回考えるMCMCサンプリングは、どれもパラメータ空間上での力学系に従う
質点の運動の離散化として得られる。
つまり、パラメータ空間上に質点があって、
その質点はある運動方程式に従って運動をしている状況を考える。
その運動は古典力学のように決定的（つまり初期状態を決めたらその後の運動は一意に決まってしまう）
かもしれないし、統計力学のように確率的かもしれない。
前者の場合には運動方程式は偏微分方程式（Partial Differential Equation, PDE）で書かれ、
後者の場合には確率微分方程式（Stochastic Differential Equation, SDE）で書かれる。
サンプリング方法の違いは、この運動方程式の違いに由来する。

概ねどの方法も確率分布 $p(\b\varphi)$ からのサンプリングは以下のように行う。

* パラメータ空間上に、$p(\b\varphi)$ から決まる運動方程式を1個固定する
* 質点を適当な初期状態にセットする
* 質点を運動方程式に従って運動させる
* 十分に時間が経つと、質点の位置の確率分布は $p(\b\varphi)$ に収束する
* 質点の位置を適当な時間間隔でサンプリングする

## Fokker-Planck方程式

$n$ 次元の確率過程 $\b \varphi_t$ がSDE

$$
\begin{align}
d\b\varphi_t = \b\mu(\b\varphi_t, t)dt + \mathcal N(\b0, 2D(\b\varphi_t, t)dt)
\end{align}
$$

に従うとする。ここで、$\b\mu$ は $n$ 次元のベクトル値の決定的な関数, $D$ は $n \times n$ の行列値の決定的な関数で、 $N(\b0, D(\b\varphi_t, t)dt)$ は $N$ 次元のワイナー過程である。
この時、時刻 $t$ での $\b\varphi_t$ の確率分布 $p_t(\b\varphi)$はPDE

$$
\frac{\partial}{\partial t}p_t(\b\varphi) =
-\nabla_{\b\varphi}\left[ \b\mu(\b\varphi, t)p(\b\varphi, t)\right]
+ \nabla_{\b\varphi}\nabla_{\b\varphi}^T:
\left[D(\b\varphi, t) p_t(\b\varphi)\right]
$$

に従う。

分布が定常状態に収束しているというのは、左辺が$0$であることを意味する。
今回の場合は、SDEに従って動くのはパラメータ空間上の質点なので、
$X_t$ に対応するのはパラメータ $\b\varphi_t$ である
（時刻に従って変化することを強調するために $t$ を添字につけた、
今後、SDEやPDEを考える時にはパラメータ $\b\varphi$, $\b\theta$ などは時刻
$t$ の関数となっていることに注意）。

## カノニカル分布とは？

確率分布をパラメータ空間の質点の運動を用いて確率分布からサンプリングを行うためには、
質点の運動に関わる物理量とサンプリングしたい確率分布を関連付ける関係性が必要となる。
それを与えてくれるのがカノニカル分布である。

カノニカル分布は元々は熱力学の概念で、平衡状態の系で系がある状態を取る確率を、
その状態のエネルギーを用いて表したものである。
端的に言えば「エネルギーの高い状態になる確率は低く、低い状態になる確率は高い」という分布である。

$\Phi$を系が取りうるパラメータ集合として、系は $\Phi$ の各要素
$\b \varphi$ によって特徴づけられるとしよう。例えば以下のようなものがある：

* $N$ 個の自由粒子からなる気体の場合、
$\b \varphi = (\b q_1, \ldots, \b q_N, \b p_1, \ldots, \b p_N)$、
ここで、 $\b q_n, \b p_n$  は $n$ 番目の粒子の位置と運動量
* イジングモデルで各ユニットの向き：
$\b \varphi=(\sigma_{1}, \ldots, \sigma_{N})$、ここで $\sigma_i \in \{-1, 1\}$
* RBFの結合重み $W$ とバイアス $\b b_v$, $\b b_h$ :
$\b \varphi = (W, \b b_v, \b b_h)$

系がパラメータ $\b \varphi$ の状態を取る確率は $\exp(-H (\b \varphi))$
に比例する（ $H(\b \varphi)$ : 系のエネルギー）。
これは、エネルギーの高い状態にはなりにくく、低い状態はなりやすいという物理的な直観にもあっている。

$$
p(\b \varphi) \propto \exp(-H(\b \varphi))
$$

熱力学の場合にはエネルギーが先にあり、そのエネルギーに従って確率分布が決まるということが多いが、
今回の場合はデータの確率分布の方が先に決まっていて、
それを用いて系の持つエネルギーを定義すると考える方が自然であることが多い。

## モデルの拡張

今回はパラメータとして、本来のモデルパラメータ $\b \theta$の他に、
運動量に対応するパラメータ $\b p$を用意する：$\b\varphi = (\b\theta, \b p)$。
確率モデルが本来のパラメータとは別にパラメータを持つようになったので、
このパラメータの確率分布もモデル化しよう。

系のエネルギーは $\theta$ から決まるポテンシャルエネルギー
$U(\b \theta)$ と、$\b p$ から決まる運動エネルギーからなる運動エネルギー
$K(\b p)$からなる：

$$
\begin{align}
H(\b \theta, \b p) = U(\b \theta) + K(\b p)\\
\end{align}
$$

ここで運動エネルギーは、通常の物理での系と同じように、

$$
\begin{align}
K(\b p) = \frac{1}{2}\b p^T \b p
\end{align}
$$

と定める。質量に対応するハイパーパラメータ$M$を用いて
$K(\b p) = \b p^T M^{-1}\b p/2$とすることもあるが、今回は単純のため、$M = I$ とする。

$U(\b \theta)$ がどういう表式かは、具体的に書き下すことはせず、
カノニカル分布を用いて、データ分布 $p(X | \b \theta)$ と紐づけることにする。
パラメータ$\b \theta, \b p$ の同時分布は、

$$
p(\b \theta, \b p | X) \propto \exp(-H(\b \theta, \b p))
$$

を満たす。正規化定数を明示的に書けば、

$$
p(\b \theta, \b p | X) = \frac{1}{Z} \exp(-H(\b \theta, \b p))
$$

となる。ここで、 $Z = \int \exp(-H(\b \theta, \b p)) d\b\theta d\b p$は正規化定数。
ハミルトニアンが分解できることから、これは$Z$はさらに計算を進めることができる。

$$
\begin{align}
Z &= \int \exp(-H(\b \theta, \b p)) d\b\theta d\b p\\
&= \int \exp(-U(\b \theta) \exp(-K(\b p)) d\b\theta d\b p\\
&= \int \exp(-U(\b \theta) d\b\theta \int \exp(-K(\b p)) d\b p\\
&= Z_U Z_K.
\end{align}
$$

ここで、$Z_U = \int \exp(U(\b\theta)) d\b\theta$、
$Z_K = \int \exp(K(\b p)) d\b p$。
ここから、 $\b\theta, \b p$の分布は周辺化すれば得られる：

$$
\begin{align}
p(\b\theta|X) &= \int p(\b\theta, \b p|X) d\b p\\
&= \int \frac{1}{Z} \exp(-H(\b \theta, \b p)) d\b p\\
&= \int \frac{1}{Z} \exp(-U(\b \theta))\exp(K(\b p)) d\b p\\
&= \frac{1}{Z} \exp(-U(\b \theta))\int \exp(K(\b p)) d\b p\\
&= \frac{Z_K}{Z} \exp(-U(\b \theta)) \quad \text{}\\
&\propto \exp(-U(\b\theta)).
\end{align}
$$

前述のように、これはポテンシャルエネルギー  $U(\b\theta)$ をデータ分布
$p(\b\theta|X)$ を用いて定めていると見るのが良いだろう。
この式を見てみると、$\b\theta$ 単独で見ても、
エネルギーと確率分布の間にカノニカル分布の関係があることがわかる。

同様に、$\b\theta$について周辺化すれば、

$$
\begin{align}
p(\b p|X) = \frac{Z_U}{Z}\exp(-K(\b p))\propto \exp(-K(\b p)).
\end{align}
$$

$\b p$ については $K(\b p)$ を定めたので、
$\b\theta$ とは逆に $K(\b p)$ から $\b p$ の確率分布を定めていると見るのが良いだろう。

すると$\b\theta$と$\b p$は条件付き独立であることがわかる：

$$
\begin{align}
p(\b \theta, \b p|X)&=\frac{1}{Z}\exp(-H(\b\theta, \b p))\\
&=\frac{1}{Z}\exp(-U(\b\theta))\exp(-K(\b p))\\
&=\frac{Z_K}{Z}\exp(-U(\b\theta))\frac{Z_U}{Z}\exp(-K(\b p))\\
&=p(\b\theta|X)p(\b p|X)
\end{align}
$$

以上の話は、ハミルトニアンが $\b\theta$に依存する部分と
$\b p$に依存する部分に分解できることがキーとなっている。
今回は、確率モデルを拡張する（確率モデルが持つパラメータを増やす）方法として、
ハミルトニアンが分解できることから $\b\theta$ と $\b p$ の条件付き独立性を導いたが、
逆に条件付き独立性からスタートして、ハミルトニアンの分解を導くこともできる。

### ポテンシャルエネルギーの勾配の計算

これらの関係式の中で特に注目すべきなのが、ポテンシャルエネルギー
$U(\b\theta)$ と事後分布 $p(\b\theta|X)$ の関係式

$$
p(\b\theta|X) \propto \exp(-U(\b\theta))
$$

で、両辺の対数を取ってさらに $\b\theta$ で微分すると、

$$
\nabla_{\b\theta} \log p(\b\theta|X) = -\nabla_{\b\theta} U(\b\theta)
$$

が得られる。左辺をベイズの定理を用いて変形すると、

$$
\begin{align}
\nabla_{\b\theta} \log p(\b\theta|X)
&= \nabla_{\b\theta}  \left( \log p(\b\theta)
+ \log p(X | \b\theta) - \log p(X) \right)\\
&= \nabla_{\b\theta} \log p(\b\theta) + \nabla_{\b\theta}\log p(X | \b\theta)
\end{align}
$$

冒頭の問題設定で述べたように今回は対数尤度の勾配を計算、もしくは推定できることを仮定している。
例えば生成モデルをニューラルネットでモデル化したならば、
$\nabla_{\b\theta} \log p(X|\b\theta)$ は誤差逆伝播で計算可能である。
従って、ポテンシャルエネルギーの勾配、すなわち運動方程式では
力に対応する値も計算できることがわかる。

## 各MCMCサンプリングの運動方程式

前述した通り、質点の運動を支配する運動方程式を様々なものに設定することにより、
各サンプリングの手法が得られる。
では、具体的に各手法で利用される運動方程式を見ていこう。

## HMC

HMCでは、古典的な運動方程式（正準方程式）を考える。

$$
\begin{align}
\frac{d\b\theta}{dt} &= \nabla_{\b p}H (\b\theta, \b p) \\
\frac{d\b p}{dt} &= - \nabla_{\b\theta}H(\b\theta, \b p).
\end{align}
$$

解析力学の文脈では、 $\b\theta$ を一般化座標と呼び、 $\b p$を一般化運動量と呼ぶ。
一般化座標は $\b q$ で表すことが多いが、 ここではモデルパラメータの時の記号
$\b\theta$ をそのまま用いることにする。

$H$ は前述のようにポテンシャルエネルギーと位置エネルギーに分離できるので、
上式の右辺はさらに変形できる：

$$
\begin{align}
\nabla_{\b p}H(\b\theta, \b p) &= \nabla_{\b p}K(\b p) = \b p\\
- \nabla_{\b\theta}H(\b\theta, \b p) &= - \nabla_{\b\theta}U(\b\theta).
\end{align}
$$

さらに、後々のSDEと式の形を揃えるために、両辺の分母を払うと結局運動方程式は以下のようになる。

$$
\begin{align}
d\b\theta &= \b pdt\\
d\b p &= -\nabla_{\b\theta}U(\b\theta) dt.
\end{align}
$$

これを離散化すると、パラメータの更新式は次のようになる。

$$
\begin{align}
\b\theta &\leftarrow \b\theta + \b p h\\
\b p &\leftarrow \b p -\nabla_{\b\theta}U(\b\theta) h
\end{align}
$$

ここで、$h$は微小時間に対応するパラメータの更新幅である。

さて、この更新を行うにはポテンシャルエネルギーの勾配
$-\nabla_{\b\theta}U(\b\theta)$ を計算できる必要がある。
この値は既に出てきており事後分布の対数 の勾配 $\nabla_{\b\theta} \log p(\b\theta|X)$  （より一般的な問題設定ならば、サンプリングを行いたい確率分布を $p(\b\varphi)$ としたら、
$\nabla_{\b\varphi} \log p(\b\varphi)$ ）であった。
前節で解説した通りこの値は計算できることを仮定している。

## Leapfrog法

前述のパラメータの更新方法についてもう少し考えよう。

$$
\begin{align}
\b\theta &\leftarrow \b\theta + \b p h\\
\b p &\leftarrow \b p -\nabla_{\b\theta}U(\b\theta) h
\end{align}
$$

この方法では、 時刻 $t$ の $\b\theta$ から時刻 $t+h$ の $\b\theta$ を得るのに、時刻 $t$ での $\b p$ から得られる勾配を利用している（今回の場合勾配  $\frac{\partial d\b\theta}{\partial t}$ は $\b p$ そのものである）。

これを次のように変更することで、離散化による近似誤差を減らす手法である。

$$
\begin{align}
\b p &\leftarrow \b p -\nabla_{\b\theta}U(\b\theta) \frac{h}{2}\\
\b\theta &\leftarrow \b\theta + \b p h\\
\b p &\leftarrow \b p -\nabla_{\b\theta}U(\b\theta) \frac{h}{2}
\end{align}
$$

すなわち、時刻 $t$ から $t+h$ での $\b\theta$ の更新に時刻 $t+h/2$ での $\b p$の推定値を利用する。同様のことは $\b p$ の更新でもできる：

$$
\begin{align}
& \b p \leftarrow \b p - \frac{h}{2} \nabla_{\b\theta}U(\b\theta)\\
& \text{For $l = 1$ to $L$}\\
& \qquad \b\theta \leftarrow \b\theta + h \b p\\
& \qquad \text{If $l \not = L$}\\
& \qquad \qquad \b p \leftarrow \b p - h \nabla_{\b\theta}U(\b\theta)\\
& \b p \leftarrow \b p - \frac{h}{2} \nabla_{\b\theta}U(\b\theta)\\
\end{align}
$$

このように、互い違いに複数のパラメータを更新していく手法をLeapfrog法という。

理論的にも、この更新方法で離散化の誤差を減らすこともわかる。
鍵となるのが、正方行列 $A$, $B$ に対して成立する、Baker-Campbell-Haussdorff(BCH)の公式

$$
\exp(hA)\exp(hB) = \exp\left(hA + hB + \frac{h^2}{2} [A, B]\right) + O\mathcal(h^3)
$$
である。ここで、 $[A, B] = AB - BA$ は $A$ と $B$ の交換子である。BCHの公式を用いると、

$$
\exp(hL_{\b\theta})\exp(hL_{\b p}) = \exp(h(L_{\b\theta} + L_{\b p})) + O(h^2)
$$

であるのに対して、

$$
\exp\left(\frac{h}{2} L_{\b p}\right)\exp(hL_{\b\theta})\exp\left(\frac{h}{2} L_{\b p}\right) = \exp(h(L_{\b\theta} + L_{\b p})) + O(h^3)
$$

となることに由来する。


## HMC法のアルゴリズム

以上を元に、HMCのアルゴリズムは次のようになる。
このサンプリング方法で、きちんと事後分布 $p(\b\theta |X)$ からサンプリングが行えることの証明は元論文を参照されたい。

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
def update(q, p, x):
    def update_p(p, q, x, dt):
        dq = estimate_grad(q, x)
        return p + dq * dt

    def update_q(q, p, dt):
        return q + p * dt

    p = update_p(p, q, x, h / 2)
    for l in six.moves.range(1, L + 1):
        q = update_q(q, p, h)
        if l != L:
          p = update_p(p, q, x, h)
    p = update_p(p, q, x, h / 2)

    return q, p

theta = initialize_param()
for epoch in six.moves.range(EPOCH):
    p = numpy.random.randn(*theta.shape)
    x = get_minibatch()
    theta, p = update(theta, p, x)
```

## SGHMC

まずは、SGHMCが利用する運動方程式を天下り的に書いてしまおう。
SGHMCで考える運動方程式は以下の(2次の)Langevin方程式である。

$$
\begin{align}
d\b\theta &= \b p dt \\
d\b p &= \nabla_{\b \theta} U(\b\theta)dt - A\b pdt + \mathcal N(\b 0, 2AId t)
\end{align}
$$

ここで、 $A$ はスカラー値、 $\mathcal N(\b 0, 2AId t)$ はパラメータ $\b\theta$ と同次元の（従って $\b p$と同次元の）ワイナー過程である。
ワイナー過程は微小時間 $dt$ の間に平均0、分散 $2AIdt$ のブラウン運動を行う確率過程である。
先ほどのHMCは1回の微分方程式だったので、初期位置を決めればその後の運動は一意的に決定される。
それに対して、この方程式はワイナー過程が入るために初期位置を決めてもその後の運動は
確率的にしか決まらないことに注意。

そこで、物体の位置を確率分布として与えて、その確率分布が時間と共にどのように変化するかを考えよう。前述のFokker-Planck方程式を今回の場合に適用すると、 $\exp(-H(\b\theta, \b p))$  に比例する確率分布 $p(\b\theta, \b p)$ が、このSDEの定常状態となっていることがわかる。従って、この確率分布を周辺化した $p(\b\theta) \propto \exp(-U(\b\theta))$ も $\theta$ に関して定常分布となっている。 詳しい導出はSGHMCの論文のTheorem3.2を参照されたい。また、本当は適当な初期分布から初めて、十分時間が経ったときに定常分布に終息するということは別途証明しなければならないが、今回は割愛する。

## SGHMCのアルゴリズム

これを踏まえると、SGHMCのアルゴリズムは次のようになる

$$
\begin{align}
& \text{Initialize $\b\theta$ and $\b p$}\\
& \text{For $i = 1$ to $\infty$}\\
& \qquad \text{For $l = 1$ to $L$}\\
& \qquad \qquad \b\theta \leftarrow \b\theta + \b p h \\
& \qquad \qquad \b\zeta \sim \mathcal N(\b 0, 2AhI)\\
& \qquad \qquad \b p \leftarrow (1-Ah)\b p + \nabla_{\b \theta} U(\b\theta) h + \b\zeta\\
& \qquad \text{Accept $\b\theta$}
\end{align}
$$

疑似コード

```python
def update(q, p, x):
    def update_p(p, q, x):
        dq = estimate_grad(q, x)
        return ((1 - A * eps) * p + dq * eps
                + math.sqrt(2 * A * eps)
                * numpy.random.randn(*q.shape))

    def update_q(q, p):
        return q + p * eps

    for l in six.moves.range(L):
        p = update_p(p, q, x)
        q = update_q(q, p)
    return q, p

theta, p = initialize_param()
for epoch in six.moves.range(EPOCH):
    x = get_minibatch()
    theta, p = update(theta, p, x)
```

## エルゴード性

アルゴリズムをみると、サンプリングはパラメータの初期状態を適当に決めて、運動方程式に従って更新している。
つまり、一つの質点の運動を長い時間観測し、定期的にその位置をサンプリングしていることに対応している。
一方で定常状態は、パラメータの初期分布を適当に決めて、その分布を運動方程式に従って変化させていく。
つまり、たくさんの質点をパラメータ空間にばらまき、それをしばらく放っておくと定常状態となるので、
適当な瞬間ですべての物体の位置のスナップショットを取っていることになる。
前者は1つの質点を長時間観測しているのに対し、後者は多数の質点のある瞬間を観測している。
この2つが同じ分布になることは自明ではない。この2つが実は一致する時、エルゴードを持つと言う。

今回のSGHMCはこのエルゴード性を持つ。そのため、前述のアルゴリズムのように、1点を適当に取りその流れを適宜サンプリングすれば、
SGHMCがエルゴード性を持つことの証明は今回は割愛する。
また、HMCはエルゴード性を持たないことに注意しよう。
運動方程式に従って質点が運動している間、ハミルトニアンが保存するため、
スタートの時に持っていたハミルトニアンの等エネルギー面から外れることができない。
これでは、今回はパラメータの同時分布が $\exp(H(\varphi))$ に従う分布を再現できないためである。
そのために、HMCのサンプリングアルゴリズムでは、運動量 $\b p$ の初期位置をサンプリングの度に引き直している。

## MCMCを確率的にする基本アイデア

さて、ここまででは、事後分布の対数の勾配 $\nabla_{\b\theta}\log p(\b\theta|X)$
（これはポテンシャルエネルギーの勾配 $\nabla_{\b\theta} U(\b\theta)$ でもあった）
が計算できるような状況を考えてきた。
しかし、訓練データ数 $|X|$ が大きくなるとそれは難しくなる。
例えば $X$ がモデルからi.i.d.でサンプリングをされているような状況を考えよう。
すると、この値は、

$$
\begin{align}
\nabla_{\b\theta}\log p(\b\theta|X)
&= \nabla_{\b\theta}\log p(\b\theta) + \nabla_{\b\theta}\log p(X | \b\theta)\\
&= \nabla_{\b\theta}\log p(\b\theta)
+ \sum_{\b x \in X} \nabla_{\b\theta} \log p(\b x | \b\theta)
\end{align}
$$

と計算できるが、訓練データが多い時には第2項の和を取るのが困難になるためである。
そこで、$\nabla_{\b\theta} U(\b\theta)$ の代わりに、
この値の推定値 $\widehat{\nabla_{\b\theta} U} (\b\theta)$
を利用しようというのがSG-MCMCの基本発想である。

どのような推定量を利用するかはアルゴリズムの設計者次第であるが、
一番オーソドックスな方法と思われるのは、すべてのデータについて勾配を計算する代わりに、
その中から $M$ 個のデータだけをサンプリングする方法だろう。
すなわち、 $\tilde X \subset X$、 $|\tilde X| \ll |X|$ となる
$\tilde X$ を利用して第2項を

$$
\sum_{\b x \in X} \nabla_{\b\theta}\log p(\b x_i | \b\theta)
\approx \frac{|X|}{|\tilde X|} \sum_{\b{\tilde x} \in \tilde X}
\nabla_{\b\theta}\log p(\b{\tilde x} | \b\theta)
$$

と推定する。 この推定値は $\tilde X$ を $X$ から一様ランダムにサンプリングしたならば、
不偏推定量である。

## HMCからSGHMCへの変形

SGHMCが提案された論文 [Chen+14]を見ると、このアルゴリズムはもともとHMCにおいて、
力に対応する項 $\nabla_{\b\varphi}U(\b\varphi)$ を前述の方法で推定した推定値
$\widehat{\nabla_{\b\varphi}U}(\b\varphi)$ で置き換えるというアイデアに由来する。

もちろん、単純にこの推定値に置き換えると、運動方程式としては別のものになってしまう。
では、両者がどのくらい違うものなのかを考えてみよう。
訓練データ $X$ がi.i.d でサンプリングされていることから、中心極限定理を考えると、
この推定値は $\nabla_{\b \varphi}U(\theta)$ を中心としたガウス分布におおよそ従っていると考えられる。
この分散の値を仮に $V(\b\theta)$ と置こう。

すると、推定値を用いた場合のHMCの更新則は以下のようになる（というかなってしまっている）。

$$
\begin{align}
\b\theta &\leftarrow \b\theta + \b p h\\
\b\zeta &\sim \mathcal N (\b 0, V(\b\theta))\\
\b p &\leftarrow \b p - \left(\nabla_{\b\theta}U(\b\theta) + \b\zeta \right)h\\
\end{align}
$$

この更新則は以下のSDEを離散化である
（$\b\zeta\sim \mathcal N (\b 0, V(\b\theta))$ なので、
$\b\zeta h \sim \mathcal N (\b 0, V(\b\theta)h^2)$
であることに注意）。

$$
\begin{align}
d\b\theta &= \b p dt \\
d\b p &= \nabla_{\b \theta} U(\b\theta)dt + \mathcal N(\b 0, V(\theta)hdt)
\end{align}
$$

ところが、[Chen+14]のCorollary 3.1で証明されているように、
$p(\b\theta, \b p)\propto \exp(-H(\b\theta, \b p))$ はこの運動方程式の定常分布とならない。
すなわち、訓練データのサンプリングによるノイズのために、正しい事後分布からサンプリングできない。
[Chen+14]では、このことを上記のSDEに従って変化する、パラメータの確率分布はエントロピーが
時間とともに増加することを利用している。

これを修正する方法として、[Chen+14]では、Metropolis-Hastingサンプリングの
棄却フェーズを追加することを検討しているが、この棄却フェーズに全データを用いた場合には
計算コストが高くなり、サンプリングをした場合には高確率で棄却されてしまうことを指摘している。

そこで、別の方法として、定常分布が所望のカノニカル分布になるようにSDEを修正する。
そのために、 $B(\b\theta) = \frac{1}{2}V(\b\theta)h$ として、この値に依存する摩擦の項を加えたSDEを考える。

$$
\begin{align}
d\b\theta &= \b p dt \\
d\b p &= \nabla_{\b \theta} U(\b\theta)dt - B(\b\theta)\b pdt + \mathcal N(\b 0, 2B(\b\theta)dt)
\end{align}
$$

すると、Fokker-Planck方程式の一般論より、カノニカル分布が定常分布になることが示される。
このSDEをそのまま離散化して更新則を得ると、次のようになる。
$\widehat{\nabla_{\b \theta} U}(\b\theta) = \nabla_{\b \theta} U(\b\theta) + \mathcal N (\b 0, V(\b\theta))$
より、ノイズの項が消えることに注意。

$$
\begin{align}
\b\theta &\leftarrow \b\theta + \b p h \\
\b p &\leftarrow \b p + \widehat{\nabla_{\b \theta} U}(\b\theta)h - B(\b\theta)\b ph
\end{align}
$$

この更新則に現れる $B(\b\theta)$ は $\nabla_{\b \theta} U$ のサンプリングに伴う分散に由来するため、
実際の値を求めることはできない。
それを回避する最も簡単な方法は、$B(\b\theta)$ の代わりに、その推定値 $B(\b\theta)$ を用いるというものである。
[Chen+14] では別の回避方法も提案している。
それは、 $\widehat{B}$ の他に $C(\b\theta) \succeq \widehat{B}(\b\theta)$ となる
半正定値行列値関数 $C$ を用意して、次のSDEを考えるというものである。

$$
\begin{align}
d\b\theta &= \b p dt \\
d\b p &= \nabla_{\b \theta} U(\b\theta)dt - C(\b\theta)\b pdt
+ \mathcal N(\b 0, 2(C(\b\theta)-\widehat{B}(\b\theta))dt) + \mathcal N (\b 0, 2B(\b\theta)dt)
\end{align}
$$

これを離散化すると次の通り、勾配として $\nabla_{\b \theta} U$ の代わりにその推定値
$\widehat{\nabla_{\b \theta} U}$ を用いているので、最後の項のガウス分布からはサンプリング
する必要がないことに注意。

$$
\begin{align}
\b\theta &\leftarrow \b\theta + \b p h \\
\b\zeta &\sim \mathcal N (\b 0, 2(C(\b\theta)-\widehat{B}(\b\theta))dt)\\
\b p &\leftarrow \b p + \widehat{\nabla_{\b \theta} U}(\b\theta)h - C(\b\theta)\b ph + \b\zeta
\end{align}
$$

実際にはできないが、$C = B$ とすれば、一つ目の回避方法に帰着される。
$B$ が $B = \frac{h}{2}V$ と $O(h)$ の大きさを持っているため、
$C$ 十分大きい値とし、時間幅 $h$ を十分小さくすれば、
推定が難しい $\widehat{B}$ の影響が小さくなり、ユーザーのコントロールできる $C$ の値が支配的になる。
これが、新たに $C \succeq \widehat{B}$ なる項を設ける利点である。

後述するSGLD, SGNHT, mSGNHT, Santaでも推定値を利用する場合にも、
それに由来するノイズをキャンセルするための上記の修正を必要に応じて行わなければならないことに注意する。
ここでは、その修正は行わずに、更新則には本来の勾配の値 $\nabla_{\b\varphi} U(\b\varphi)$
を利用することにする。

## SGLD

SGLDは次の1次のLangevin Dynamicsを離散化したものである。

$$
\begin{align}
d\b\theta =  \nabla_{\b \theta} U(\b\theta) dt + \mathcal N(\b 0, 2Idt)
\end{align}
$$

HMC, SGHMCとは異なりSGLDには運動量に対応するパラメータ $\b p$ は存在しない。更新されるのは元のモデルのパラメータ $\b\theta$ のみである。

SGHMCの時と同様に、 $p(\b\theta) \propto \exp(-U(\b\theta))$ がこの方程式の定常状態である事やエルゴード性が成り立つ。
従って、運動方程式を離散化すれば、サンプリングのアルゴリズムは次の通り。

$$
\begin{align}
& \text{Initialize $\b\theta$ and $\b p$}\\
& \text{For $i = 1$ to $\infty$}\\
& \qquad \text{For $l = 1$ to $L$}\\
& \qquad \qquad \b\zeta \sim \mathcal N(\b 0, 2Ih)\\
& \qquad \qquad \b\theta \leftarrow \b\theta + \nabla_{\b \theta} U(\b\theta) h + \b\zeta\\
& \qquad \text{Accept $\b\theta$}
\end{align}
$$

疑似コード

```python
def update(theta, x):
    for l in six.moves.range(L):
        d_theta = estimate_grad(theta, x)
        eta = numpy.random.randn() * numpy.sqrt(2 * eps)
        theta += d_theta * eps + eta
    return theta

theta = initializer_param()
for epoch in six.moves.range(EPOCH):
    x = get_minibatch()
    theta = update(theta, x)
```

## SGHMCからSGLDの導出

SGLDはSGHMCの極限をとったものと見る事ができる。
歴史的にはSGLDは2011年でSGHMCは2014年なので、順序としては逆だが、SGHMCからSGLDを導出してみよう。

まず、SGHMCを以下のように微分方程式の形に直しておく

$$
\begin{align}
\frac{d\b\theta}{dt} &= A\b p \\
\frac{d\b p}{dt} &= \nabla_{\b \theta} U(\b\theta) - A\b p + \b\zeta \\
\b\zeta & = \mathcal N(\b 0, 2AI)
\end{align}
$$

ここで、 $\b\zeta$ は確率過程で以下を満たす。

$$
\begin{align}
\left< \b\zeta(t) \right> &= \b 0\\
\left< \b\zeta_i(t), \b\zeta_j(s) \right> &= 2A \delta_{ij} (t - s)
\end{align}
$$

$\b p$を消去すると

$$
\begin{align}
A \frac{d^2\b\theta}{dt^2} &= \nabla_{\b \theta} U(\b\theta) - A\frac{d\b\theta}{dt} + \b\zeta
\end{align}
$$


まず、$u = At$とスケール変換すると、

$$
\begin{align}
A \frac{d^2\b\theta}{du^2} &= \nabla_{\b \theta} U(\b\theta) - \frac{d\b\theta}{du} + \tilde{\b\zeta}
\end{align}
$$

ここで、 $\tilde{\b\zeta}$ は次を満たす

$$
\begin{align}
\left< \tilde{\b\zeta(u)} \right> &= \b 0\\
\left< \tilde{\b\zeta_i}(u), \tilde{\b\zeta_j}(v) \right> &= \delta_{ij} (u - v))
\end{align}
$$

この状態で $A\to \infty$ とすると

$$
\begin{align}
\b 0 &= \nabla_{\b \theta} U(\b\theta) - \frac{d\b\theta}{du} + \tilde{\b\zeta}
\end{align}
$$

式を整理して分母を払い、変数を取り替えると1次のLangevin dynamicsが得られる。

$$
\begin{align}
d\b\theta =  \nabla_{\b \theta} U(\b\theta) dt + \mathcal N(\b 0, 2Idt)
\end{align}
$$

## SGNHT

SGLDはSGHMCの極限をとり、変数を消去する事で得られたが、SGNHTは逆に系をコントロールする新しい変数を導入することで得られる。

具体的には、これまでと同様のモデルのパラメータ $\b\theta$, 運動量に対応するパラメータ $\b p$
の他に、スカラー値のパラメータ $\xi$ を導入し、次の運動方程式を考える

$$
\begin{align}
d\b\theta &= \b p dt \\
d\b p &= \left(\nabla_{\b \theta} U(\b \theta) - \xi \b p\right) dt + \mathcal N (\b 0, 2AIdt)\\
d\xi &= \left( \frac{1}{d} \b p^T \b p - 1\right) dt.
\end{align}
$$

ここで、 $d$ は $\b p$ の次元、 $A > 0$ はスカラー値である。
アルゴリズムも、この更新式からほとんどそのまま導出できる。

$$
\begin{align}
& \text{Initialize $\b\theta$ , $\b p$ , and $\xi$}\\
& \text{For $i = 1$ to $\infty$}\\
& \qquad \text{For $l = 1$ to $L$}\\
& \qquad \qquad \b\theta \leftarrow \b\theta + \b p h\\
& \qquad \qquad \b\zeta \sim \mathcal N(\b 0, 2AIh)\\
& \qquad \qquad \b p \leftarrow (1 - \xi h)\b p + \nabla_{\b \theta} U(\b\theta) h + \b\zeta\\
& \qquad \qquad \xi \leftarrow \xi + \left( \frac{1}{d} \b p^T \b p - 1\right) h\\
& \qquad \text{Accept $\b\theta$}
\end{align}
$$

疑似コード
```python
def update(q, p, xi, x):
    def update_p(p, q, xi, x):
        dq = estimate_grad(q, x)
        return ((1 - xi * eps) * p + dq * eps
                + math.sqrt(2 * A * eps)
                * numpy.random.randn(*q.shape))

    def update_q(q, p, xi):
        return theta + p * eps

    def update_xi(xi, p, q):
        return xi + (numpy.sum(p * p) / p.size - 1) * eps

    for l in six.moves.range(L):
        p = update_p(p, q, xi, x, eps)
        q = update_q(q, p, xi, eps)
        xi = update_xi(xi, p, q, eps)
    return q, p, xi

theta, p, xi = initialize_param()
x = model.generate(args.N, args.theta1, args.theta2)
for epoch in six.moves.range(args.epoch):
    x = get_minibatch()
    theta, p, xi = update(theta, p, xi, x)
```

## SGNHTの直感的解釈

では、この運動方程式について、もう少し考える。そのために唐突であるが、 運動エネルギー $K(\b p)$
の期待値 $\mathbb{E}_{\b p}\left[ K(\b p)\right]$ を計算してみよう。
我々は運動エネルギー $K(\b p)$ を $K(\b p) = \frac{1}{2} \b p^T \b p$ と定義したことを思い出そう。
また、期待値は定常状態での $\b p$ の 確率分布 $p(\b p) \propto \exp(-K(\b p))$ についてとる。

$$
\begin{align}
\mathbb{E}_{\b p} \left[K(\b p)\right] &= \int \left[\frac{1}{2} \b p^T \b p\right]
\left[\frac{1}{Z_K} \exp\left(-\frac{1}{2}\b p^T \b p\right) \right]d\b p\\
&= \frac{1}{Z_K} \sum_{i=1}^{d} \left[\int \frac{1}{2}p_i^2 \exp\left(-\frac{1}{2} p_i^2\right)dp_i
\prod_{j \not=i} \int \exp\left( -\frac{1}{2} p_j^2 \right) dp_j \right]\\
\end{align}
$$

ここで、$Z_K$ は $\exp(-K(\b p)))$ についての分配関数 $Z_K = \int \exp(-K(\b p)) d\b p$
であった。部分積分により、

$$
\begin{align}
\int \frac{1}{2} p_i^2 \exp\left(-\frac{1}{2} p_i^2\right)dp_i = \frac{1}{2} \int \exp\left(-\frac{1}{2} p_i^2\right)dp_i
\end{align}
$$

なので、

$$
\begin{align}
\mathbb{E}_{\b p} \left[K(\b p)\right]
&= \frac{1}{2Z_K} \sum_{i=1}^{d} \left[
\prod_{j=1}^{d} \int \exp\left( -\frac{1}{2} p_j^2 \right) dp_j \right]\\
&= \frac{1}{2Z_K} \sum_{i=1}^{d} Z_K = \frac{d}{2}
\end{align}
$$

となる。
これは、エネルギー等分配の法則を部分的に証明したことに対応している。
これを変形すると、

$$
\begin{align}
\mathbb{E}_{\b p} \left[ \frac{1}{d} \b p^T \b p - 1 \right] = 0
\end{align}
$$

となり、SGNHTの運動方程式での $\xi$ の勾配が $\frac{d\xi}{dt}$ が現れる。
つまり、 $\xi$ は現在の状態に比べて運動エネルギーが（従って温度が）高かったら増加し、
逆に運動エネルギーが低かったら低下するように設計されている。
この意味で、 $\xi$ は系の温度をコントロールするサーモスタットの役割を果たしている。
$\b p$ に関する方程式を見ると、$\xi$ は摩擦に対応する項として現れている。
すなわち、温度が高すぎる場合には摩擦を増加させて運動をより早く減衰させるようにし、
逆に温度が低すぎる場合には摩擦を減少させる。


## mSGNHT

SGNHTでは、サーモスタットの役割を果たす変数として $\xi$ 1つを用意し、 $\b p$ 全体をコントロールしたが、
mSGNHTでは、 $\b p$ の各次元に対してサーモスタットを用意して各次元をコントロールする。
すなわち、 $\b p$ と同次元の変数 $\b \xi$ を用意し、以下の運動方程式を考える。

$$
\begin{align}
d\b\theta &= \b p dt \\
d\b p &= \left(\nabla_{\b \theta} U(\b \theta) - \b\xi \odot \b p\right) dt + \mathcal N (\b 0, 2AIdt)\\
d\b\xi &= \left( \b p \odot \b p - \b 1\right) dt.
\end{align}
$$

サンプリングアルゴリズムと疑似コードは次の通り

$$
\begin{align}
& \text{Initialize $\b\theta$ , $\b p$ , and $\b\xi$}\\
& \text{For $i = 1$ to $\infty$}\\
& \qquad \text{For $l = 1$ to $L$}\\
& \qquad \qquad \b\theta \leftarrow \b\theta + \b p h\\
& \qquad \qquad \b\zeta \sim \mathcal N(\b 0, 2AIh)\\
& \qquad \qquad \b p \leftarrow (\b 1 - \b\xi h)^T\b p + \nabla_{\b \theta} U(\b\theta) h + \b\zeta\\
& \qquad \qquad \xi \leftarrow \xi + \left( \b p \odot \b p - 1\right) h\\
& \qquad \text{Accept $\b\theta$}
\end{align}
$$

疑似コード
```python
def update(q, p, xi, x):
    def update_p(p, q, xi, x):
        dq = estimate_grad(q, x)
        return ((1 - xi * eps) * p + dq * eps
                + math.sqrt(2 * A * eps)
                * numpy.random.randn(*q.shape))

    def update_q(q, p, xi):
        return theta + p * eps

    def update_xi(xi, p, q):
        return xi + (p * p - 1) * eps

    for l in six.moves.range(L):
        p = update_p(p, q, xi, x, eps)
        q = update_q(q, p, xi, eps)
        xi = update_xi(xi, p, q, eps)
    return q, p, xi

theta, p, xi = initialize_param()
x = model.generate(args.N, args.theta1, args.theta2)
for epoch in six.moves.range(args.epoch):
    x = get_minibatch()
    theta, p, xi = update(theta, p, xi, x)
```

## Santa

$$
\begin{align}
d\b\theta &= G_1(\b\theta) \b pdt\\
d\b p &= \left(\nabla_{\b \theta} U(\b \theta) - \b \xi \b p\right) dt + \b F(\b\theta, \b\xi) dt + \left(\frac{2}{\beta} G_2(\b\theta)\right) d\b W\\
d\b\xi &= \left(\b p\odot \b p - \frac{\b 1}{\beta}\right) dt
\end{align}
$$
ここで、$\b F(\b\theta, \b\xi) = \frac{1}{\beta} \nabla_{\b \theta} G_1(\b\theta) + G_1(\b\theta)\left(\b\xi - G_2(\b\theta)\right) \nabla_{\b \theta} G_2(\b\theta).$



## SSIによる近似精度向上

Leapfrog法では、時刻 $t$ から 時刻 $t+h$ での $\b\theta$ の更新に、時刻 $t+h/2$ での
$\b p$ の値を用いることで、近似誤差を減らした。このアイデアはSymmetric Splitting Integrators(SSI)
という方法に一般化することができる。

アイデアはHMCと似ており、考えている運動方程式をいくつかの部分に分解し、1つずつを順番に更新していくというものである。
[Chen+15a]では、SGHMCに、[Chen+15b]では、mSGNHTに、[Chan+15c]ではSantaに、
[Leimkuhler+15]ではこの論文の中で提案しているAdaptive Langevin Thermostat(Ad-Langevin)に、
それぞれSSIを適用している。

まず、Leapfrog法を見直してみよう。HMCの運動方程式を見直してみよう。

$$
\begin{align}
d\b\theta &= \b pdt\\
d\b p &= -\nabla_{\b\theta}U(\b\theta) dt.
\end{align}
$$

この方程式を次の2つに分解する

$$
\begin{array}{ll}
A \left\{
\begin{matrix}
d\b\theta = \b pdt\\
d\b p = 0
\end{matrix}
\right.
&
B \left\{
\begin{matrix}
d\b\theta = 0\\
d\b p = -\nabla_{\b\theta}U(\b\theta) dt
\end{matrix}
\right.
\end{array}
$$

すると、これらの方程式は解析的に解ける。
例えば $A$ の場合、 $\b p$ は時刻によらない定数であり、 $\b \theta$ は $t$ に関する1次式となる。
右辺も同様である。

この方程式を

* 時刻 $t$ を初期条件として$A$を解き、時刻 $t+h/2$ の値を得る
* 時刻 $t$ を初期条件として $B$ を解き、時刻 $t+h$ の値を得る
* 時刻 $t+h/2$ を初期条件として $A$を解き、時刻 $t+h$ の値を得る

という順番で解くと、Leapfrog法を適用したHMCの更新則となる。
このように、今解きたい運動方程式をいくつかの解析的に解ける部分に分解し、1つずつ解析的な解で
更新していく手法をSymmetric Splitting Integrator(SSI)と呼ぶ。
Symmetricとついているのは、更新の順番が$A\rightarrow B \rightarrow A$
と対称的であることに由来する。

それでは、SSIをSGHMCに適用してみよう。
SGHMCの更新則は以下の通りであった。

$$
\begin{align}
d\b\theta &= \b p dt \\
d\b p &= \nabla_{\b \theta} U(\b\theta)dt - A\b pdt + \mathcal N(\b 0, 2AId t)
\end{align}
$$

今回はこれを以下の3つの部分に分解する。
これらはどれも解析的に解けることに注意しよう。

$$
\begin{array}{ll}
A \left\{
\begin{matrix}
d\b\theta = \b pdt\\
d\b p = 0
\end{matrix}
\right.
&
B \left\{
\begin{matrix}
d\b\theta = 0\\
d\b p = -A\b pdt
\end{matrix}
\right.
\end{array}
O \left\{
\begin{matrix}
d\b\theta = 0\\
d\b p = \nabla_{\b\theta}U(\b\theta) + \mathcal N (\b 0, 2AIdt)
\end{matrix}
\right.
$$

この上で時刻 $t$ から $t+h$ の 更新では、 $A(h/2)\to B(h/2)\to O(h)\to B(h/2)\to A(h/2)$ の順番に方程式を解いていく。
括弧の中は更新する時間幅である。
結局、更新則は次のようになる

$$
\begin{align}
A: & \b\theta \leftarrow \b\theta + \b p \frac{h}{2}\\
B: & \b p \leftarrow \exp\left(-A\frac{h}{2}\right) \b p\\
O: & \b p \leftarrow \b p + \nabla_{\b\theta}U(\b\theta)
+ \b\zeta \qquad \b \zeta \sim \mathcal N (\b 0, 2AIdt)\\
B: & \b p \leftarrow \exp\left(-A\frac{h}{2}\right) \b p\\
A: & \b\theta \leftarrow \b\theta + \b p \frac{h}{2}\\
\end{align}
$$

mSGNHTに対してSSIを適用した場合も同様である。詳細は割愛して分解の方法と結論の更新則だけを引用しよう。
運動方程式は以下の3つに分解する。

$$
\begin{array}{ll}
A \left\{
\begin{matrix}
d\b\theta = \b pdt\\
d\b p = 0\\
d\b \xi = (\b p \odot \b p - I) dt
\end{matrix}
\right.
&
B \left\{
\begin{matrix}
d\b\theta = 0\\
d\b p = -\b \xi \odot \b pdt\\
d\b\xi = 0
\end{matrix}
\right.
\end{array}
O \left\{
\begin{matrix}
d\b\theta = 0\\
d\b p = \nabla_{\b\theta}U(\b\theta) + \mathcal N (\b 0, 2AIdt)\\
d\b\xi = 0
\end{matrix}
\right.
$$

対応する更新則は以下のとおり
$$
\begin{align}
A: & \b\theta \leftarrow \b\theta + \b p \frac{h}{2}, \quad \b\xi \leftarrow (\b p \odot \b p -I)\frac{h}{2}\\
B: & \b p \leftarrow \exp\left(-\b\xi\frac{h}{2}\right) \odot \b p\\
O: & \b p \leftarrow \b p + \nabla_{\b\theta}U(\b\theta)
+ \b\zeta \qquad \b \zeta \sim \mathcal N (\b 0, 2AIdt)\\
B: & \b p \leftarrow \exp\left(-\b\xi\frac{h}{2}\right) \odot \b p\\
A: & \b\theta \leftarrow \b\theta + \b p \frac{h}{2}, \quad \b\xi \leftarrow (\b p \odot \b p -I)\frac{h}{2}\\
\end{align}
$$


## 統一的な理解

[Ma+15] では、実はここまでに挙げた運動方程式は統一的に書けることを示している。

$$
\begin{align}
d\b\varphi &= \b f(\b\varphi) dt + N(\b 0, 2D(\b\varphi)dt)\\
\end{align}
$$

ここで、 $\varphi$ はモデルのパラメータ、$D$ は半正定値行列に値を持つ関数、 $\b f$ は次の表式である。

$$
\begin{align}
\b f(\b\varphi) &= \left[D(\b\varphi) + Q(\b\varphi)\right]\nabla_{\b\varphi}H(\b\varphi)
+ \b\Gamma(\b\varphi)\\
\b\Gamma_i(\b\varphi) &= \nabla_{\b\varphi}^T \left[ D_{i \cdot}(\b\varphi) + Q_{i \cdot}(\b\varphi)\right].
\end{align}
$$

ここで、 $Q$ は歪行列に値を持つ関数、$D_{i\cdot}, Q_{i\cdot}$ はそれぞれ、$D, Q$ の $i$ 行目を表す。
[Ma+15]のTheorem1では、$p(\b\varphi) \propto \exp(-H(\b\varphi))$ がこの運動方程式の定常分布であり、
さらに $D(\b\varphi)$ が（任意の $\b\varphi$ で？）正定値行列であるか、（系が？）エルゴード性を満たすならば、この定常分布が唯一であることを示している。

この式で、パラメータとなっているのは $H$, $D$, $Q$ である。
これらをそれぞれ以下のように設定すると、これまで解説してきたサンプリングアルゴリズムが得られる。

$$
\newcommand{SGNHTD}{
\left[
\begin{matrix} A\tilde{I} & 0\\0 & 0 \end{matrix}
\right]
}
$$

$$
\newcommand{SGNHTQ}{
\left[
\begin{array}{cc}
J & \begin{matrix} 0 \\ \b p / d \end{matrix}\\
\begin{matrix} 0 & -\b p^T /d \end{matrix} & 0
\end{array}
\right]
}
$$

$$
\newcommand{mSGNHTQ}{
\left[
\begin{array}{cc}
J & \begin{matrix} 0 \\ \mathrm{diag} (\b p)\end{matrix}\\
\begin{matrix} 0 & -\mathrm{diag} (\b p)\end{matrix} & 0
\end{array}
\right]
}
$$

||$\b\varphi$|$H(\b\varphi)$|$D$|$Q$|
|:-----|:-----:|:-----:|:-----:|:-----:|
|HMC|$(\b\theta, \b p)$|$U(\b\theta) + K(\b p)$|$0$|$J$|
|SGHMC|$(\b\theta, \b p)$|$U(\b\theta) + K(\b p)$|$A\tilde{I}$|$J$|
|SGLD|$\b\theta$|$U(\b\theta)$|$D$|$0$|
|SGNHT|$(\b\theta, \b p, \xi)$|$U(\b\theta) + K(\b p) + V(\xi)$|$\SGNHTD$|$\SGNHTQ$|
|mSGNHT|$(\b\theta, \b p, \b\xi)$|$U(\b\theta) + K(\b p) + V'(\b\xi)$|$\SGNHTD$|$\mSGNHTQ$|
|Santa|||||

ここで、$U(\b\theta) = -\log p(\b\theta|X) + \mathrm{const.}$, $K(\b p) = \frac{1}{2} \b p^T \b p$,
$V(\xi) = d(\xi - A)^2$ , $V'(\b\xi) = (\b\xi - A\b 1)^T(\b\xi - A\b 1)$ ,
$\tilde{I} = \begin{bmatrix}0&0 \\ 0 &I\end{bmatrix}$,
$J = \begin{bmatrix}0&-I\\I&0\end{bmatrix}$



## 参考文献

* HMC: [Neal11] Neal, R. M. (2011). MCMC using Hamiltonian dynamics. Handbook of Markov Chain Monte Carlo, 2.
* SGHMC: [Chen+14] Chen, T., Fox, E. B., & Guestrin, C. (2014). Stochastic gradient hamiltonian monte carlo. arXiv preprint arXiv:1402.4102.
* SGLD: [Welling+11] Welling, M., & Teh, Y. W. (2011). Bayesian learning via stochastic gradient Langevin dynamics. In Proceedings of the 28th International Conference on Machine Learning (ICML-11) (pp. 681-688).
* SGNHT: [Ding+14] Ding, N., Fang, Y., Babbush, R., Chen, C., Skeel, R. D., & Neven, H. (2014). Bayesian sampling using stochastic gradient thermostats. In Advances in Neural Information Processing Systems (pp. 3203-3211).
* mSGNHT: [Gan+15] Gan, Z., EDU, D., Chen, C., Henao, R., & Carlson, D. Scalable Deep Poisson Factor Analysis for Topic Modeling.
* Santa: [Chen+15c] Chen, C., Carlson, D., Gan, Z., Li, C., & Carin, L. (2015). Bridging the Gap between Stochastic Gradient MCMC and Stochastic Optimization. arXiv preprint arXiv:1512.07962.
* [Ma+15]: Ma, Y. A., Chen, T., & Fox, E. (2015). A complete recipe for stochastic gradient MCMC. In Advances in Neural Information Processing Systems (pp. 2899-2907).
* [Chen+15a]: Chen, C., Ding, N., & Carin, L. (2015). On the convergence of stochastic gradient MCMC algorithms with high-order integrators. In Advances in Neural Information Processing Systems (pp. 2269-2277).
* [Chen+15b]: Li, C., Chen, C., Fan, K., & Carin, L. (2015). High-Order Stochastic Gradient Thermostats for Bayesian Learning of Deep Models. arXiv preprint arXiv:1512.07662.
* [Leimkuhler+15]: Leimkuhler, B., & Shang, X. (2015). Adaptive Thermostats for Noisy Gradient Systems. arXiv preprint arXiv:1505.06889.

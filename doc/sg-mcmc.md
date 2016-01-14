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

* 細字はスカラー、太字はベクトルを表す
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

## 確率微分方程式

## Fokker-Planck方程式

確率過程 $X_t$ がSDE
$$
\begin{align}
dX_t = \mu(X_t, t)dt + \sigma(X_t, t)dW
\end{align}
$$
に従うとする。ここで、$\mu, \sigma$は決定的な関数で、$dW$はワイナー過程である。
この時、時刻$t$での$X_t$の確率分布は$p(x, t)$はPDE
$$
\frac{\partial}{\partial t}p(x, t) =
-\frac{\partial}{\partial x}\left[ \mu(x, t)p(x, t)\right]
+ \frac{1}{2}\frac{\partial^2}{\partial x^2}
\left[\sigma^2(x, t) p(x, t)\right]
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


## エルゴード性

## SDE

## 各MCMCサンプリングの運動方程式

前述した通り、質点の運動を支配する運動方程式を様々なものに設定することにより、
各サンプリングの手法が得られる。
では、具体的に各手法で利用される運動方程式を見ていこう。

### HMC

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


### SGHMC

まずは、SGHMCが利用する運動方程式を天下り的に書いてしまおう。

$$
\begin{align}
d\b\theta &= \b p dt \\
d\b p &= \nabla_{\b \theta} U(\b\theta)dt - A\b pdt + \sqrt{2A}d \b W
\end{align}
$$

ここで、 $\b W$ はパラメータ $\b\theta$ と同次元の（従って $\b p$と同次元の）ブラウン運動を行う確率過程である。 $$


SGHMCの説明の仕方は2通りあり、最初から運動方程式を天下り的に


### SGLD



### SGNHT, mSGNHT

さらにオプショナルなパラメータとして、 $\b \xi$を用意する。後々この$\b \xi$ は系の温度を調節するサーモスタットとしての役割を果たす：


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

SGHMCは2次のLangevin Dynamicsを離散化したものである。

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


theta = sample_from_prior()
for epoch in six.moves.range(EPOCH):
    p = numpy.random.randn(*theta.shape)
    for i in six.moves.range(0, args.N, args.batchsize):
        x = get_minibatch()
        p, theta = update(p, theta, x)
```

### SGLD




### SGHMC

### mSGNHT

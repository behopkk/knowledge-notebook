# EM算法详解 - 知乎
目录：

1.  摘要
2.  EM算法简介
3.  预备知识

3.1 极大似然估计

3.2 Jensen不等式

4\. EM算法详解

4.1 问题描述

4.2 EM算法推导流程

4.3 EM算法流程

5\. EM算法若干思考

5.1 对EM算法的初始化研究

5.2 EM算法是否一定收敛？

5.3 如果EM算法收敛，能否保证收敛到全局最大值？

6\. EM算法实例

7\. 对EM算法总结

**1\. 摘要**
----------

EM（Expectation-Maximum）算法也称期望最大化算法，曾入选“数据挖掘十大算法”中，可见EM算法在机器学习、数据挖掘中的影响力。EM算法是最常见的隐变量估计方法，在机器学习中有极为广泛的用途，例如常被用来学习高斯混合模型（Gaussian mixture model，简称GMM）的参数；隐式马尔科夫算法（HMM）、LDA主题模型的变分推断等等。本文就对EM算法的原理做一个详细的总结。

**【扩展阅读】** 数据挖掘中十大算法论文：

Wu X, Kumar V, Quinlan J R, et al. Top 10 algorithms in data mining\[J\]. Knowledge and information systems, 2008, 14(1): 1-37.

论文下载地址：[http://www.cs.uvm.edu/~icdm/algorithms/10Algorithms-08.pdf](https://link.zhihu.com/?target=http%3A//www.cs.uvm.edu/~icdm/algorithms/10Algorithms-08.pdf)

**2\. EM算法简介**
--------------

EM算法是一种迭代优化策略，由于它的计算方法中每一次迭代都分两步，其中一个为期望步（E步），另一个为极大步（M步），所以算法被称为EM算法（Expectation-Maximization Algorithm）。EM算法受到缺失思想影响，最初是为了解决数据缺失情况下的参数估计问题，其算法基础和收敛有效性等问题在Dempster、Laird和Rubin三人于1977年所做的文章《Maximum likelihood from incomplete data via the EM algorithm》中给出了详细的阐述。其基本思想是：首先根据己经给出的观测数据，估计出模型参数的值；然后再依据上一步估计出的参数值估计缺失数据的值，再根据估计出的缺失数据加上之前己经观测到的数据重新再对参数值进行估计，然后反复迭代，直至最后收敛，迭代结束。

**【扩展阅读】** 提出EM算法的论文

Dempster A P, Laird N M, Rubin D B. Maximum likelihood from incomplete data via the EM algorithm\[J\]. Journal of the royal statistical society. Series B (methodological), 1977: 1-38.

论文下载地址：[http://web.mit.edu/6.435/www/Dempster77.pdf](https://link.zhihu.com/?target=http%3A//web.mit.edu/6.435/www/Dempster77.pdf)

**3\. 预备知识**
------------

想清晰的了解EM算法推导过程和其原理，我们需要知道两个基础知识：“极大似然估计”和“Jensen不等式”。

**3.1 极大似然估计**

（1）问题描述

假如我们需要调查学校的男生和女生的身高分布 ，我们抽取100个男生和100个女生，将他们按照性别划分为两组。然后，统计抽样得到100个男生的身高数据和100个女生的身高数据。如果我们知道他们的身高服从正态分布，但是这个分布的均值 μ\\mu 和方差 δ2\\delta^2 是不知道，这两个参数就是我们需要估计的。

问题：我们知道样本所服从的概率分布模型和一些样本，我们需要求解该模型的参数。如图1所示。

![](https://pic3.zhimg.com/v2-f31b117fd7e067454e47a2aa9954516e_b.jpg)

图1：问题求解过程

我们已知的条件有两个：样本服从的分布模型、随机抽取的样本。我们需要求解模型的参数。根据已知条件，通过极大似然估计，求出未知参数。总的来说：极大似然估计就是用来估计模型参数的统计学方法。

（2）用数学知识解决现实问题

问题数学化：样本集 X\={x1,x2,...,xN},N\=100X=\\left\\{ x\_{1},x\_{2},...,x\_{N} \\right\\},N=100 。概率密度是：p(xi|θ)p(x\_{i}|\\theta) 抽到第i个男生身高的概率。由于100个样本之间独立同分布，所以同时抽到这100个男生的概率是它们各自概率的乘积，就是从分布是p(X|θ)的总体样本中抽取到这100个样本的概率，也就是样本集X中各个样本的联合概率，用下式表示：

L(θ)\=L(x1,x2,...,xn;θ)\=∏i\=1np(xi;θ),θ∈ΘL(\\theta)=L(x\_{1},x\_{2},...,x\_{n};\\theta)=\\prod\_{i=1}^{n}p(x\_{i};\\theta),\\theta\\in\\Theta

这个概率反映了在概率密度函数的参数是θ时，得到X这组样本的概率。 我们需要找到一个参数θ，使得抽到X这组样本的概率最大，也就是说需要其对应的似然函数L(θ)最大。满足条件的θ叫做θ的最大似然估计值，记为：

θ^\=argmaxL(θ) \\hat{\\theta}=argmaxL(\\theta)

（3）最大似然函数估计值的求解步骤

*   首先，写出似然函数：

L(θ)\=L(x1,x2,...,xn;θ)\=∏i\=1np(xi;θ),θ∈ΘL(\\theta)=L(x\_{1},x\_{2},...,x\_{n};\\theta)=\\prod\_{i=1}^{n}p(x\_{i};\\theta),\\theta\\in\\Theta

*   其次，对似然函数取对数：

l(θ)\=lnL(θ)\=ln∏i\=1np(xi;θ)\=∑i\=1nlnp(xi;θ)l(\\theta)=lnL(\\theta)=ln\\prod\_{i=1}^{n}p(x\_{i};\\theta)=\\sum\_{i=1}^{n}{lnp(x\_{i};\\theta)}

*   然后，对上式求导，另导数为0，得到似然方程。
*   最后，解似然方程，得到的参数值即为所求。

多数情况下，我们是根据已知条件来推算结果，而极大似然估计是已知结果，寻求使该结果出现的可能性最大的条件，以此作为估计值。

**3.2 Jensen不等式**

（1）定义

设f是定义域为实数的函数，如果对所有的实数x，f(x)的二阶导数都大于0，那么f是凸函数。

Jensen不等式定义如下：

如果f是凸函数，X是随机变量，那么： E\[f(X)\]≥f(E\[X\])E\\left\[ f(X) \\right\]\\geq f(E\\left\[ X \\right\]) 。当且仅当X是常量时，该式取等号。其中，E(X)表示X的数学期望。

注：Jensen不等式应用于凹函数时，不等号方向反向。当且仅当x是常量时，该不等式取等号。

（2）举例

![](https://pic1.zhimg.com/v2-c6ea0537af6cd4ceb25705c6ccc8575c_b.jpg)

图2：Jensen不等式

图2中，实线f表示凸函数，X是随机变量，有0.5的概率是a，有0.5的概率是b。X的期望值就是a和b的中值，从图中可以看到 E\[f(X)\]≥f(E\[X\])E\\left\[ f(X) \\right\]\\geq f(E\\left\[ X \\right\]) 成立。

**4\. EM算法详解**
--------------

**4.1 问题描述**

我们目前有100个男生和100个女生的身高，但是我们不知道这200个数据中哪个是男生的身高，哪个是女生的身高，即抽取得到的每个样本都不知道是从哪个分布中抽取的。这个时候，对于每个样本，就有两个未知量需要估计：

（1）这个身高数据是来自于男生数据集合还是来自于女生？

（2）男生、女生身高数据集的正态分布的参数分别是多少？

![](https://pic4.zhimg.com/v2-918917eb6432b766ffd86a194967d927_b.jpg)

图3：EM算法要解决的问题

那么，对于具体的身高问题使用EM算法求解步骤如图4所示。

![](https://pic2.zhimg.com/v2-7142d1ddbdb37514dad011d5c64fb121_b.jpg)

图4：身高问题EM算法求解步骤

（1）初始化参数：先初始化男生身高的正态分布的参数：如均值=1.65，方差=0.15

（2）计算每一个人更可能属于男生分布或者女生分布；

（3）通过分为男生的n个人来重新估计男生身高分布的参数（最大似然估计），女生分布也按照相同的方式估计出来，更新分布。

（4）这时候两个分布的概率也变了，然后重复步骤（1）至（3），直到参数不发生变化为止。

**4.2 EM算法推导流程**

对于n个样本观察数据 x\=(x1,x2,...xn)x=(x\_{1},x\_{2},...x\_{n}) ，找出样本的模型参数θ, 极大化模型分布的对数似然函数如下：

θ^\=argmax∑i\=1nlogp(xi;θ) \\hat{\\theta}=argmax\\sum\_{i=1}^{n}{logp(x\_{i};\\theta)}

如果我们得到的观察数据有未观察到的隐含数据 z\=(z1,z2,...,zn)z=(z\_{1},z\_{2},...,z\_{n}) ，即上文中每个样本属于哪个分布是未知的，此时我们极大化模型分布的对数似然函数如下：

θ^\=argmax∑i\=1nlogp(xi;θ)\=argmax∑i\=1nlog∑zip(xi,zi;θ) \\hat{\\theta}=argmax\\sum\_{i=1}^{n}{logp(x\_{i};\\theta)}=argmax\\sum\_{i=1}^{n}{log\\sum\_{z\_{i}}^{}{p(x\_{i},z\_{i};\\theta)}}

上面这个式子是根据 xix\_{i} 的边缘概率计算得来，没有办法直接求出θ。因此需要一些特殊的技巧，使用Jensen不等式对这个式子进行缩放如下：

∑i\=1nlog∑zip(xi,zi;θ)\=∑i\=1nlog∑ziQi(zi)p(xi,zi;θ)Qi(zi)(1)≥∑i\=1n∑ziQi(zi)logp(xi,zi;θ)Qi(zi)(2)\\begin{align\*} \\sum\_{i=1}^{n}{log\\sum\_{z\_{i}}^{}{p(x\_{i},z\_{i};\\theta)}}=\\sum\_{i=1}^{n}{log\\sum\_{z\_{i}}^{}{Q\_{i}(z\_{i})\\frac{p(x\_{i},z\_{i};\\theta)}{Q\_{i}(z\_{i})}}} (1) \\\\ \\geq \\sum\_{i=1}^{n}{\\sum\_{z\_{i}}^{}{Q\_{i}(z\_{i})log\\frac{p(x\_{i},z\_{i};\\theta)}{Q\_{i}(z\_{i})}}}(2) \\end{align\*}

*   (1)式是引入了一个未知的新的分布 Qi(zi)Q\_{i}(z\_{i}) ，分子分母同时乘以它得到的。
*   (2)式是由(1)式根据Jensen不等式得到的。由于 ∑ziQi(zi)log\[p(xi,zi;θ)Qi(zi)\]\\sum\_{z\_{i}}^{}{Q\_{i} (z\_{i})log\\left\[ \\frac{p(x\_{i},z\_{i};\\theta)}{Q\_{i}(z\_{i})} \\right\]} 为 p(xi,zi;θ)Qi(zi)\\frac{p(x\_{i},z\_{i};\\theta)}{Q\_{i}(z\_{i})} 的期望，且log(x)为凹函数，根据Jensen不等式可由(1)式得到(2)式。

上述过程可以看作是对 logl(θ)logl(\\theta) 求了下界（ l(θ)\=∑i\=1nlogp(xi;θ)l(\\theta)=\\sum\_{i=1}^{n}{logp(x\_{i};\\theta)} ）。对于Qi(zi)Q\_{i}(z\_{i}) 我们如何选择呢？假设θ已经给定，那么logl(θ)logl(\\theta)的值取决于Qi(zi)Q\_{i}(z\_{i})和 p(xi,zi)p(x\_{i},z\_{i}) 。我们可以通过调整这两个概率使(2)式下界不断上升，来逼近logl(θ)logl(\\theta)的真实值。那么如何算是调整好呢？当不等式变成等式时，说明我们调整后的概率能够等价于logl(θ)logl(\\theta)了。按照这个思路，我们要找到等式成立的条件。

如果要满足Jensen不等式的等号，则有：

p(xi,zi;θ)Qi(zi)\=c，c为常数\\frac{p(x\_{i},z\_{i};\\theta)}{Q\_{i}(z\_{i})}=c，c为常数

由于 Qi(zi)Q\_{i}(z\_{i}) 是一个分布，所以满足：∑zQi(zi)\=1\\sum\_{z}^{}{Q\_{i}(z\_{i})}=1，则 ∑zp(xi,zi;θ)\=c\\sum\_{z}^{}{p(x\_{i},z\_{i};\\theta)}=c 。

由上面两个式子，我们可以得到：

Qi(zi)\=p(xi,zi;θ)∑zp(xi,zi;θ)\=p(xi,zi;θ)p(xi;θ)\=p(zi|xi;θ)Q\_{i}(z\_{i})=\\frac{p(x\_{i},z\_{i};\\theta)}{\\sum\_{z}^{}{p(x\_{i},z\_{i};\\theta)}}=\\frac{p(x\_{i},z\_{i};\\theta)}{p(x\_{i};\\theta)}=p(z\_{i}|x\_{i};\\theta)

至此，我们推出了在固定其他参数θ后，Qi(zi)Q\_{i}(z\_{i})的计算公式就是后验概率，解决了Qi(zi)Q\_{i}(z\_{i})如何选择的问题。

如果Qi(zi)\=p(zi|xi;θ)Q\_{i}(z\_{i})=p(z\_{i}|x\_{i};\\theta)，则(2)式是我们包含隐藏数据的对数似然函数的一个下界。如果我们能最大化(2)式这个下界，则也是在极大化我们的对数似然函数。即我们需要最大化下式：

argmax∑i\=1n∑ziQi(zi)logp(xi,zi;θ)Qi(zi)argmax\\sum\_{i=1}^{n}{\\sum\_{z\_{i}}^{}{Q\_{i}(z\_{i})log\\frac{p(x\_{i},z\_{i};\\theta)}{Q\_{i}(z\_{i})}}}

上式也就是我们的EM算法的M步，那E步呢？解决了Qi(zi)Q\_{i}(z\_{i})如何选择的问题，这一步就是E步，该步建立了 l(θ)l(\\theta) 的下界。

**4.3 EM算法流程**

现在我们总结一下EM算法流程。

**输入：** 观察到的数据x\=(x1,x2,...xn)x=(x\_{1},x\_{2},...x\_{n})，联合分布 p(x,z;θ)p(x,z;\\theta) ，条件分布 p(z|x,θ)p(z|x,\\theta) ，最大迭代次数J。

**算法步骤：** 

（1）随机初始化模型参数θ的初值 θ0\\theta\_{0} 。

（2）j=1,2,...,J 开始EM算法迭代：

*   E步：计算联合分布的条件概率期望：

Q\_{i}(z\_{i})=p(z\_{i}|x\_{i},\\theta\_{j})

l(\\theta,\\theta\_{j})=\\sum\_{i=1}^{n}{\\sum\_{z\_{i}}^{}{Q\_{i}(z\_{i})log\\frac{p(x\_{i},z\_{i};\\theta)}{Q\_{i}(z\_{i})}}}

*   M步：极大化 l(\\theta,\\theta\_{j}) ,得到 \\theta\_{j+1} :

\\theta\_{j+1}=argmaxl(\\theta,\\theta\_{j})

*   如果\\theta\_{j+1} 已经收敛，则算法结束。否则继续进行E步和M步进行迭代。

**输出：** 模型参数θ。

**5\. EM算法若干思考**
----------------

**5.1 对EM算法的初始化研究**

上面介绍的传统EM算法对初始值敏感，聚类结果随不同的初始值而波动较大。总的来说，EM算法收敛的优劣很大程度上取决于其初始参数。

针对传统EM算法对初始值敏感的问题，许多研究者在EM算法初始化方面做了许多研究，下面我列出了一部分对EM算法初始化的研究，如果感兴趣可以自己查阅相关资料。

扩展阅读：

【1】Blömer J, Bujna K. Simple methods for initializing the EM algorithm for Gaussian mixture models\[J\]. CoRR, 2013.

论文下载地址：[https://pdfs.semanticscholar.org/7d4a/2da54c78cf62a2e8ea60e18cef35ab0d5e25.pdf](https://link.zhihu.com/?target=https%3A//pdfs.semanticscholar.org/7d4a/2da54c78cf62a2e8ea60e18cef35ab0d5e25.pdf)

【2】Chen F. An Improved EM algorithm\[J\]. arXiv preprint arXiv:1305.0626, 2013.

【3】Kwedlo W. (2013) A New Method for Random Initialization of the EM Algorithm for Multivariate Gaussian Mixture Learning. In: Burduk R., Jackowski K., Kurzynski M., Wozniak M., Zolnierek A. (eds) Proceedings of the 8th International Conference on Computer Recognition Systems CORES 2013. Advances in Intelligent Systems and Computing, vol 226. Springer, Heidelberg.

**5.2 EM算法是否一定收敛？**

**结论：EM算法可以保证收敛到一个稳定点，即EM算法是一定收敛的。** 

这里我直接给出结论，想看证明过程的可以看以下链接：

【1】[EM算法原理总结 - 刘建平Pinard - 博客园](https://link.zhihu.com/?target=http%3A//www.cnblogs.com/pinard/p/6912636.html)

【2】[EM算法学习(Expectation Maximization Algorithm)](https://link.zhihu.com/?target=https%3A//www.cnblogs.com/mindpuzzle/archive/2013/04/05/2998746.html)

**5.3 如果EM算法收敛，能否保证收敛到全局最大值？**

**结论：EM算法可以保证收敛到一个稳定点，但是却不能保证收敛到全局的极大值点，因此它是局部最优的算法，当然，如果我们的优化目标 l(\\theta,\\theta\_{l}) 是凸的，则EM算法可以保证收敛到全局最大值，这点和梯度下降法这样的迭代算法相同。** 

证明过程如下：

【1】[EM算法原理总结 - 刘建平Pinard - 博客园](https://link.zhihu.com/?target=http%3A//www.cnblogs.com/pinard/p/6912636.html)

**6\. EM算法实例**
--------------

EM算法是在数据不完全的情况下的参数估计。我们用一个投硬币的例子来解释EM算法流程。假设我们有A，B两枚硬币，其中正面朝上的概率分别为 \\theta\_{A},\\theta\_{B} ，这两个参数即为需要估计的参数。我们设计5组实验，每次实验投掷5次硬币，但是每次实验都不知道用哪一枚硬币进行的本次实验。投掷结束后，会得到一个数组 x=(x\_{1},x\_{2},...,x\_{5}) ，表示每组实验有几次硬币正面朝上，因此 0\\leq x\_{i}\\leq 5 。

如果我们知道每一组实验中 x\_{i} 是A硬币投掷的结果还是B硬币投掷的结果，我们可以很容易的估算出 \\theta\_{A},\\theta\_{B} ，只需要统计所有的实验中两个硬币分别正面朝上的次数，然后除以它们各自投掷的总次数。但是，数据不完全的意思在于，我们并不知道每一个数据是由哪一枚硬币产生的。EM算法可以解决类似这样的问题。

虽然我们不知道每组实验用的是哪一枚硬币，但如果我们用某种方法**猜测每组实验是哪个硬币投掷的**，我们就可以将数据缺失的估计问题转化成一个**最大似然问题和完整参数估计问题**。

假设5次试验的结果如下：

![](https://pic4.zhimg.com/v2-acfd32beb1acf383ece4e521ef3115a3_b.jpg)

图5：五次实验结果

首先，随机选取 \\theta\_{A},\\theta\_{B} 的初始值，比如 \\theta\_{A}=0.2,\\theta\_{B}=0.7 。EM算法的E步骤，是计算在当前的预估参数下，隐含变量（是A硬币还是B硬币）的每个值出现的概率。也就是给定\\theta\_{A},\\theta\_{B}和观测数据，计算这组数据出自A硬币的概率和这组数据出自B硬币的概率。对于第一组实验，3正面2反面。

*   如果是A硬币得到这个结果的概率为： 0.2^3\\times0.8^2=0.00512
*   如果是B硬币得到这个结果的概率为： 0.7^3\\times0.3^2=0.03087

因此，第一组实验结果是A硬币得到的概率为：0.00512 / (0.00512 + 0.03087)=0.14，第一组实验结果是B硬币得到的概率为：0.03087/ (0.00512 + 0.03087)=0.86。整个5组实验的A,B投掷概率如下：

![](https://pic4.zhimg.com/v2-11bb3207197a4bc8adfd1503278a429f_b.jpg)

图6：A硬币、B硬币的概率分布

根据隐含变量的概率，可以计算出两组训练值的期望。依然以第一组实验来举例子：3正2反中，如果是A硬币投掷的结果：0.14\*3=0.42个正面和0.14\*2=0.28个反面；如果是B硬币投掷的结果：0.86\*3=2.58个正面和0.86\*2=1.72个反面。

5组实验的期望如下表：

![](https://pic4.zhimg.com/v2-340bf3f9a0eb8b29564653d5bb5bedef_b.jpg)

图7：五组实验的期望

通过计算期望，我们把一个有隐含变量的问题变化成了一个没有隐含变量的问题，由上表的数据，估计\\theta\_{A},\\theta\_{B}变得非常简单。

\\theta\_{A} =4.22/(4.22+7.98)=0.35

\\theta\_{B} =6.78/(6.78+6.02)=0.5296875

这一步中，我们根据E步中求出的A硬币、B硬币概率分布，依据最大似然概率法则去估计\\theta\_{A},\\theta\_{B}，被称作M步。

最后，一直循环迭代E步M步，直到\\theta\_{A},\\theta\_{B}不更新为止。

**7\. 对EM算法总结**
---------------

EM算法是迭代求解最大值的算法，同时算法在每一次迭代时分为两步，E步和M步。一轮轮迭代更新隐含数据和模型分布参数，直到收敛，即得到我们需要的模型参数。

一个最直观了解EM算法思路的是K-Means算法。在K-Means聚类时，每个聚类簇的质心是隐含数据。我们会假设K个初始化质心，即EM算法的E步；然后计算得到每个样本最近的质心，并把样本聚类到最近的这个质心，即EM算法的M步。重复这个E步和M步，直到质心不再变化为止，这样就完成了K-Means聚类。当然，K-Means算法是比较简单的，高斯混合模型（GMM）也是EM算法的一个应用。

**Reference：** 
---------------

【1】[EM算法（Expectation Maximization Algorithm）详解](https://link.zhihu.com/?target=https%3A//blog.csdn.net/zhihua_oba/article/details/73776553)

【2】[EM算法原理总结 - 刘建平Pinard - 博客园](https://link.zhihu.com/?target=http%3A//www.cnblogs.com/pinard/p/6912636.html)

【3】[机器学习系列之EM算法 - emma\_zhang - 博客园](https://link.zhihu.com/?target=https%3A//www.cnblogs.com/Gabby/p/5344658.html)

【4】[EM算法学习(Expectation Maximization Algorithm)](https://link.zhihu.com/?target=https%3A//www.cnblogs.com/mindpuzzle/archive/2013/04/05/2998746.html)

【5】《机器学习》周志华著。

【6】[如何感性地理解EM算法？](https://link.zhihu.com/?target=https%3A//www.jianshu.com/p/1121509ac1dc)

【7】[EM算法的学习笔记 - CSDN博客](https://link.zhihu.com/?target=https%3A//blog.csdn.net/littleorange6/article/details/74218025)

【8】[https://www.cmi.ac.in/~madhavan/courses/datamining12/reading/em-tutorial.pdf](https://link.zhihu.com/?target=https%3A//www.cmi.ac.in/~madhavan/courses/datamining12/reading/em-tutorial.pdf)

我的个人**微信公众号**：**Microstrong**  
**微信公众号ID:MicrostrongAI**  
公众号介绍：Microstrong(小强)同学主要研究机器学习、深度学习、图像处理、计算机视觉相关内容，分享在学习过程中的读书笔记！期待您的关注，欢迎一起学习交流进步！  
个人博客：
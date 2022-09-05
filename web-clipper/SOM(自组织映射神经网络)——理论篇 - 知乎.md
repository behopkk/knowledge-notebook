# SOM(自组织映射神经网络)——理论篇 - 知乎
SOM介绍
-----

自组织映射(Self-organizing map, SOM)通过学习输入空间中的数据，生成一个低维、离散的映射(Map)，从某种程度上也可看成一种降维算法。

SOM是一种**无监督**的人工神经网络。不同于一般神经网络基于损失函数的反向传递来训练，它运用**竞争学习**(competitive learning)策略,依靠神经元之间互相竞争逐步优化网络。且使用近邻关系函数(neighborhood function)来维持输入空间的拓扑结构。

维持输入空间的拓扑结构：意味着 二维映射包含了数据点之间的相对距离。输入空间中相邻的样本会被映射到相邻的输出神经元。

由于基于无监督学习,这意味着训练阶段不需要人工介入(即不需要样本标签)，我们可以在不知道类别的情况下，对数据进行聚类；可以识别针对某问题具有内在关联的特征。

特点归纳：

*   神经网络，竞争学习策略
*   无监督学习，不需要额外标签
*   非常适合高维数据的可视化，能够维持输入空间的拓扑结构
*   具有很高的泛化能力，它甚至能识别之前从没遇过的输入样本  
    

网上有很多开源的实现，这里列了一份我在github上搜集的清单：

![](https://pic2.zhimg.com/v2-a75a5f3e75b63f3160bacc883794b871_b.jpg)

为了避免空谈理论导致晦涩难懂，我们以miniSom库的部分实现为例，辅助理解训练过程中的一些公式。

网络结构结构
------

SOM的网络结构有2层：输入层、输出层(也叫竞争层)

![](https://pic2.zhimg.com/v2-c6ffeb31057b724331a528526a222689_b.jpg)

输入层神经元的数量是由输入向量的维度决定的，一个神经元对应一个特征

SOM网络结构的区别主要在竞争层：可以有1维、2维(最常见的)

> 竞争层也可以有更高的维度。不过出于可视化的目的，高维竞争层用的比较少

其中，二维平面有2种平面结构：

*   Rectangular
*   Hexagonal

![](https://pic1.zhimg.com/v2-758525667f484437503d94eb07563950_b.jpg)

竞争层SOM神经元的数量决定了最终模型的粒度与规模；这对最终模型的准确性与泛化能力影响很大。

一条经验公式：

竞争层最少节点数量 = 5\\sqrt{N}

N：训练样本的个数

> 如果是正方形输出层，边长等于 竞争层节点数再开一次根号，并向上取整就行

训练计算过程
------

第一步：与其他神经网络相同，需要将Weighs初始化为很小的随机数

第二步：随机取一个 输入样本Xi

第三步：

1.  遍历竞争层中每一个节点：计算Xi与节点之间的**相似度**(通常使用欧式距离)
2.  选取距离最小的节点作为**优胜节点**(winner node)，有的时也叫**BMU**(best matching unit)

第四步：根据邻域半径σ(sigma)确定**优胜邻域**将包含的节点；并通过neighborhood function计算它们各自更新的幅度(基本思想是：越靠近优胜节点，更新幅度越大；越远离优胜节点，更新幅度越小)

第五步：更新优胜邻域内节点的Weight：

W\_v(s+1) = W\_v(s) + θ(u,v,s) · α(s) · (D(t) - W\_v(s))

> θ(u,v,s)是对更新的约束，基于离BMU的距离 即neighborhood function的返回值  
> W\_v(s)是节点v当前的Wight

第六步：完成一轮迭代(迭代次数+1)，返回第二步，直到满足设定的迭代次数

![](https://pic4.zhimg.com/v2-f52b70ceb4be67a91ebd807236a27fbf_b.gif)

如gif所演示的训练过程，优胜节点更新后会更靠近输入样本Xi在空间中的位置。优胜节点拓扑上的邻近节点也类似地被更新。这就是SOM网络的竞争调节策略。

neighborhood function
---------------------

neighborhood函数用来确定**优胜节点对其近邻节点的影响强弱**，即优胜邻域中每个节点的更新幅度。最常见的选择是高斯函数，它可以表征优胜邻域内，影响强弱与距离的关系。

```text
g = neighborhood_func(winner, sigma) 
w_new = learning_rate * g * (x-w)
```

winner是优胜节点在输出平面的坐标

sigma确定邻域范围，sig越大，邻域范围越大

sigma的取值范围：

*   sigma必须大于0，否则没有神经元会被更新;
*   且sigma不能大于2维输出平面的边长

由于高斯函数包含指数项，它的计算量非常大。我们可以用bubble函数很好地去近似估计高斯，bubble在优胜节点的邻域范围内是个常数，因此，邻域内的所有神经元更新的幅度是相同的。由sigma唯一确定有多少神经元参与更新。

Bubble是函数很好地在计算成本与高斯核的准确性之间平衡。

优胜邻域内节点更新程度可视化：
---------------

我们以尺寸为5X5的竞争层为例，假设中心节点是优胜节点

```text
from numpy import outer, logical_and
size = 5
neigx = arange(size)
neigy = arange(size)
```

高斯近邻函数：
-------

![](https://pic2.zhimg.com/v2-15921d780f110a0787e1d74843ad918d_b.jpg)

高斯函数：是连续的，因此sigma的有效取值范围也是连续的

```text
def gaussian(c, sigma):
    """Returns a Gaussian centered in c."""
    d = 2*pi*sigma*sigma
    ax = exp(-power(neigx-c[0], 2)/d)
    ay = exp(-power(neigy-c[1], 2)/d)
    return outer(ax, ay)  # the external product gives a matrix

out = gaussian((2,2),1)
```

![](https://pic3.zhimg.com/v2-1b3f5cd5b91c3699ae81a2aafc64bc7e_b.jpg)

sigma = 1

当选sigma设为1时，所有的节点都有一定的更新幅度，中心优胜节点是1，越远离优胜节点，更新幅度越低，

sigma其实是在控制这种随距离衰退的程度

![](https://pic4.zhimg.com/v2-88c164e0077f401d6254d87fd8ba1bc3_b.jpg)

sigma = 0.1

当sigma取值很小时，只有优胜节点更新幅度是1，其余几乎都接近0

![](https://pic3.zhimg.com/v2-bc9247746fe262a5a7619cd290f3ed46_b.jpg)

sigma = 4

当sigma取值较大时，衰退的程度很慢，即使是边缘的节点，也有较大的更新幅度

Bubble近邻函数：
-----------

![](https://pic3.zhimg.com/v2-08c5620417f3bab430299904aed5b29a_b.jpg)

Bubble函数：只要是在优胜邻域内的神经元，更新系数都是相同的

```text
(X > winner_x - sigma) & (X < winner_x + sigma)
```

因此, sigma的有效取值是离散的：

0.5：仅优胜节点

1.5：周围一圈

2.5：周围2圈

> (1,2\] 之间的取值，效果都一样 都是周围一圈

```text
def bubble( c, sigma):
    """Constant function centered in c with spread sigma.
    sigma should be an odd value.
    """
    ax = logical_and(neigx > c[0]-sigma,
                     neigx < c[0]+sigma)
    ay = logical_and(neigy > c[1]-sigma,
                     neigy < c[1]+sigma)
    return outer(ax, ay)*1.

out = bubble((2,2),sigma= 1.5)
```

![](https://pic3.zhimg.com/v2-c35b63efe7f25053e0db722851f8dba6_b.jpg)

sigma=1.5

图中可以看出，中心为优胜节点，当sigma=1.5时，周围一圈的节点都处于优胜邻域中

且它们的更新幅度相同，都等于1

```text
out = bubble((2,2),sigma= 0.5)
```

![](https://pic3.zhimg.com/v2-650b85acc55161a91f7d316eb8e3c0ae_b.jpg)

sigma = 0.5

学习率α、邻域范围σ随时间衰减
---------------

SOM网络的另一个特点是，学习率和邻域范围随着迭代次数会逐渐衰减

衰减函数 decay func
---------------

最常用的decay funciton是 \\frac{1}{1+t/T}

sigma(t) = sigma / (1 + t/T)

learning\_rate(t) = learning\_rate / (1 + t/T)

> t 代表当前迭代次数  
> T = 总迭代次数 / 2

```as3
import numpy as np
import matplotlib.pyplot as plt

num_iteration = 1000
T = num_iteration/2

t = np.arange(0,num_iteration,10)
decay_rate = 1 / (1+t/(T))
plt.plot(decay_rate)
```

使用matplot画图，可以看到这是一条单调递减的曲线

![](https://pic3.zhimg.com/v2-310fa48808963de721ce29a2cd8a3d1a_b.jpg)

数据预处理
-----

由于SOM是基于距离的算法，所以输入矩阵X中的类别型特征必须进行One-Hot编码

可以考虑进行标准化(均值为0，标准差为1)；这样有助于使每个特征对于计算相似度的贡献相同

Initilization
-------------

三种初始化方法：

*   Random initialization：适用于对输入数据有很少或没有任何先验知识  
    
*   Initialization using initial samples：优点是初始时刻，网络节点就与输入数据的拓扑结构很相似  
    
*   Linear initialization(PCA)：让网络向输入数据能力最大的方向延伸

其实在miniSom创建模型后，就已经进行Random initialization了；som.random\_weights\_init()其实是"Initialization using initial samples"

起初我以为这个是Random initialization....

> Principal component initialization is preferable (in dimension one) if the principal curve approximating the dataset can be univalently and linearly projected on the first principal component (quasilinear sets). For nonlinear datasets, however, random initiation performs better.

很多论文做过实验，PCA Initialization并不是在任何情况下都优于 random initiation，取决于输入数据本身的性质。推荐调参的时候，两种都试试吧~

* * *

可视化
---

SOM本质是在逼近输入数据的概率密度，以下几种工具能非常好的可视化训练好的SOM网络

*   U-Matrix
*   Component Plane

U-Matrix(unified distance matrix)
---------------------------------

U-matrix包含每个节点与它的邻居节点(在输入空间)的欧式距离：

*   在矩阵中较小的值表示该节点与其邻近节点在输入空间靠得近
*   在矩阵中较大的值表示该节点与其邻近节点在输出空间离得远

因此，U-matrix可以看作输入空间中数据点概率密度在二维平面上的映射

![](https://pic3.zhimg.com/v2-9ead3a96c63906ae36ac5d077d5190fe_b.jpg)

彩色

![](https://pic3.zhimg.com/v2-c87b623580dbaaff8ac9a4c7a227870a_b.jpg)

黑白

通常使用Heatmap来可视化U-matrix，且用颜色编码(数值越大，颜色越深)

在图上，浅色区域可以理解为 簇的簇心，深色区域可以理解为分隔边界

```text
# miniSOM API
som.distance_map()
```

Component Plane
---------------

通过component plane，能够可视化相关变量或者额外变量)的分布

![](https://pic2.zhimg.com/v2-7aaa2e5cbb3672e6eeabe9bf544b09d1_b.jpg)

Component plane可以理解成SOM网络的切片版本。每一个component plane包含了一个输入特征的相对分布。

在这种图中，深色表示相对小的取值，浅色表示相对大的取值。

通过比较component planes，我们可以看出两个component的相关信。如果看上去类似，那么它们强相关

解释性
---

有两种方式来解释SOM：

解释一：在训练阶段，整个邻域中节点的权重往相同的方向移动。因此，SOM形成语义映射，其中相似的样本被映射得彼此靠近，不同的样本被分隔开。这可以通过U-Matrix来可视化

解释二：另一种方法是将神经元权重视为输入空间的指针。它们形成对训练样本分布的离散近似。更多的神经元指向的区域，训练样本浓度较高；而较少神经元指向的区域，样本浓度较低。

![](https://pic3.zhimg.com/v2-0124198515f6d71758d5091206cb64d2_b.jpg)

逼近输入空间的过程

![](https://pic4.zhimg.com/v2-9b616abccc6bb844ab37c832a3fb0137_b.jpg)

总结：
---

本篇介绍了SOM算法的基本理论，另外还有一篇关于SOM具体的应用方法以及效果，有详细的案例和代码。

案例(实战)篇传送门：

附录：miniSomAPI参考
---------------

创建网络：创建时就会随机初始化网络权重 som = minisom.MiniSom(size,size,Input\_size, sigma=sig,learning\_rate=learning\_rate, neighborhood\_function='gaussian')

som.random\_weights\_init(X\_train)：随机选取样本进行初始化 som.pca\_weights\_init(X\_train)：PCA初始化

> 三种初始化方式，选一种即可  

两种训练方式：

som.train\_batch(X\_train, max\_iter, verbose=False)：逐个样本进行迭代 som.train\_random(X\_train, max\_iter, verbose=False)：随机选样本进行迭代

训练好的模型：

*   som.get\_weights()： Returns the weights of the neural network
*   som.distance\_map()：Returns the distance map of the weights
*   som.activate(X)： Returns the activation map to x 值越小的神经元，表示与输入样本 越匹配
*   som.quantization(X)：Assigns a code book 给定一个 输入样本，找出该样本的优胜节点，然后返回该神经元的权值向量(每个元素对应一个输入单元)  
    
*   som.winner(X)： 给定一个 输入样本，找出该样本的优胜节点 格式：输出平面中的位置  
    
*   som.win\_map(X)：将各个样本，映射到平面中对应的位置 返回一个dict { position: samples\_list }
*   som.activation\_response(X)： 返回输出平面中，各个神经元成为 winner的次数 格式为 1个二维矩阵
*   quantization\_error(量化误差)： 输入样本 与 对应的winner神经元的weight 之间的 平方根
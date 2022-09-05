# [论文笔记]——BYOL：无需负样本就可以做对比自监督学习(DeepMind) - 知乎
Reference
---------

paper：[Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2006.07733)

github源码：[https://github.com/TengdaHan/CoCLR](https://link.zhihu.com/?target=https%3A//github.com/google-research/simclr)

[Understanding self-supervised and contrastive learning with "Bootstrap Your Own Latent" (BYOL)](https://link.zhihu.com/?target=https%3A//generallyintelligent.ai/understanding-self-supervised-contrastive-learning.html)

[如何评价Deepmind自监督新作BYOL？ - 田永龙的回答 - 知乎](https://www.zhihu.com/question/402452508/answer/1294166177)

[计算机视觉 - 自监督学习 - Bootstrap Your Own Latent (BYOL, DeepMind)\_哔哩哔哩 (゜-゜)つロ 干杯~-bilibili](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV15D4y1d7GQ)

Introduction
------------

这篇paper讨论了最近火热的contrastive loss中负样本存在的必要性，在没有使用负样本的情况下在ImageNet达到了sota的水平。为什么这样的架构不会坍缩到trivial solution，反而能学到很好的representation呢？

其实BYOL实际上也是contrastive learning的思想，只是它在**隐性地在做contrastive learning。** 

Abstract
--------

首先还是简要介绍一下BYOL和SimCLR，MOCO等模型的区别。SimCLR提出了**nonlinear projection head**的概念，MOCO和SimCLR的区别主要是：

1.  **nonlinear projection head里面没有BN**；
2.  SimCLR提取两张经过augmentation的图片的CNN网络是相同的，而MOCO训练的时候则是有两个不同的网络， 其中一个网络根据另一个网络的parameters慢慢进行更新，且这个网络会提供一个**memory bank**，会提供更多大量的negative sample。因为contrastive learning是非常依仗negative sample的数量，所以**negative sample数量越多，contrastive task越难，最终提取到的representation就越好。** 

而BYOL则是在MOCO的基础上直接去掉了negative sample。如下图所示，前面的结构都和MOCO相同（除了 gθg\_\\theta 的结构，加入了BN，后面会提到），两个不同的网络分别为上面的online网络和target网络。不同之处是online网络在经过projection得到 zθz\_\\theta 后加了一个predictor（由1层或2层FC组成），然后用这个predictor来预测target网络得到的 zξ′z\_\\xi' ，相当于一个回归任务，loss函数采用MSE（值得注意的是，zθz\_\\theta和 zξ′z\_\\xi' 都经过了L2 normalize）：

![](https://pic1.zhimg.com/v2-c2017cb6f2758d0629f9a66dcd7c08f4_b.jpg)
![](https://pic2.zhimg.com/v2-7edffcfa0921027f9405512ae4308181_b.jpg)

这显然是非常令人震惊的，因为这个网络结构中**只是让positive pair提取到的feature尽可能的相似，而没有使用到negative samples。** 一般的contrastive loss都是从所有试图欺骗我们的负样本中选出唯一正确的那个y，这个问题通常就表达成一个softmax-CE loss，这个loss可以进一步分解成两个部分（具体推导可以看[ICML2020的论文](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2005.10242)）：

![](https://pic1.zhimg.com/v2-18cb8f8bd7f2c153bc15d2f149c8f078_b.jpg)

> 这个意思就是说，我可以把contrastive loss分解成两个部分，第一部分叫做**alignment**，就是希望positive pair的feature接近，第二部分叫做**uniformity**，就是希望所有点的feature尽量均匀的分部在unit sphere上面，都挺好理解的吧？这两部分理论上是都需要的，假如只有alignment，没有uniformity，那就很容易都坍缩到０，就是退化解。

所以BYOL就是去掉uniformity，只保留了alignment。这听起来似乎不科学，因为模型很容易学到trivial solution：就是使online网络和target网络永远都输出同样的constant。所以模型为什么会work呢？看了一些大佬的分享（详见Reference），总结大概有以下几点：EMA，predictor，BN。

EMA
---

target网络的参数 ξ\\xi 是以EMA（exponential moving average）的方式根据online网络的参数 θ\\theta 和decay rate τ\\tau 来进行更新的：

![](https://pic2.zhimg.com/v2-021b1fdecf0131f9f59807d4956563e9_b.jpg)

而**EMA可能在帮助悄悄scatter feature**。两个网络初始化的随机性本身就把feature scatter开来了，而当τ取值比较大时，target网络的更新是比较慢的，这或许能保持这种分散开的特性，使得online network的不同图片在regress的时候target是不同的，进而帮助阻止模型塌陷。

换句话说，**由于EMA这种方式，可以有效保持两个网络是不一样的**，所以如果模型想学trivial solution（online网络和target网络永远都输出同样的constant）是比较难的。

实验部分也证明了，如下表所示。对比第三条和最后一条可以看出来EMA非常重要，没有使用EMA效果比随机常数网络的效果好不了多少。而且τ的参数选取也非常重要，可以看到0.999和0.99就会导致3%的差别。

![](https://pic3.zhimg.com/v2-adf2163fa9f9e0cff5e8844a28c99d6e_b.jpg)

Predictor
---------

如下图的实验所示，online网络后面的predictor也非常重要，虽然它只是1或2层全连接层。

> **我觉得它给了online network很好的灵活性，就是online network的feature出来后不用完完全全去match那个EMA模型，只需要再经过一个predictor去match就好了。然后这个predictor的weight是不会update到EMA的，相当于一个允许online和EMA feature不同的缓冲地带。** 

**“灵活性”**这个词真的很好的描述了predictor的作用。

而且最后regress的feature是**L2-normalized**的，可能这一步可以**防止MSE把feature的scale都拉倒接近０**。L2-normalized后这个MSE loss其实就变成**把特征的方向对上，而scale就不在乎了。** 

![](https://pic4.zhimg.com/v2-4e235c9cabd979fc6de5ef3a5bb95103_b.jpg)

Batch Normalization
-------------------

前面提到，BYOL相比MOCO一个比较大差异就是**nonlinear projection gθg\_\\theta 的结构中加入了BN**，如下图所示：

![](https://pic1.zhimg.com/v2-aa92f3b64fcb5042e5f8c2820f52fb28_b.jpg)
![](https://pic2.zhimg.com/v2-5a969ac6579b22355d2261566b8dc4f1_b.jpg)

我们知道，BN实际上就是规范化一个batch的分布。得到的mean和variance都和batch里面所有的image有关，所以**BN相当于一个隐性的contrastive learning**：每一个image都和batch的mean做contrastive learning。

> Why batch normalization is implicit contrastive learning: **all examples are compared to the mode**

从uniformity的角度来理解，**BN会把不同点的特征scatter开来，就等于默默地在做dispersion**。

> **Mode collapse is prevented precisely because all samples in the mini-batch cannot take on the same value after batch normalization**.

正是因为在BN之后，batch中的所有样本都不能采用相同的值，所以可以防止模型坍塌。

实际上，Reference中的blog作者对BYOL进行了复现。如下图所示，没有BN的BYOL的效果和random baseline一样。

![](https://pic3.zhimg.com/v2-ba15ef0e66c91c3bcf101c64bbee2e42_b.jpg)

下图是对online网络和target网络做cosine similarity，可以看到没有BN的BYOL最终学习到的两个网络相似度是非常高的。

![](https://pic1.zhimg.com/v2-a15ad61f2d7ed31f244e9aad13f52ba0_b.jpg)

Experiment
----------

最后再看一些其它的实验。

首先是在imageNet上的linear evaluation。可以看到模型都达到了soat效果，可以看到图b最后一行，Top-1的accuracy能打到79.6%，这应该算是非常惊人的效果。因为在pretrain的过程中是完全没有用到label的，用到label时只是在训练一个linear classifier，准确率能接近80%说明学到的representation是非常好的。

![](https://pic4.zhimg.com/v2-b10265a270178bf31babf05c5195b713_b.jpg)

只利用一小部分的data对整个BYOL网络进行fine-tune，可以看到最终的效果也是非常高的，要比有监督的pretrain效果高很多。（imagenet中每个class有1000张image，所以1%相当于每个class仅用到了10张）

![](https://pic1.zhimg.com/v2-4f2d6b55e170d82f8f57c136252af8d4_b.jpg)

下面这个实验显示了BYOL对数据的迁移能力：先在imagenet上做pretrain，然后在其他dataset上进行fine-tune。supervised-IN表示的是有监督的在imagenet上做pre-train，可以看到BYOL和supervised-IN最终的效果都是comparable的。

![](https://pic4.zhimg.com/v2-c36026581fff45f302466463bbffd8db_b.jpg)

在检测，分割，深度估计等其他下游任务上的performance也都超过了有监督的方法。

![](https://pic2.zhimg.com/v2-af4865d5727c3afb23ce7391b65a6789_b.jpg)

最后是BYOL对一些hyperparamter的敏感度。可以看到相比SimCLR，随着batch size的减小和对augmentation方法的消除，BYOL的performance也在降低，但是下降程度是要小于SimCLR的。证明了BYOL对这些参数的敏感度更低。

![](https://pic4.zhimg.com/v2-e69db3bd42acc9a7539ce39ea303d32b_b.jpg)

Conclusion
----------

![](https://pic1.zhimg.com/v2-18cb8f8bd7f2c153bc15d2f149c8f078_b.jpg)

contrastive loss分解成两个部分，第一部分叫做**alignment**，就是希望positive pair的feature接近，第二部分叫做**uniformity**，就是希望所有点的feature尽量均匀的分部在unit sphere上面

假如只有alignment，没有uniformity，那就很容易都坍缩到０，就是trivial solution。而总结下来来看，BYOL实际上就是通过EMA，predictor，BN等方式在隐性地进行uniformity：

*   EMA尽量**保持了两个网络的参数不同**：两个网络初始化的随机性本身就把feature scatter开来了，而target网络更新慢或许能保持这种分散开的特性，使得online network的不同图片在regress的时候target是不同的，进而帮助阻止模型塌陷；
*   **predictor给了online network很好的灵活性**，online网络的feature出来后不用完完全全去match那个EMA模型，只需要再经过一个predictor去match就好了。然后这个predictor的weight是不会update到EMA的，相当于一个允许online和EMA feature不同的缓冲地带；
*   最终regress的feature是L2-normalized的，可能这一步可以防止MSE把feature的scale都拉倒接近０。L2-normalized后这个MSE loss其实就**变成把特征的方向对上，而scale就不在乎了。** 
*   **BN相当于一个隐性的contrastive learning**：每一个image都和batch的mean做contrastive learning。BN会把不同点的特征scatter开来，就等于默默地在做dispersion。正是因为在BN之后，batch中的所有样本都不能采用相同的值，所以可以防止模型坍塌；

新的进展又发现，BYOL不需要BN： [BYOL works even without batch statistics](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2010.10241)。还是非常有趣的，因为Reference中的blog在复现实验中指出BN是非常重要的，不知道这篇论文是怎么做的，后面看了有收获再做笔记。
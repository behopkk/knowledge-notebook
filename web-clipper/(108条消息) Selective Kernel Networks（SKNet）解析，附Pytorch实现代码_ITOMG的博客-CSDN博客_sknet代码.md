# (108条消息) Selective Kernel Networks（SKNet）解析，附Pytorch实现代码_ITOMG的博客-CSDN博客_sknet代码
       在神经科学界，视皮层[神经元](https://so.csdn.net/so/search?q=%E7%A5%9E%E7%BB%8F%E5%85%83&spm=1001.2101.3001.7020)的感受野大小受刺激的调节，即对不同刺激，感受野的大小应该不同。目前很多卷积神经网络的相关工作都只是通过改进网络的空间结构来优化模型，如[Inception](https://so.csdn.net/so/search?q=Inception&spm=1001.2101.3001.7020)模块通过引入不同大小的卷积核来获得不同感受野上的信息。或inside-outside网络参考空间上下文信息，等等。但在构建传统CNN时一般在同一层只采用一种[卷积核](https://so.csdn.net/so/search?q=%E5%8D%B7%E7%A7%AF%E6%A0%B8&spm=1001.2101.3001.7020)，也就是说对于特定任务特定模型，卷积核大小是确定的，很少考虑多个卷积核的作用。

![](https://imgconvert.csdnimg.cn/aHR0cDovL2ltZy5tcC5pdGMuY24vdXBsb2FkLzIwMTcwODAyLzkyYmM5Y2NiMzg2YzRlM2M5M2EwZTA3ZDg0YmQ3MDk5X3RoLmpwZw?x-oss-process=image/format,png)

那么我们针对标红的两点提出两个问题：

**1.那么是否可以在网络中加入特征维度信息呢？**

基于这一想法，诞生了SENet（Squeeze-and-Excitation Networks），2017ImageNet分类比赛冠军模型，论文发表于2018年CVPR，目前是CVPR2018引用量最多的论文。Paper : [Sequeeze-and-excitation networks](https://arxiv.org/abs/1709.01507)

**2.那么是否可以使网络可以根据输入信息的多个尺度自适应的调节接受域大小呢？**

SKNet(Selective Kernel Networks)是2019CVPR的一篇文章。也是本篇文章着重要讲解的。Paper : [Selective Kernel Networks](https://arxiv.org/abs/1903.06586?context=cs)

        SENet和SKNet，听这名字就知道，这两个是兄弟。SENet提出了Sequeeze and Excitation block，而SKNet提出了Selective Kernel Convolution. 二者都可以很方便的嵌入到现在的网络结构，比如ResNet、Inception、ShuffleNet，实现精度的提升。但因为之前写的一篇博客[《CBAM: Convolutional Block Attention Module论文代码解析》](https://blog.csdn.net/ITOMG/article/details/88804936)，对SENet已经有了一个比较详细的讲述，故在此不再展开讲，感兴趣的读者朋友可以看一下这篇博客。

       文章关注点主要是不同大小的感受野对于不同尺度的目标有不同的效果，而我们的目的是使得网络可以自动地利用对分类有效的感受野捕捉到的信息。所以如何做到这一点呢？因为网络训练好了之后参数就固定了，大多数的设计都是多尺度的信息就直接全部应用了，在segmentation领域尤为明显。  
       为了解决这个问题，作者提出了一种在[CNN](https://so.csdn.net/so/search?q=CNN&spm=1001.2101.3001.7020)中对卷积核的动态选择机制，该机制允许每个神经元根据输入信息的多尺度自适应地调整其感受野（卷积核）的大小。其灵感来源是，我们在看不同尺寸不同远近的物体时，视觉皮层神经元接受域大小是会根据刺激来进行调节的。具体是设计了一个称为选择性内核单元（SK）的构建块，其中，多个具有不同内核大小的分支在这些分支中的信息引导下，使用SoftMax进行融合。由多个SK单元组成SKNet，SKNet中的神经元能够捕获不同尺度的目标物体。

![](https://img-blog.csdnimg.cn/20190511162735863.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0lUT01H,size_16,color_FFFFFF,t_70)

       这个网络主要分为Split，Fuse，Select三个操作。Split算子产生多条不同核大小的路径，上图中的模型只设计了两个不同大小的卷积核，实际上可以设计多个分支的多个卷积核。fuse运算符结合并聚合来自多个路径的信息，以获得用于选择权重的全局和综合表示。select操作符根据选择权重聚合不同大小内核的特征图。

（1）Split：如模型图所示，使用多个卷积核对 X 进行卷积，以形成多个分支。

       这里的Split是指对输入向量X进行不同卷积核大小的完整卷积操作(包括efficient grouped/depthwise convolutions，Batch Normalization，ReLU function)。图中使用3×3和5×5的卷积核的两个分支。为了进一步提高效率节省计算量，将常规的5x5卷积替换为5x5的空洞卷积，即3x3，rate = 2卷积核。空洞卷积在Segmentation中也已经被广泛的使用了。下图为5x5的空洞卷积：

![](https://img-blog.csdnimg.cn/20190429143859186.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0lUT01H,size_16,color_FFFFFF,t_70)

（2）Fuse：首先通过元素求和从多个分支中融合出结果。（这部分和SE模块的处理大致相同）

        首先两个特征图进行element-wise summation（其实就是add），然后得到了新的特征图U：

                     ![](https://img-blog.csdnimg.cn/20190405202651481.png)

        然后 U 通过简单地使用**全局平均池**来嵌入全局信息，从而生成信道统计信息S，s∈![](https://private.codecogs.com/gif.latex?R%5EC)
，**C是模型图中S的特征维数或公式s的特征维数。** 

                 ![](https://img-blog.csdnimg.cn/20190405203309413.png)

       再通过一个简单的全连接（fc）层创建了一个紧凑的特征Z，这里的z相当于一个squeeze操作。使其能够进行精确和自适应的选择特征，同时减少了维度以提高效率。z∈![](https://private.codecogs.com/gif.latex?R%5E%7Bd%5Ctimes%201%7D)

                  ![](https://img-blog.csdnimg.cn/20190405203823328.png)

       其中 δ 是relu函数，B表示批标准化，W∈![](https://private.codecogs.com/gif.latex?R%5E%7Bd%5Ctimes%20%D7C%7D)
。为了研究 **d(全连接后的特征维数，即公式z或模型图中Z的特征维数)** 对模型效率的影响，我们使用一个折减比 r 来控制其值。可以参考SENet中的压缩比率r。

                ![](https://img-blog.csdnimg.cn/20190405204539561.png)
 

      L表示d的极小值，通过 L=32 是原文中实验的设置。

（3）Select：按照信道的方向使用softmax。

       Select操作对应于SE模块中的Scale。区别是Select使用输出的两个矩阵a和b，其中矩阵b为冗余矩阵，在如图两个分支的情况下b=1-a。

               ![](https://img-blog.csdnimg.cn/20190405205539856.png)

      然后与Split卷积后的特征进行乘和求和操作。通过这两个分支的情况，可以推断更多分支的情况。

               ![](https://img-blog.csdnimg.cn/2019040520584527.png)

      （1）（2）都比较好理解，（3）就是比较巧妙的部分了，即有几个尺度的特征图（图中的例子是两个），则将squeeze出来的特征再通过几个全连接将特征数目回复到c，（假设我们用了三种RF，我们squeeze之后的特征要接三个全连接，每个全连接的神经元的数目都是c）这个图上应该在空线上加上FC会比较好理解吧。然后将这N个全连接后的结果拼起来（可以想象成一个cxN的矩阵），然后纵向的（每一列）进行softmax。如图中的蓝色方框所示——即不同尺度的同一个channel就有了不同的权重。

![](https://img-blog.csdnimg.cn/20190326181934573.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FpeHV0dW82MDg3,size_16,color_FFFFFF,t_70)

        如上图所示使ResNext-50，SENet-ResNext-50，SKNet-ResNext-50三个网络的结构对比。可以看到SENet是在(1×1卷积+3×3卷积+1×1卷积)完整卷积操作后直接加入全连接层，学习通道间依赖关系，再将学习到的通道权重加权回原向量。  
       而SKNet则是用两个或多个不同大小的卷积核卷积操作加上学习通道权重的FC层替代了ResNext中3\*3卷积部分，输出向量再继续进行1×1卷积操作。从参数量来看，由于模块嵌入位置不同，SKNet的参数量与SENet大致持平(或略小于后者)。计算量也略有上升（当然，带来的精度提升远小于增加的计算量成本）。

       SKNet整体感觉融合了较多的trick，Select部分使用的soft attention和Sequeeze and Excitation block中对特征图加权操作类似，区别在于Sequeeze and Excitation block考虑的是Channel之间的权重，而Select部分的attention不仅考虑了Channel之间的权重，还考虑了两路不同卷积的权重。但SKNet使网络可以获取不同感受野的信息，这或许可以成为一种泛化能力更好的网络结构。毕竟虽然Inception网络结构精妙，效果也不错，但总觉得有种人工设计特征痕迹过重的感觉。如果有网络可以自适应的调整结构，以获取不同感受野信息，那么或许可以实现CNN模型极限的下一次突破。

```null
def __init__(self, features, WH, M, G, r, stride=1 ,L=32):            features: input channel dimensionality.            WH: input spatial dimensionality, used for GAP kernel size.            M: the number of branchs.            G: num of convolution groups.            r: the radio for compute d, the length of z.            stride: stride, default 1.            L: the minimum dim of the vector z in paper, default 32.super(SKConv, self).__init__()        d = max(int(features/r), L)        self.convs = nn.ModuleList([])            self.convs.append(nn.Sequential(                nn.Conv2d(features, features, kernel_size=3+i*2, stride=stride, padding=1+i, groups=G),                nn.BatchNorm2d(features),        self.gap = nn.AvgPool2d(int(WH/stride))        self.fc = nn.Linear(features, d)        self.fcs = nn.ModuleList([])        self.softmax = nn.Softmax(dim=1)for i, conv in enumerate(self.convs):            fea = conv(x).unsqueeze_(dim=1)                feas = torch.cat([feas, fea], dim=1)        fea_U = torch.sum(feas, dim=1)        fea_s = self.gap(fea_U).squeeze_()for i, fc in enumerate(self.fcs):            vector = fc(fea_z).unsqueeze_(dim=1)                attention_vectors = vector                attention_vectors = torch.cat([attention_vectors, vector], dim=1)        attention_vectors = self.softmax(attention_vectors)        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)        fea_v = (feas * attention_vectors).sum(dim=1)def __init__(self, in_features, out_features, WH, M, G, r, mid_features=None, stride=1, L=32):            in_features: input channel dimensionality.            out_features: output channel dimensionality.            WH: input spatial dimensionality, used for GAP kernel size.            M: the number of branchs.            G: num of convolution groups.            r: the radio for compute d, the length of z.            mid_features: the channle dim of the middle conv with stride not 1, default out_features/2.            L: the minimum dim of the vector z in paper.super(SKUnit, self).__init__()            mid_features = int(out_features/2)        self.feas = nn.Sequential(            nn.Conv2d(in_features, mid_features, 1, stride=1),            nn.BatchNorm2d(mid_features),            SKConv(mid_features, WH, M, G, r, stride=stride, L=L),            nn.BatchNorm2d(mid_features),            nn.Conv2d(mid_features, out_features, 1, stride=1),            nn.BatchNorm2d(out_features)if in_features == out_features:             self.shortcut = nn.Sequential()            self.shortcut = nn.Sequential(                nn.Conv2d(in_features, out_features, 1, stride=stride),                nn.BatchNorm2d(out_features)return fea + self.shortcut(x)def __init__(self, class_num):super(SKNet, self).__init__()        self.basic_conv = nn.Sequential(            nn.Conv2d(3, 64, 3, padding=1),        self.stage_1 = nn.Sequential(            SKUnit(64, 256, 32, 2, 8, 2, stride=2),            SKUnit(256, 256, 32, 2, 8, 2),            SKUnit(256, 256, 32, 2, 8, 2),        self.stage_2 = nn.Sequential(            SKUnit(256, 512, 32, 2, 8, 2, stride=2),            SKUnit(512, 512, 32, 2, 8, 2),            SKUnit(512, 512, 32, 2, 8, 2),        self.stage_3 = nn.Sequential(            SKUnit(512, 1024, 32, 2, 8, 2, stride=2),            SKUnit(1024, 1024, 32, 2, 8, 2),            SKUnit(1024, 1024, 32, 2, 8, 2),        self.pool = nn.AvgPool2d(8)        self.classifier = nn.Sequential(            nn.Linear(1024, class_num),        fea = self.classifier(fea)    x = torch.rand(8,64,32,32)    conv = SKConv(64, 32, 3, 8, 2)print('out shape : {}'.format(out.shape))print('loss value : {}'.format(loss))
```
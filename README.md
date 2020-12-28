# annotatedTransformer
## Basic architecture
- EncoderDecoder (Basically decode(encode(src, tgt)))
  - Encoder
    - Encoder Layer
      - Multi head attention
      - sub layer connection
  - Decoder
    - Decoder Layer
      - Multi head attention
      - sub layer connection
  - Generator (Simply FC+softmax in the last layer)

## Details
### class Encoder
Encoder板块由N个encoder layer组成
```
for layer in layers:
  x = layer(x)
x = layerNorm(x)
return x 
```
这里的x维度为【batch_size, seq_len, embedded_size】

#### **第一个问题，LayerNorm和BatchNorm的区别？**
```
作者：秩法策士
链接：https://zhuanlan.zhihu.com/p/74516930
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

Batch 顾名思义是对一个batch进行操作。假设我们有 10行 3列 的数据，即我们的batchsize = 10，每一行数据有三个特征，假设这三个特征是【身高、体重、年龄】。那么BN是针对每一列（特征）进行缩放，例如算出【身高】的均值与方差，再对身高这一列的10个数据进行缩放。体重和年龄同理。这是一种“列缩放”。而layer方向相反，它针对的是每一行进行缩放。即只看一笔数据，算出这笔所有特征的均值与方差再缩放。这是一种“行缩放”。细心的你已经看出来，layer normalization 对所有的特征进行缩放，这显得很没道理。我们算出一行这【身高、体重、年龄】三个特征的均值方差并对其进行缩放，事实上会因为特征的量纲不同而产生很大的影响。但是BN则没有这个影响，因为BN是对一列进行缩放，一列的量纲单位都是相同的。那么我们为什么还要使用LN呢？因为NLP领域中，LN更为合适。如果我们将一批文本组成一个batch，那么BN的操作方向是，对每句话的第一个词进行操作。但语言文本的复杂性是很高的，任何一个词都有可能放在初始位置，且词序可能并不影响我们对句子的理解。而BN是针对每个位置进行缩放，这不符合NLP的规律。而LN则是针对一句话进行缩放的，且LN一般用在第三维度，如[batchsize, seq_len, dims]中的dims，一般为词向量的维度，或者是RNN的输出维度等等，这一维度各个特征的量纲应该相同。因此也不会遇到上面因为特征的量纲不同而导致的缩放问题。
```

#### **第二个问题，Normalization的作用？**
```
BN的基本思想其实相当直观：因为深层神经网络在做非线性变换前的激活输入值（就是那个x=WU+B，U是输入）随着网络深度加深或者在训练过程中，其分布逐渐发生偏移或者变动，之所以训练收敛慢，一般是整体分布逐渐往非线性函数的取值区间的上下限两端靠近（对于Sigmoid函数来说，意味着激活输入值WU+B是大的负值或正值），所以这导致后向传播时低层神经网络的梯度消失，这是训练深层神经网络收敛越来越慢的本质原因，而BN就是通过一定的规范化手段，把每层神经网络任意神经元这个输入值的分布强行拉回到均值为0方差为1的标准正太分布而不是萝莉分布（哦，是正态分布），其实就是把越来越偏的分布强制拉回比较标准的分布，这样使得激活输入值落在非线性函数对输入比较敏感的区域，这样输入的小变化就会导致损失函数较大的变化，意思是这样让梯度变大，避免梯度消失问题产生，而且梯度变大意味着学习收敛速度快，能大大加快训练速度。
```


这里面提到“如果都通过BN，那么不就跟把非线性函数替换成线性函数效果相同了？”，其实是因为例如说tanh的0附近是类似线性的。

https://blog.csdn.net/malefactor/article/details/51476961

### class Encoder Layer



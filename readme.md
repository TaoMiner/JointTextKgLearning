## Intro

本程序受[1]，[2]启发，利用skip-gram model+align model在wiki text，wiki outlinks和wiki anchors数据上实现文本和实体embedding的联合学习；

## 程序说明

#### tf implement

最初基于tensorflow tutorial官方的word2vec_optimized.py实现，自定义了2个op，分别用于align model的align_op，以及实体outlink的sg_op。

它需要三个输入：

train_text, 格式为 <word>space<word>...

 train_kg, 格式为<target entity>\t\t<link entity1>;<link entity2>;...

train_anchor, 格式为<target entity>\t\t<left context word1>;<left context word2>;...

<target entity>\t\t<right context word1>;<right context word2>;...

这三个文件可以有preprocess中的preprocess.py从enwiki-text.dat及enwiki-outlinks.dat生成，而这两个文件是从enwiki dump中抽取而成。

**程序最大的问题是受限于tensorflow目前版本protobuf具有2Gb的限制，即单个variable size不能超过2gb，对于文本处理，200维\*float 8byte的向量，词典不能超过125w＋，实在太小。也尝试了将variable切片，再concat连接或者使用初始化器初始化，或者转化成numpy.narray格式存储，load时进行初始化均失败。连接失败在于运算过后输出不再是variable，而是tensor，这样的话输入内置word2vec.neg_train的op会报错：貌似格式不对。初始化失败仍然是单个variable不能超过2gb。其实还有一种解决办法，重写内置op，但经测试，没有改一行代码，将word2vec的kernerl生成动态链接.so文件，效率下降十多倍。故放弃。**

#### c implement:

基于google官方的word2vec trunk实现，速度非常快，1亿词的text，vocab 37w，1亿多的outlinks，200w＋的实体以及＊＊超链接，单机训练只需要2分钟。。。

特别的，在align model中使用cw和sg函数切换是将word vector向entity vector靠拢(不改变entity vector)还是反过来。因为entity数量通常比word多，所以cw效果更好些。都置1则先跑cw再sg，使用anchor训练两遍，效果待测试。

需要两个输入: train_text: <word>\s<word>\s[[entity|mention]]\s<word>\s[[entity]]\s<word>...

train_kg与上面相同。

可以使用preprocess中的remainAnchor.py抽取。

#### FindMention

使用java实现的mention统计软件，基于[3]的ac自动机+double array trie树快速进行mention匹配。

在text中找mention会发生冲突，这里取了最长的mention保留。

另外，匹配的时候以单词为单位。




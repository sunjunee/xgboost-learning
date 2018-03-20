
# XGBoost python调参完整指南

### Introduction

如果你需要做预测，那么你需要XGBoost。XGBoost算法已经成为许多数据科学家的终极武器。 这是一个非常复杂的算法，足以处理各种不规则的数据

搭建一个XGBoost模型很简单，但是提升模型的性能并不简单。XGBoost的参数很多，为了优化模型的性能，调参是不可避免的。

这篇问孩子那个适合XGBoost的初学者，在这篇文章中，我们将学习XGBoost的调参技巧，同时了解一些XGBoost的有用信息。而且我们使用pyton，将算法应用到了一个实际任务上。


### What should you know？

XGBoost(eXtreme Gradient Boosting) 是梯度提升的一种新的实现。阅读这篇文章之前，你至少需要知道Gradient Boosting是什么玩意儿。

### Table of Contents

1、XGBoost的优势

2、理解XGBoost的参数

3、调参

### 1. XGBoost的优势

我非常欣赏这种算法为预测模型加入的boosting性能。它具有如下的优势：

#### （1）正则化：

标准的GBM模型，没有正则化，而XGBoost加入了正则化，这能帮助防止过拟合。实际上，XGBoost也被称为“正则化的boosting”。其正则化中，将叶子节点个数，还有叶子节点的分数两种参数，组合成了正则项。

#### （2）并行处理：

并行处理，使得XGBoost比GBM快超级多。但是我们知道XGBoost是一个顺序执行的模型，怎么能并行化呢。每棵树都必须在前面的树生成之后，才能被建立，那是怎么实现的呢？http://zhanpengfang.github.io/418home.html 里面有提到。**有待探究...**

据称，XGBoost可以在Hadoop上实现。

### （3）高灵活性

XGBoost允许用户自定义优化的目标，以及评价的标准。这样，我们可以用模型实现任何我们想做的。

### （4）处理缺失值

XGBoost内置处理缺失值的方法。用于需要将缺失值用特殊的值传递到模型中，XGBoost会尝试不同的事情，因为它会在每个节点上遇到缺失的值，并且会在未来学习为缺失值采取的路径。

### （5）树剪枝

GBM在遇到负的loss时，会停止树的分裂，所以它更多地是一种贪心算法。

XGBoost最多分类到max_depth深度，然后开始剪枝，移除loss是负的以下的分裂结果。

XGBoost的做法的优点的，如果GBM在遇到一个分裂，其loss是-2，则会停止，但是当下一次分裂是+10时，实际上，loss合并起来是+8，XGBoost就会继续分裂到更好的结果。

### （6）内置交叉验证

用户可以在XGBoost的每次迭代中，使用交叉验证。因此我们可以很容易得到每次运行需要多少次迭代。

而相GBM模型，我们必须运行网格搜索，只能测试有限的值。

### （7）在已有模型上训练

用户可以接着之前定的模型继续训练。

sklearn里面实现的GBM也实现了这一点。


## 2.XGBoost的参数


作者将XGBoost的参数，分成了3类：

* 通用参数： 用于调整全局的功能
* Booster参数： 用于调节每次循环中的单个booster（树/回归）
* 学习任务的参数：用于最优化

### （1）通用参数
* booster [default=gbtree]
选择基本的模型类型，有两种选择：
gbtree: tree-based models (基于树的模型)
gblinear: linear models (线性模型)

* **silent [default=0]:**
是否在运行过程中输出信息，默认为0不输出

* nthread
线程数，用于并行计算。
默认选择最大的线程数。

还有另外两个通用的参数，XGBoost会自动设置，我们不需要管它。

### （2）booster参数

总共有两种booster，前面已经看到了，我们这里只讨论树booster的参数，因为它的性能比线性的booster要好很多，而线性的我们基本不用呀。

* **eta [default=0.3]**
每次迭代，学习率的衰减量。类似于GBM中的学习率？通常的取值在0.01-0.2之间。

* **min_child_weight [default=1]**
定义生成一个子节点，需要的最小的权重观测值。
该值有利于防止过拟合，值越大，越能防止模型拟合噪声点。但是过大的值，会造成欠拟合。

* **max_depth [default=6]**
每棵树的最大深度，用来控制过拟合。因为过深的树结构，越能拟合噪声点，一般取3-10。

* **max_leaf_nodes**
一棵树上，最多可能的叶子节点数目。也能通过max_depth来决定，因为对于一颗二叉树而言，深度n表示最多有2^n个叶子节点。

* **gamma [default=0]**
和min_child_weight不同，这里虽然也是控制叶子节点生成的参数，但是它是针对loss来说的，它表示的是，生成一个叶子节点所需要的最小的loss减少量。
针对不同的损失函数，这个参数需要被单独优化！


* max_delta_step [default=0]
这参数限制每棵树权重改变的最大步长。如果这个参数的值为0，那就意味着没有约束。如果它被赋予了某个正值，那么它会让这个算法更加保守。 通常，这个参数不需要设置。但是当各类别的样本十分不平衡时，它对逻辑回归是很有帮助的。

* subsample [default=1]
这个参数控制对于每棵树，随机采样的比例。 减小这个参数的值，算法会更加保守，避免过拟合。但是，如果这个值设置得过小，它可能会导致欠拟合。 典型值：0.5-1

8. colsample_bytree [default=1]
用来控制每棵随机采样的列数的占比(每一列是一个特征)。 典型值：0.5-1。类似于随机森林对特征数量的随机选择。

9. colsample_bylevel [default=1]
Denotes the subsample ratio of columns for each split, in each level.
I don’t use this often because subsample and colsample_bytree will do the job for
you. but you can explore further if you feel so.
10. lambda [default=1]
L2 regularization term on weights (analogous to Ridge regression)
This used to handle the regularization part of XGBoost. Though many data scientists don’t
use it often, it should be explored to reduce over􀃚tting.
11. alpha [default=0]
L1 regularization term on weight (analogous to Lasso regression)
Can be used in case of very high dimensionality so that the algorithm runs faster when
implemented
12. scale_pos_weight [default=1]
A value greater than 0 should be used in case of high class imbalance as it helps in faster
convergence.

### （3）学习任务的参数

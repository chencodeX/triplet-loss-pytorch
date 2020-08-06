# triplet loss pytorth

本项目使用了pytorch本身自带的TripletMarginLoss 来实现三元组损失。同时自己设计了一个高度兼容的组织三元组数据的Dataloader。

Dataloader 的实现参考了pytorch本身Dataloader的设计理念，使用了数据缓冲区和线程池配合的方案，能让你的模型在GPU上全力运算的同时，CPU和IO提前为你准备好下一个batch的数据。

简而言之，手离键盘脚离地的使用它！

在训练文件中，你可以看到如何使用这个三元组数据装载器的示例，在这里我使用了一个细粒度分类的任务进行处理。
并同时使用了TripletMarginLoss和CrossEntropyLoss进行训练，因为多次的实验表明，这种组合的方式会得到更好的效果（最起码它会收敛更快:)）。

# Exposure
![:name](https://count.getloli.com/get/@:chencodeX)

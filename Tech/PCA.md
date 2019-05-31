# PCA数学原理及其实现

目的： 分析数据特征之间的关系时，当数据维度过高时，一是 处理困难，二是 数据变得稀疏， 所以会引发**维数灾难**
       怎样降维才会最少的损失信息？利用维度之间的相关性，降低损失。


协方差矩阵理解： https://blog.csdn.net/GoodShot/article/details/79940438
迹的理解：      http://www.360doc.com/content/17/1107/09/32196507_701568401.shtml
博客链接：blog.codinglabs.org/articles/pca-tutorial.html


* 大致原理


 
 假设原始数据为<img src="http://chart.googleapis.com/chart?cht=tx&chl= X=(x1,x2,...,xn)^{T}" style="border:none;">，
 我们的优化目标是使得投影之后（也就是线性变换之后） 在每个维度上的数据的方差最大，方差的定义是 <img src="http://chart.googleapis.com/chart? cht=tx&chl= Var(a)=\frac{1}{m}*\sum (a_{i}-u)^2" style="border:none;">， 也就可以直接考虑成 


 


 

 

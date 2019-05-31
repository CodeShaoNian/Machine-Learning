# PCA数学原理及其实现

目的： 分析数据特征之间的关系时，当数据维度过高时，一是 处理困难，二是 数据变得稀疏， 所以会引发**维数灾难**
       怎样降维才会最少的损失信息？利用维度之间的相关性，降低损失。


协方差矩阵理解： https://blog.csdn.net/GoodShot/article/details/79940438
迹的理解：      http://www.360doc.com/content/17/1107/09/32196507_701568401.shtml
博客链接：blog.codinglabs.org/articles/pca-tutorial.html


* 大致原理


 
 假设原始数据 为 
 
<img src="http://chart.googleapis.com/chart?cht=tx&chl=\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" style="border:none;">

 <script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>
 $$x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}$$
\\(x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}\\)
 

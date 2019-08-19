使用卷积神经网络进行高光谱遥感数据分类，使用的数据源为[Pavia University高光谱数据](http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes#Pavia_Centre_and_University)
文件结构如下

    文件夹log--日志文件夹，存放TensorBorad日志、网络参数文件、混淆矩阵图
    文件夹Patch--存放数据处理的切片结果
    文件夹PaviaU--高光谱数据下载存放位置
    文件夹predicted--CNN对原始影像的分类结果
    data.py--对原始高光谱影像进行数据处理，生成切片
    net.py--神经网络模型
    train.py--训练神经网络
    utils.py--需要用到的函数
    show.py--使用训练好的神经网络模型对原始数据进行分类

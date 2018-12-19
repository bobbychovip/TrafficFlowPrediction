# TrafficFlowPrediction
LCTFP: A freeway traffic flow prediction model based on CNN and LSTM
## 使用
### 准备数据
进入PeMS文件夹，解压PEMS.zip文件至PeMS目录下。文件的组织形式应如/PeMS/Station%形式。
### 数据处理
Station 文件夹内保存了各个站点半年内的交通流数据。每个txt文件为该站点在一天内的交通流。运行脚本data_preprocess.py。该脚本读取所有txt文件并对原始数据进行处理：包括数据清理、归一化、处理时间序列。最后把经过处理后的数据存入数组。

## 实验说明
### 模型结构
LCTFP模型使用1D CNN + LSTM的组合结构对高速公路短时交通流进行预测。1D CNN用来学习短时交通流的空间特征，LSTM用来学习交通流演变的时间特征。脚本cnn_lstm_param.py可进行超参数搜索，运行前需安装hyperas。
### LCTFP模型结果
![LCTFP](https://github.com/bobbychovip/TrafficFlowPrediction/raw/master/images/cnn_lstm.png)
### 三种模型结果的比较
![LCTFP](https://github.com/bobbychovip/TrafficFlowPrediction/raw/master/images/threemodels.png)

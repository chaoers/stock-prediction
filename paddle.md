# 基于PaddlePaddle的A股预测

> 本项目尝试通过百度自研PaddlePaddle框架实现深度神经网络DNN和长短期记忆网络LSTM来对A股进行预测，使用数据为2000-2021年上证指数的数据

本项目[Github仓库](https://github.com/chaoers/stock-prediction)

## 运行环境


```python
! pip install statsmodels arch
```

    Looking in indexes: https://mirror.baidu.com/pypi/simple/
    Requirement already satisfied: statsmodels in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (0.12.2)
    Collecting arch
      Downloading https://mirror.baidu.com/pypi/packages/5f/a8/a85dad77039d2884547d4fb83d54edfc13a12e31981934d6d0fb6303b791/arch-4.19-cp37-cp37m-manylinux1_x86_64.whl (807 kB)
    [K     |████████████████████████████████| 807 kB 16.2 MB/s eta 0:00:01
    [?25hRequirement already satisfied: scipy>=1.2.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from arch) (1.3.0)
    Collecting property-cached>=1.6.4
      Downloading https://mirror.baidu.com/pypi/packages/5c/6c/94d8e520b20a2502e508e1c558f338061cf409cbee78fd6a3a5c6ae812bd/property_cached-1.6.4-py2.py3-none-any.whl (7.8 kB)
    Requirement already satisfied: numpy>=1.14 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from arch) (1.16.4)
    Collecting cython>=0.29.14
      Downloading https://mirror.baidu.com/pypi/packages/0c/15/cca3ac44776df9ee27286941315dd8b14a598e8d80970200d05f720b9274/Cython-0.29.23-cp37-cp37m-manylinux1_x86_64.whl (2.0 MB)
    [K     |████████████████████████████████| 2.0 MB 32.1 MB/s eta 0:00:01
    [?25hRequirement already satisfied: pandas>=0.23 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from arch) (1.1.5)
    Requirement already satisfied: patsy>=0.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from statsmodels) (0.5.1)
    Requirement already satisfied: python-dateutil>=2.7.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pandas>=0.23->arch) (2.8.0)
    Requirement already satisfied: pytz>=2017.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pandas>=0.23->arch) (2019.3)
    Requirement already satisfied: six in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from patsy>=0.5->statsmodels) (1.15.0)
    Installing collected packages: property-cached, cython, arch
      Attempting uninstall: cython
        Found existing installation: Cython 0.29
        Uninstalling Cython-0.29:
          Successfully uninstalled Cython-0.29
    Successfully installed arch-4.19 cython-0.29.23 property-cached-1.6.4
    [33mWARNING: You are using pip version 21.0.1; however, version 21.1 is available.
    You should consider upgrading via the '/opt/conda/envs/python35-paddle120-env/bin/python -m pip install --upgrade pip' command.[0m



```python
# -*- coding: utf-8 -*-

%matplotlib inline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers

import statsmodels.api as sm 
from statsmodels.graphics.api import qqplot
from statsmodels.tsa.arima_model import ARIMA
import math
import arch
```

### GPU(CPU)调用


```python
# place = fluid.CUDAPlace(0) # GPU环境用这个
place = fluid.CPUPlace() # CPU环境用这个
```

## 数据清洗

### 导入数据


```python
data_pd = pd.read_csv('./data/data72449/shangzheng.csv')
```

### 查看数据头部


```python
data_pd.head()
# data_pd.tail()
# data_pd.describe()
# data_pd.info()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>over</th>
      <th>begin</th>
      <th>high</th>
      <th>low</th>
      <th>trade</th>
      <th>wave</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2000年1月4日</td>
      <td>1406.37</td>
      <td>1368.69</td>
      <td>1407.52</td>
      <td>1361.21</td>
      <td>437.09M</td>
      <td>2.91%</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2000年1月5日</td>
      <td>1409.68</td>
      <td>1407.83</td>
      <td>1433.78</td>
      <td>1398.32</td>
      <td>688.09M</td>
      <td>0.24%</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2000年1月6日</td>
      <td>1463.94</td>
      <td>1406.04</td>
      <td>1463.95</td>
      <td>1400.25</td>
      <td>809.05M</td>
      <td>3.85%</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2000年1月7日</td>
      <td>1516.60</td>
      <td>1477.15</td>
      <td>1522.83</td>
      <td>1477.15</td>
      <td>1.60B</td>
      <td>3.60%</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2000年1月10日</td>
      <td>1545.11</td>
      <td>1531.71</td>
      <td>1546.72</td>
      <td>1506.40</td>
      <td>1.51B</td>
      <td>1.88%</td>
    </tr>
  </tbody>
</table>
</div>



### 异常项清洗


```python
data_pd.info() # 没有异常项需要处理
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5130 entries, 0 to 5129
    Data columns (total 7 columns):
     #   Column  Non-Null Count  Dtype  
    ---  ------  --------------  -----  
     0   date    5130 non-null   object 
     1   over    5130 non-null   float64
     2   begin   5130 non-null   float64
     3   high    5130 non-null   float64
     4   low     5130 non-null   float64
     5   trade   5130 non-null   object 
     6   wave    5130 non-null   object 
    dtypes: float64(4), object(3)
    memory usage: 280.7+ KB


### 画图查看数据


```python
plt.plot(data_pd['over'])
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/cbook/__init__.py:2349: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      if isinstance(obj, collections.Iterator):
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/cbook/__init__.py:2366: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      return list(data) if isinstance(data, collections.MappingView) else data





    [<matplotlib.lines.Line2D at 0x7f48fe6e6d50>]




![png](output_14_2.png)


### 提取需要数据并格式化


```python
data_pd.set_index('date', inplace=True)
data_pd.drop(columns=['trade', 'wave'], inplace=True)
data_pd.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>over</th>
      <th>begin</th>
      <th>high</th>
      <th>low</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2000年1月4日</th>
      <td>1406.37</td>
      <td>1368.69</td>
      <td>1407.52</td>
      <td>1361.21</td>
    </tr>
    <tr>
      <th>2000年1月5日</th>
      <td>1409.68</td>
      <td>1407.83</td>
      <td>1433.78</td>
      <td>1398.32</td>
    </tr>
    <tr>
      <th>2000年1月6日</th>
      <td>1463.94</td>
      <td>1406.04</td>
      <td>1463.95</td>
      <td>1400.25</td>
    </tr>
    <tr>
      <th>2000年1月7日</th>
      <td>1516.60</td>
      <td>1477.15</td>
      <td>1522.83</td>
      <td>1477.15</td>
    </tr>
    <tr>
      <th>2000年1月10日</th>
      <td>1545.11</td>
      <td>1531.71</td>
      <td>1546.72</td>
      <td>1506.40</td>
    </tr>
  </tbody>
</table>
</div>



### 标准化数据

> 在这里只能通过训练集特征来对全部数据进行标准化，因为理论来说我们只知道当前时间之前的数据。详见[我的博客](https://www.asteriscum.cn/2021/03/03/53/21/), 所以这里就要确认数据集划分比例


```python
split_boundary = int(data_pd.shape[0] * 0.8)

scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(data_pd[:split_boundary])
data_transform = scaler.transform(data_pd)

# data_transform
```

### 建立时间滑窗，我们使用前5天数据来预测后一天的数据


```python
sequence_length = 5
delay = 1
# Generated training sequences for use in the model.
def create_sequences(values, time_steps=sequence_length+delay):
    output = []
    for i in range(len(values) - time_steps):
        output.append(values[i : (i + time_steps)])
    return np.stack(output)

data = create_sequences(data_transform)
print("Training input shape: ", data.shape)
```

    Training input shape:  (5124, 6, 4)


### 划分训练集和测试集


```python
train = data[: split_boundary]
test = data[split_boundary:]

train.shape,test.shape
```




    ((4104, 6, 4), (1020, 6, 4))



### 建立batch生成器


```python
def switch_reader(is_val: bool = False):
    def reader():
        # 判断是否是验证集
        if is_val:
            # 抽取数据使用迭代器返回
            for te in test:
                yield te[:sequence_length],te[-1:][:,0]
        else:
            # 抽取数据使用迭代器返回
            for tr in train:
                yield tr[:sequence_length],tr[-1:][:,0]#只取第0列pm2.5的值为label
        
    return reader   # 注意！此处不需要带括号
```

#### 测试batch生成器


```python
batch_size = 128
train_reader = fluid.io.batch(reader=switch_reader(), batch_size=batch_size)
val_reader = fluid.io.batch(reader=switch_reader(is_val=True), batch_size=batch_size)
for data in train_reader():
    # print(data[0].shape,data[1].shape)
    train_x=np.array([x[0] for x in data],np.float32)
    train_y = np.array([x[1] for x in data]).astype('int64')
    print(train_x.shape,train_y.shape)
```

    (128, 5, 4) (128, 1)
    (128, 5, 4) (128, 1)
    (128, 5, 4) (128, 1)
    (128, 5, 4) (128, 1)
    (128, 5, 4) (128, 1)
    (128, 5, 4) (128, 1)
    (128, 5, 4) (128, 1)
    (128, 5, 4) (128, 1)
    (128, 5, 4) (128, 1)
    (128, 5, 4) (128, 1)
    (128, 5, 4) (128, 1)
    (128, 5, 4) (128, 1)
    (128, 5, 4) (128, 1)
    (128, 5, 4) (128, 1)
    (128, 5, 4) (128, 1)
    (128, 5, 4) (128, 1)
    (128, 5, 4) (128, 1)
    (128, 5, 4) (128, 1)
    (128, 5, 4) (128, 1)
    (128, 5, 4) (128, 1)
    (128, 5, 4) (128, 1)
    (128, 5, 4) (128, 1)
    (128, 5, 4) (128, 1)
    (128, 5, 4) (128, 1)
    (128, 5, 4) (128, 1)
    (128, 5, 4) (128, 1)
    (128, 5, 4) (128, 1)
    (128, 5, 4) (128, 1)
    (128, 5, 4) (128, 1)
    (128, 5, 4) (128, 1)
    (128, 5, 4) (128, 1)
    (128, 5, 4) (128, 1)
    (8, 5, 4) (8, 1)


## DNN网络

### 定义DNN网络


```python
class MyDNNModel(fluid.dygraph.Layer):
    '''
    DNN网络
    '''
    def __init__(self):
        super(MyDNNModel,self).__init__()
        self.fc1=fluid.dygraph.Linear(5*4,10,act='relu')
        self.fc2=fluid.dygraph.Linear(10,1)
        
    def forward(self,input):        # forward 定义执行实际运行时网络的执行逻辑
        '''前向计算'''
        # print('input',input.shape)
        input =fluid.layers.reshape(input,shape=[-1,5*4])
        out=self.fc1(input)
        out=self.fc2(out)
        # print(out.shape)
        return out
```

### 损失作图函数


```python
def draw_train_loss(Batchs, train_loss,eval_loss, label=""):
    # print(len(Batchs), len(train_loss), len(eval_loss))
    plt.figure(figsize=(10, 6))
    title= label + " training-eval loss"
    plt.title(title, fontsize=24)
    plt.xlabel("batch", fontsize=14)
    plt.ylabel("loss", fontsize=14)
    plt.plot(Batchs, train_loss, color='red', label='training loss')
    plt.plot(Batchs, eval_loss, color='g', label='eval loss')
    plt.legend()
    plt.grid()
    plt.savefig('./img/'+ label + '_loss.png', dpi=300)
    plt.show()

    # print(Batchs, train_loss)
```

### 开始训练

> 注：有时训练DNN会出现损失函数反常现象，这时请再次运行训练


```python
with fluid.dygraph.guard(place):
    model=MyDNNModel() #模型实例化
    model.train() #训练模式
    # opt=fluid.optimizer.SGDOptimizer(learning_rate=train_parameters['learning_strategy']['lr'], parameter_list=model.parameters())#优化器选用SGD随机梯度下降，学习率为0.001.
    opt=fluid.optimizer.AdamOptimizer(learning_rate=0.0001, parameter_list=model.parameters()) 
    epochs_num=200#迭代次数
    batch_size = 128 * 3
    train_reader = fluid.io.batch(reader=switch_reader(), batch_size=batch_size)
    val_reader = fluid.io.batch(reader=switch_reader(is_val=True), batch_size=batch_size)
    Batch=0
    Batchs=[]
    all_train_loss=[]
    all_eval_loss=[]
    for pass_num in range(epochs_num):
        for batch_id, data in enumerate(train_reader()): 
            data_x=np.array([x[0] for x in data],np.float32)
            data_y = np.array([x[1] for x in data]).astype('float32')
            data_x = fluid.dygraph.to_variable(data_x)
            data_y = fluid.dygraph.to_variable(data_y)
        
            # print(data_x.shape, data_y.shape)

            predict=model(data_x)
            # print(predict.shape)
            loss=fluid.layers.mse_loss(predict,data_y)
            avg_loss=fluid.layers.mean(loss)#获取loss值
            avg_loss.backward()       
            opt.minimize(avg_loss)    #优化器对象的minimize方法对参数进行更新 
            model.clear_gradients()   #model.clear_gradients()来重置梯度
            if batch_id!=0 and batch_id%10==0:
                Batch = Batch+10
                Batchs.append(Batch)
                all_train_loss.append(avg_loss.numpy()[0])
                evalavg_loss=[]
                for eval_data in val_reader():
                    eval_data_x = np.array([x[0] for x in eval_data],np.float32)
                    eval_data_y = np.array([x[1] for x in eval_data]).astype('float32')

                    eval_data_x = fluid.dygraph.to_variable(eval_data_x)
                    eval_data_y = fluid.dygraph.to_variable(eval_data_y)

                    eval_predict=model(eval_data_x)
                    eval_loss=fluid.layers.mse_loss(eval_predict,eval_data_y)
                    eval_loss=fluid.layers.mean(eval_loss)
                    evalavg_loss.append(eval_loss.numpy()[0])#获取loss值
                all_eval_loss.append(sum(evalavg_loss)/len(evalavg_loss))
        print("epoch:{},batch_id:{},train_loss:{},eval_loss:{}".format(pass_num,batch_id,avg_loss.numpy(),sum(evalavg_loss)/len(evalavg_loss)))     

    fluid.save_dygraph(model.state_dict(),'./model/MyDNNModel')#保存模型
    fluid.save_dygraph(opt.state_dict(),'./model/MyDNNModel')#保存模型
    print("Final loss: {}".format(avg_loss.numpy()))    
#绘制训练图和验证损失图，以了解训练的进行情况。     
draw_train_loss(Batchs,all_train_loss,all_eval_loss, 'DNN')
```

    epoch:0,batch_id:10,train_loss:[0.11340201],eval_loss:0.10979689781864484
    epoch:1,batch_id:10,train_loss:[0.10834701],eval_loss:0.10497679188847542
    epoch:2,batch_id:10,train_loss:[0.1034525],eval_loss:0.10027983660499255
    epoch:3,batch_id:10,train_loss:[0.09868655],eval_loss:0.09569774195551872
    epoch:4,batch_id:10,train_loss:[0.09405579],eval_loss:0.09124852592746417
    epoch:5,batch_id:10,train_loss:[0.08956681],eval_loss:0.0869408609966437
    epoch:6,batch_id:10,train_loss:[0.08522683],eval_loss:0.08277822037537892
    epoch:7,batch_id:10,train_loss:[0.08103742],eval_loss:0.07876073941588402
    epoch:8,batch_id:10,train_loss:[0.07699691],eval_loss:0.07488522678613663
    epoch:9,batch_id:10,train_loss:[0.07310266],eval_loss:0.07114929022888343
    epoch:10,batch_id:10,train_loss:[0.06935226],eval_loss:0.06755035929381847
    epoch:11,batch_id:10,train_loss:[0.06574283],eval_loss:0.06408604544897874
    epoch:12,batch_id:10,train_loss:[0.06227097],eval_loss:0.06075322938462099
    epoch:13,batch_id:10,train_loss:[0.05893348],eval_loss:0.05754899668196837
    epoch:14,batch_id:10,train_loss:[0.0557294],eval_loss:0.054470483834544815
    epoch:15,batch_id:10,train_loss:[0.0526526],eval_loss:0.05151473544538021
    epoch:16,batch_id:10,train_loss:[0.04970048],eval_loss:0.04867857011655966
    epoch:17,batch_id:10,train_loss:[0.04686978],eval_loss:0.04595908708870411
    epoch:18,batch_id:10,train_loss:[0.04415732],eval_loss:0.043353563795487084
    epoch:19,batch_id:10,train_loss:[0.04155984],eval_loss:0.04085916529099146
    epoch:20,batch_id:10,train_loss:[0.03907475],eval_loss:0.03847289613137642
    epoch:21,batch_id:10,train_loss:[0.03669919],eval_loss:0.036191885049144425
    epoch:22,batch_id:10,train_loss:[0.03442962],eval_loss:0.034013282197217144
    epoch:23,batch_id:10,train_loss:[0.03226336],eval_loss:0.031934346072375774
    epoch:24,batch_id:10,train_loss:[0.03019765],eval_loss:0.029952156047026317
    epoch:25,batch_id:10,train_loss:[0.02822966],eval_loss:0.028064117456475895
    epoch:26,batch_id:10,train_loss:[0.02635642],eval_loss:0.026267704864343006
    epoch:27,batch_id:10,train_loss:[0.02457511],eval_loss:0.024559970634679
    epoch:28,batch_id:10,train_loss:[0.02288317],eval_loss:0.02293830302854379
    epoch:29,batch_id:10,train_loss:[0.02127784],eval_loss:0.021400041567782562
    epoch:30,batch_id:10,train_loss:[0.01975648],eval_loss:0.019942651347567637
    epoch:31,batch_id:10,train_loss:[0.01831645],eval_loss:0.018563537082324427
    epoch:32,batch_id:10,train_loss:[0.0169552],eval_loss:0.017260171473026276
    epoch:33,batch_id:10,train_loss:[0.01567007],eval_loss:0.016030108090490103
    epoch:34,batch_id:10,train_loss:[0.01445829],eval_loss:0.014870830035458008
    epoch:35,batch_id:10,train_loss:[0.01331751],eval_loss:0.013779860145101944
    epoch:36,batch_id:10,train_loss:[0.01224531],eval_loss:0.012754882918670774
    epoch:37,batch_id:10,train_loss:[0.01123919],eval_loss:0.011793422124659022
    epoch:38,batch_id:10,train_loss:[0.01029673],eval_loss:0.010893159623568257
    epoch:39,batch_id:10,train_loss:[0.00941558],eval_loss:0.010051839984953403
    epoch:40,batch_id:10,train_loss:[0.00859341],eval_loss:0.009267098968848586
    epoch:41,batch_id:10,train_loss:[0.00782794],eval_loss:0.008536684870099029
    epoch:42,batch_id:10,train_loss:[0.00711686],eval_loss:0.007858400155479709
    epoch:43,batch_id:10,train_loss:[0.00645763],eval_loss:0.0072300890848661465
    epoch:44,batch_id:10,train_loss:[0.00584832],eval_loss:0.006649572402238846
    epoch:45,batch_id:10,train_loss:[0.00528694],eval_loss:0.006114850635640323
    epoch:46,batch_id:10,train_loss:[0.00477161],eval_loss:0.005623850971460342
    epoch:47,batch_id:10,train_loss:[0.00429993],eval_loss:0.005174518562853336
    epoch:48,batch_id:10,train_loss:[0.00386985],eval_loss:0.004764895692157249
    epoch:49,batch_id:10,train_loss:[0.0034794],eval_loss:0.004393074622688194
    epoch:50,batch_id:10,train_loss:[0.00312648],eval_loss:0.004057217117709418
    epoch:51,batch_id:10,train_loss:[0.00280905],eval_loss:0.0037554295655960837
    epoch:52,batch_id:10,train_loss:[0.00252544],eval_loss:0.0034859265433624387
    epoch:53,batch_id:10,train_loss:[0.00227371],eval_loss:0.003246874276859065
    epoch:54,batch_id:10,train_loss:[0.00205212],eval_loss:0.0030365876543025174
    epoch:55,batch_id:10,train_loss:[0.00185895],eval_loss:0.002853393438272178
    epoch:56,batch_id:10,train_loss:[0.00169252],eval_loss:0.002695668372325599
    epoch:57,batch_id:10,train_loss:[0.00155129],eval_loss:0.002561853383667767
    epoch:58,batch_id:10,train_loss:[0.00143366],eval_loss:0.002450426730016867
    epoch:59,batch_id:10,train_loss:[0.00133806],eval_loss:0.002359888244730731
    epoch:60,batch_id:10,train_loss:[0.00126303],eval_loss:0.0022888502959782877
    epoch:61,batch_id:10,train_loss:[0.00120712],eval_loss:0.002235966191316644
    epoch:62,batch_id:10,train_loss:[0.00116895],eval_loss:0.0021998675850530467
    epoch:63,batch_id:10,train_loss:[0.0011472],eval_loss:0.002179350044267873
    epoch:64,batch_id:10,train_loss:[0.0011406],eval_loss:0.0021731209320326648
    epoch:65,batch_id:10,train_loss:[0.00114787],eval_loss:0.002179978609395524
    epoch:66,batch_id:10,train_loss:[0.00116778],eval_loss:0.0021987755705292025
    epoch:67,batch_id:10,train_loss:[0.00119928],eval_loss:0.002228443316804866
    epoch:68,batch_id:10,train_loss:[0.00124123],eval_loss:0.0022679383788878718
    epoch:69,batch_id:10,train_loss:[0.00129265],eval_loss:0.0023162655221919217
    epoch:70,batch_id:10,train_loss:[0.00135251],eval_loss:0.0023725192295387387
    epoch:71,batch_id:10,train_loss:[0.00141989],eval_loss:0.0024358485049257674
    epoch:72,batch_id:10,train_loss:[0.00149389],eval_loss:0.002505350198286275
    epoch:73,batch_id:10,train_loss:[0.00157374],eval_loss:0.0025802721502259374
    epoch:74,batch_id:10,train_loss:[0.00165859],eval_loss:0.002659854246303439
    epoch:75,batch_id:10,train_loss:[0.00174767],eval_loss:0.0027433912424991527
    epoch:76,batch_id:10,train_loss:[0.00184035],eval_loss:0.0028302388188118735
    epoch:77,batch_id:10,train_loss:[0.00193602],eval_loss:0.0029198077584927282
    epoch:78,batch_id:10,train_loss:[0.00203404],eval_loss:0.0030115260742604733
    epoch:79,batch_id:10,train_loss:[0.00213386],eval_loss:0.0031047860393300653
    epoch:80,batch_id:10,train_loss:[0.00223493],eval_loss:0.0031991171029706797
    epoch:81,batch_id:10,train_loss:[0.00233674],eval_loss:0.0032941042445600033
    epoch:82,batch_id:10,train_loss:[0.00243885],eval_loss:0.003389321306409935
    epoch:83,batch_id:10,train_loss:[0.0025408],eval_loss:0.003484372515231371
    epoch:84,batch_id:10,train_loss:[0.00264226],eval_loss:0.0035788948104406395
    epoch:85,batch_id:10,train_loss:[0.00274288],eval_loss:0.0036725340178236365
    epoch:86,batch_id:10,train_loss:[0.00284236],eval_loss:0.0037650464801117778
    epoch:87,batch_id:10,train_loss:[0.00294048],eval_loss:0.0038561548960084715
    epoch:88,batch_id:10,train_loss:[0.0030369],eval_loss:0.003945622981215517
    epoch:89,batch_id:10,train_loss:[0.00313144],eval_loss:0.0040332522088040905
    epoch:90,batch_id:10,train_loss:[0.00322396],eval_loss:0.0041188388131558895
    epoch:91,batch_id:10,train_loss:[0.00331425],eval_loss:0.004202273363868396
    epoch:92,batch_id:10,train_loss:[0.00340213],eval_loss:0.00428341212682426
    epoch:93,batch_id:10,train_loss:[0.00348751],eval_loss:0.004362132012223204
    epoch:94,batch_id:10,train_loss:[0.00357027],eval_loss:0.004438332204396526
    epoch:95,batch_id:10,train_loss:[0.00365036],eval_loss:0.004511923141156633
    epoch:96,batch_id:10,train_loss:[0.00372771],eval_loss:0.0045828210810820265
    epoch:97,batch_id:10,train_loss:[0.00380221],eval_loss:0.004651023540645838
    epoch:98,batch_id:10,train_loss:[0.00387388],eval_loss:0.004716490084926288
    epoch:99,batch_id:10,train_loss:[0.00394268],eval_loss:0.004779189980278413
    epoch:100,batch_id:10,train_loss:[0.00400859],eval_loss:0.00483912982357045
    epoch:101,batch_id:10,train_loss:[0.00407161],eval_loss:0.004896301232899229
    epoch:102,batch_id:10,train_loss:[0.00413179],eval_loss:0.004950731759890914
    epoch:103,batch_id:10,train_loss:[0.00418912],eval_loss:0.00500244228169322
    epoch:104,batch_id:10,train_loss:[0.00424366],eval_loss:0.005051464696104328
    epoch:105,batch_id:10,train_loss:[0.00429544],eval_loss:0.005097825701038043
    epoch:106,batch_id:10,train_loss:[0.00434449],eval_loss:0.005141587074225147
    epoch:107,batch_id:10,train_loss:[0.00439091],eval_loss:0.005182793751979868
    epoch:108,batch_id:10,train_loss:[0.0044347],eval_loss:0.005221490670616428
    epoch:109,batch_id:10,train_loss:[0.00447596],eval_loss:0.0052577479897687835
    epoch:110,batch_id:10,train_loss:[0.00451474],eval_loss:0.005291612741226952
    epoch:111,batch_id:10,train_loss:[0.00455109],eval_loss:0.005323165484393637
    epoch:112,batch_id:10,train_loss:[0.00458508],eval_loss:0.005352462641894817
    epoch:113,batch_id:10,train_loss:[0.00461679],eval_loss:0.005379568242157499
    epoch:114,batch_id:10,train_loss:[0.0046463],eval_loss:0.005404567423587044
    epoch:115,batch_id:10,train_loss:[0.00467367],eval_loss:0.005427515522266428
    epoch:116,batch_id:10,train_loss:[0.00469897],eval_loss:0.005448492476716638
    epoch:117,batch_id:10,train_loss:[0.00472228],eval_loss:0.005467560995991032
    epoch:118,batch_id:10,train_loss:[0.00474368],eval_loss:0.005484804278239608
    epoch:119,batch_id:10,train_loss:[0.00476323],eval_loss:0.005500288447365165
    epoch:120,batch_id:10,train_loss:[0.00478104],eval_loss:0.005514086922630668
    epoch:121,batch_id:10,train_loss:[0.00479714],eval_loss:0.005526265439887841
    epoch:122,batch_id:10,train_loss:[0.00481163],eval_loss:0.005536881896356742
    epoch:123,batch_id:10,train_loss:[0.00482456],eval_loss:0.005546012427657843
    epoch:124,batch_id:10,train_loss:[0.00483601],eval_loss:0.0055537268829842406
    epoch:125,batch_id:10,train_loss:[0.00484606],eval_loss:0.005560084090878566
    epoch:126,batch_id:10,train_loss:[0.00485476],eval_loss:0.00556514613951246
    epoch:127,batch_id:10,train_loss:[0.00486218],eval_loss:0.005568975893159707
    epoch:128,batch_id:10,train_loss:[0.00486838],eval_loss:0.005571639941384395
    epoch:129,batch_id:10,train_loss:[0.00487341],eval_loss:0.0055731795728206635
    epoch:130,batch_id:10,train_loss:[0.00487733],eval_loss:0.005573668051511049
    epoch:131,batch_id:10,train_loss:[0.00488021],eval_loss:0.005573158462842305
    epoch:132,batch_id:10,train_loss:[0.00488211],eval_loss:0.005571691319346428
    epoch:133,batch_id:10,train_loss:[0.00488306],eval_loss:0.005569317067662875
    epoch:134,batch_id:10,train_loss:[0.00488312],eval_loss:0.005566088482737541
    epoch:135,batch_id:10,train_loss:[0.00488233],eval_loss:0.00556204483533899
    epoch:136,batch_id:10,train_loss:[0.00488075],eval_loss:0.00555723129461209
    epoch:137,batch_id:10,train_loss:[0.00487841],eval_loss:0.005551679835965236
    epoch:138,batch_id:10,train_loss:[0.00487536],eval_loss:0.005545444165666898
    epoch:139,batch_id:10,train_loss:[0.00487164],eval_loss:0.005538557811329762
    epoch:140,batch_id:10,train_loss:[0.00486729],eval_loss:0.005531057249754667
    epoch:141,batch_id:10,train_loss:[0.00486234],eval_loss:0.005522983769575755
    epoch:142,batch_id:10,train_loss:[0.00485683],eval_loss:0.005514367173115413
    epoch:143,batch_id:10,train_loss:[0.00485079],eval_loss:0.0055052304329971475
    epoch:144,batch_id:10,train_loss:[0.00484426],eval_loss:0.0054956030411024885
    epoch:145,batch_id:10,train_loss:[0.00483728],eval_loss:0.005485522405554851
    epoch:146,batch_id:10,train_loss:[0.00482986],eval_loss:0.005475016310811043
    epoch:147,batch_id:10,train_loss:[0.00482202],eval_loss:0.005464108660817146
    epoch:148,batch_id:10,train_loss:[0.0048138],eval_loss:0.005452821807314952
    epoch:149,batch_id:10,train_loss:[0.00480522],eval_loss:0.005441179343809684
    epoch:150,batch_id:10,train_loss:[0.00479629],eval_loss:0.0054291946192582445
    epoch:151,batch_id:10,train_loss:[0.00478704],eval_loss:0.005416894021133582
    epoch:152,batch_id:10,train_loss:[0.0047775],eval_loss:0.005404298193752766
    epoch:153,batch_id:10,train_loss:[0.00476766],eval_loss:0.005391418002545834
    epoch:154,batch_id:10,train_loss:[0.00475756],eval_loss:0.005378264002501965
    epoch:155,batch_id:10,train_loss:[0.00474721],eval_loss:0.0053648568379382295
    epoch:156,batch_id:10,train_loss:[0.00473662],eval_loss:0.0053512160666286945
    epoch:157,batch_id:10,train_loss:[0.00472581],eval_loss:0.00533735038091739
    epoch:158,batch_id:10,train_loss:[0.0047148],eval_loss:0.005323290669669707
    epoch:159,batch_id:10,train_loss:[0.00470359],eval_loss:0.0053090304136276245
    epoch:160,batch_id:10,train_loss:[0.0046922],eval_loss:0.005294591033210357
    epoch:161,batch_id:10,train_loss:[0.00468065],eval_loss:0.005279975943267345
    epoch:162,batch_id:10,train_loss:[0.00466892],eval_loss:0.005265198182314634
    epoch:163,batch_id:10,train_loss:[0.00465706],eval_loss:0.005250272341072559
    epoch:164,batch_id:10,train_loss:[0.00464505],eval_loss:0.005235200592627128
    epoch:165,batch_id:10,train_loss:[0.00463292],eval_loss:0.00521999776052932
    epoch:166,batch_id:10,train_loss:[0.00462066],eval_loss:0.005204673468445738
    epoch:167,batch_id:10,train_loss:[0.0046083],eval_loss:0.0051892326058199005
    epoch:168,batch_id:10,train_loss:[0.00459583],eval_loss:0.005173672689124942
    epoch:169,batch_id:10,train_loss:[0.00458325],eval_loss:0.00515800598077476
    epoch:170,batch_id:10,train_loss:[0.00457058],eval_loss:0.005142243501419823
    epoch:171,batch_id:10,train_loss:[0.00455782],eval_loss:0.005126386337603132
    epoch:172,batch_id:10,train_loss:[0.00454497],eval_loss:0.005110431540136536
    epoch:173,batch_id:10,train_loss:[0.00453206],eval_loss:0.005094390828162432
    epoch:174,batch_id:10,train_loss:[0.00451908],eval_loss:0.0050782684702426195
    epoch:175,batch_id:10,train_loss:[0.00450602],eval_loss:0.005062067415565252
    epoch:176,batch_id:10,train_loss:[0.00449291],eval_loss:0.005045793717727065
    epoch:177,batch_id:10,train_loss:[0.00447972],eval_loss:0.00502944605735441
    epoch:178,batch_id:10,train_loss:[0.00446648],eval_loss:0.00501303281635046
    epoch:179,batch_id:10,train_loss:[0.00445318],eval_loss:0.00499655903937916
    epoch:180,batch_id:10,train_loss:[0.00443983],eval_loss:0.004980028451730807
    epoch:181,batch_id:10,train_loss:[0.00442644],eval_loss:0.0049634337580452366
    epoch:182,batch_id:10,train_loss:[0.00441298],eval_loss:0.00494678543570141
    epoch:183,batch_id:10,train_loss:[0.00439948],eval_loss:0.004930084183191259
    epoch:184,batch_id:10,train_loss:[0.00438595],eval_loss:0.004913337140654524
    epoch:185,batch_id:10,train_loss:[0.00437235],eval_loss:0.004896547645330429
    epoch:186,batch_id:10,train_loss:[0.00435871],eval_loss:0.0048797158524394035
    epoch:187,batch_id:10,train_loss:[0.00434503],eval_loss:0.004862834777062138
    epoch:188,batch_id:10,train_loss:[0.00433131],eval_loss:0.0048459073683867855
    epoch:189,batch_id:10,train_loss:[0.00431756],eval_loss:0.004828937196483214
    epoch:190,batch_id:10,train_loss:[0.00430378],eval_loss:0.004811928219472368
    epoch:191,batch_id:10,train_loss:[0.00428997],eval_loss:0.00479488509396712
    epoch:192,batch_id:10,train_loss:[0.00427614],eval_loss:0.0047778121661394835
    epoch:193,batch_id:10,train_loss:[0.00426227],eval_loss:0.004760703692833583
    epoch:194,batch_id:10,train_loss:[0.00424838],eval_loss:0.0047435582770655555
    epoch:195,batch_id:10,train_loss:[0.00423446],eval_loss:0.0047263773158192635
    epoch:196,batch_id:10,train_loss:[0.0042205],eval_loss:0.004709166086589296
    epoch:197,batch_id:10,train_loss:[0.00420653],eval_loss:0.0046919243565450115
    epoch:198,batch_id:10,train_loss:[0.00419253],eval_loss:0.0046746494093288975
    epoch:199,batch_id:10,train_loss:[0.0041785],eval_loss:0.004657346832876404
    Final loss: [0.0041785]



![png](output_33_1.png)


## LSTM网络

### 定义LSTM网络


```python
import paddle.fluid as fluid
class MyLSTMModel(fluid.dygraph.Layer):
    '''
    LSTM网络
    '''
    def __init__(self):
        super(MyLSTMModel,self).__init__()
        self.rnn = paddle.nn.LSTM(4, 20, 2)
        self.flatten = paddle.nn.Flatten()
        self.fc1=fluid.dygraph.Linear(5*20,10)
        self.fc2=fluid.dygraph.Linear(10,1)

        
    def forward(self,input):        # forward 定义执行实际运行时网络的执行逻辑
        '''前向计算'''
        # print('input',input.shape)
        out, (h, c)=self.rnn(input)
        out =self.flatten(out)
        out=self.fc1(out)
        out=self.fc2(out)
        return out
```

### 开始训练


```python
with fluid.dygraph.guard(place):
    model=MyLSTMModel() #模型实例化
    # model=MyModel()
    model.train() #训练模式
    # opt=fluid.optimizer.SGDOptimizer(learning_rate=0.001, parameter_list=model.parameters())#优化器选用SGD随机梯度下降，学习率为0.001.
    opt=fluid.optimizer.AdamOptimizer(learning_rate=0.01, parameter_list=model.parameters()) 
    epochs_num=100#迭代次数
    batch_size = 128*3
    train_reader = fluid.io.batch(reader=switch_reader(), batch_size=batch_size)
    val_reader = fluid.io.batch(reader=switch_reader(is_val=True), batch_size=batch_size)
    Batch=0
    Batchs=[]
    all_train_loss=[]
    all_eval_loss=[]
    for pass_num in range(epochs_num):
        for batch_id, data in enumerate(train_reader()): 
            data_x=np.array([x[0] for x in data],np.float32)
            data_y = np.array([x[1] for x in data]).astype('float32')
            data_x = fluid.dygraph.to_variable(data_x)
            data_y = fluid.dygraph.to_variable(data_y)
        
            # print(data_x.shape, data_y.shape)

            predict=model(data_x)
            # print(predict.shape)
            loss=fluid.layers.mse_loss(predict,data_y)
            avg_loss=fluid.layers.mean(loss)#获取loss值
            avg_loss.backward()       
            opt.minimize(avg_loss)    #优化器对象的minimize方法对参数进行更新 
            model.clear_gradients()   #model.clear_gradients()来重置梯度
            if batch_id!=0 and batch_id%10==0:
                Batch = Batch+10
                Batchs.append(Batch)
                all_train_loss.append(avg_loss.numpy()[0])
                evalavg_loss=[]
                for eval_data in val_reader():
                    eval_data_x = np.array([x[0] for x in eval_data],np.float32)
                    eval_data_y = np.array([x[1] for x in eval_data]).astype('float32')

                    eval_data_x = fluid.dygraph.to_variable(eval_data_x)
                    eval_data_y = fluid.dygraph.to_variable(eval_data_y)

                    eval_predict=model(eval_data_x)
                    eval_loss=fluid.layers.mse_loss(eval_predict,eval_data_y)
                    eval_loss=fluid.layers.mean(eval_loss)
                    evalavg_loss.append(eval_loss.numpy()[0])#获取loss值
                all_eval_loss.append(sum(evalavg_loss)/len(evalavg_loss))
        print("epoch:{},batch_id:{},train_loss:{},eval_loss:{}".format(pass_num,batch_id,avg_loss.numpy(),sum(evalavg_loss)/len(evalavg_loss)))     

    fluid.save_dygraph(model.state_dict(),'./model/MyLSTMModel')#保存模型
    fluid.save_dygraph(opt.state_dict(),'./model/MyLSTMModel')#保存模型
    print("Final loss: {}".format(avg_loss.numpy()))    
#让我们绘制训练图和验证损失图，以了解训练的进行情况。     
draw_train_loss(Batchs,all_train_loss,all_eval_loss, 'LSTM')
```

    epoch:0,batch_id:10,train_loss:[0.02476512],eval_loss:0.023129687023659546
    epoch:1,batch_id:10,train_loss:[0.01190875],eval_loss:0.0027992411827047667
    epoch:2,batch_id:10,train_loss:[0.00578104],eval_loss:0.007348374386007587
    epoch:3,batch_id:10,train_loss:[0.00204595],eval_loss:0.0005536427391537776
    epoch:4,batch_id:10,train_loss:[0.00154447],eval_loss:0.0006637740686225394
    epoch:5,batch_id:10,train_loss:[0.00479219],eval_loss:0.004033941503924628
    epoch:6,batch_id:10,train_loss:[0.00176596],eval_loss:0.0006612322079793861
    epoch:7,batch_id:10,train_loss:[0.00113699],eval_loss:0.0010525552206672728
    epoch:8,batch_id:10,train_loss:[0.00096475],eval_loss:0.0005845498429456105
    epoch:9,batch_id:10,train_loss:[0.00300774],eval_loss:0.002365170163102448
    epoch:10,batch_id:10,train_loss:[0.00120252],eval_loss:0.0006219494583395621
    epoch:11,batch_id:10,train_loss:[0.00083899],eval_loss:0.0005603440125317624
    epoch:12,batch_id:10,train_loss:[0.00102286],eval_loss:0.0012319269978130858
    epoch:13,batch_id:10,train_loss:[0.00426955],eval_loss:0.0034388689479480186
    epoch:14,batch_id:10,train_loss:[0.00085353],eval_loss:0.0015879209386184812
    epoch:15,batch_id:10,train_loss:[0.00080557],eval_loss:0.00068331413785927
    epoch:16,batch_id:10,train_loss:[0.00273022],eval_loss:0.0018719392052541177
    epoch:17,batch_id:10,train_loss:[0.00626801],eval_loss:0.009894635217885176
    epoch:18,batch_id:10,train_loss:[0.00109935],eval_loss:0.003970463216925661
    epoch:19,batch_id:10,train_loss:[0.00131427],eval_loss:0.0020004890005414686
    epoch:20,batch_id:10,train_loss:[0.0023917],eval_loss:0.007124613194415967
    epoch:21,batch_id:10,train_loss:[0.00307149],eval_loss:0.0024654136893029013
    epoch:22,batch_id:10,train_loss:[0.0021796],eval_loss:0.0010769719374366105
    epoch:23,batch_id:10,train_loss:[0.00259087],eval_loss:0.001912260699706773
    epoch:24,batch_id:10,train_loss:[0.00203189],eval_loss:0.0013631775315540533
    epoch:25,batch_id:10,train_loss:[0.0010464],eval_loss:0.000631379031498606
    epoch:26,batch_id:10,train_loss:[0.00116064],eval_loss:0.0008927419936905304
    epoch:27,batch_id:10,train_loss:[0.0013302],eval_loss:0.0008405626479846736
    epoch:28,batch_id:10,train_loss:[0.00114944],eval_loss:0.0006608979504865905
    epoch:29,batch_id:10,train_loss:[0.00111384],eval_loss:0.0007448670997594794
    epoch:30,batch_id:10,train_loss:[0.0010512],eval_loss:0.0006799223095489045
    epoch:31,batch_id:10,train_loss:[0.00105763],eval_loss:0.0006558604654856026
    epoch:32,batch_id:10,train_loss:[0.00097343],eval_loss:0.0005693176256803175
    epoch:33,batch_id:10,train_loss:[0.00090096],eval_loss:0.0005581089741705606
    epoch:34,batch_id:10,train_loss:[0.00088702],eval_loss:0.0005715569811097035
    epoch:35,batch_id:10,train_loss:[0.00093551],eval_loss:0.0005855638203987231
    epoch:36,batch_id:10,train_loss:[0.00084698],eval_loss:0.0005135537649039179
    epoch:37,batch_id:10,train_loss:[0.00080869],eval_loss:0.0005053362983744591
    epoch:38,batch_id:10,train_loss:[0.00077005],eval_loss:0.0005072135439453026
    epoch:39,batch_id:10,train_loss:[0.00092438],eval_loss:0.0006347331218421459
    epoch:40,batch_id:10,train_loss:[0.0008026],eval_loss:0.0005083387853422513
    epoch:41,batch_id:10,train_loss:[0.00076915],eval_loss:0.0005015839221111188
    epoch:42,batch_id:10,train_loss:[0.00072225],eval_loss:0.0005357527115847915
    epoch:43,batch_id:10,train_loss:[0.00097838],eval_loss:0.0008529478024380902
    epoch:44,batch_id:10,train_loss:[0.00076922],eval_loss:0.00048660030006431043
    epoch:45,batch_id:10,train_loss:[0.00084326],eval_loss:0.0004992615334534397
    epoch:46,batch_id:10,train_loss:[0.00073537],eval_loss:0.0007329825505924722
    epoch:47,batch_id:10,train_loss:[0.00069322],eval_loss:0.0005240654475831737
    epoch:48,batch_id:10,train_loss:[0.00099598],eval_loss:0.0010548499218809109
    epoch:49,batch_id:10,train_loss:[0.00144259],eval_loss:0.0009987241161676745
    epoch:50,batch_id:10,train_loss:[0.00081561],eval_loss:0.0010781279027772446
    epoch:51,batch_id:10,train_loss:[0.00071387],eval_loss:0.0006181684536083291
    epoch:52,batch_id:10,train_loss:[0.00117428],eval_loss:0.0010714233309651415
    epoch:53,batch_id:10,train_loss:[0.00554339],eval_loss:0.00784131713832418
    epoch:54,batch_id:10,train_loss:[0.00143748],eval_loss:0.0018618832109495997
    epoch:55,batch_id:10,train_loss:[0.0008043],eval_loss:0.0004746119666378945
    epoch:56,batch_id:10,train_loss:[0.00075131],eval_loss:0.0006996508551916728
    epoch:57,batch_id:10,train_loss:[0.00070997],eval_loss:0.0005390806181821972
    epoch:58,batch_id:10,train_loss:[0.00189351],eval_loss:0.0027249664223442474
    epoch:59,batch_id:10,train_loss:[0.00258413],eval_loss:0.0029318872839212418
    epoch:60,batch_id:10,train_loss:[0.00098062],eval_loss:0.001369148395800342
    epoch:61,batch_id:10,train_loss:[0.00082528],eval_loss:0.001035258115734905
    epoch:62,batch_id:10,train_loss:[0.0022483],eval_loss:0.0025354589840086796
    epoch:63,batch_id:10,train_loss:[0.00974505],eval_loss:0.012821685522794724
    epoch:64,batch_id:10,train_loss:[0.00131491],eval_loss:0.0011287251836620271
    epoch:65,batch_id:10,train_loss:[0.0020068],eval_loss:0.001491215235243241
    epoch:66,batch_id:10,train_loss:[0.00081396],eval_loss:0.002772449360539516
    epoch:67,batch_id:10,train_loss:[0.00073212],eval_loss:0.0005995353470401218
    epoch:68,batch_id:10,train_loss:[0.00119416],eval_loss:0.0011141194069447617
    epoch:69,batch_id:10,train_loss:[0.01428292],eval_loss:0.019625961780548096
    epoch:70,batch_id:10,train_loss:[0.00102674],eval_loss:0.0026280643651261926
    epoch:71,batch_id:10,train_loss:[0.00103887],eval_loss:0.0018194512425300975
    epoch:72,batch_id:10,train_loss:[0.00193495],eval_loss:0.0009000993256146709
    epoch:73,batch_id:10,train_loss:[0.00709763],eval_loss:0.009286629656950632
    epoch:74,batch_id:10,train_loss:[0.00072825],eval_loss:0.0018517125475530822
    epoch:75,batch_id:10,train_loss:[0.00085227],eval_loss:0.0009171417720305423
    epoch:76,batch_id:10,train_loss:[0.00128105],eval_loss:0.001684492841983835
    epoch:77,batch_id:10,train_loss:[0.00087326],eval_loss:0.0008236404974013567
    epoch:78,batch_id:10,train_loss:[0.00118945],eval_loss:0.0006329664417232076
    epoch:79,batch_id:10,train_loss:[0.00090942],eval_loss:0.0007650821741359929
    epoch:80,batch_id:10,train_loss:[0.00083706],eval_loss:0.0006686117073210577
    epoch:81,batch_id:10,train_loss:[0.0008618],eval_loss:0.0005586617529237022
    epoch:82,batch_id:10,train_loss:[0.00075319],eval_loss:0.0004434360792705168
    epoch:83,batch_id:10,train_loss:[0.00069049],eval_loss:0.0004396723040069143
    epoch:84,batch_id:10,train_loss:[0.00070669],eval_loss:0.0004420625773491338
    epoch:85,batch_id:10,train_loss:[0.00068239],eval_loss:0.00043454012969353545
    epoch:86,batch_id:10,train_loss:[0.00069615],eval_loss:0.0004331421902558456
    epoch:87,batch_id:10,train_loss:[0.00066224],eval_loss:0.0004299435095163062
    epoch:88,batch_id:10,train_loss:[0.00066545],eval_loss:0.0004246221554543202
    epoch:89,batch_id:10,train_loss:[0.00065586],eval_loss:0.00042371698752200854
    epoch:90,batch_id:10,train_loss:[0.00065759],eval_loss:0.00041867724212352186
    epoch:91,batch_id:10,train_loss:[0.00063547],eval_loss:0.00042270025975691777
    epoch:92,batch_id:10,train_loss:[0.0006432],eval_loss:0.000412944258035471
    epoch:93,batch_id:10,train_loss:[0.00063258],eval_loss:0.0004148415706974144
    epoch:94,batch_id:10,train_loss:[0.00063357],eval_loss:0.0004073873909267907
    epoch:95,batch_id:10,train_loss:[0.00061791],eval_loss:0.00041087625140789896
    epoch:96,batch_id:10,train_loss:[0.00062197],eval_loss:0.0004016706225229427
    epoch:97,batch_id:10,train_loss:[0.0006156],eval_loss:0.0004028872596488024
    epoch:98,batch_id:10,train_loss:[0.00062035],eval_loss:0.0003957119721841688
    epoch:99,batch_id:10,train_loss:[0.00059839],eval_loss:0.00040493354996821534
    Final loss: [0.00059839]



![png](output_38_1.png)


## ARIMA

### 从数据图可知，该数据不是平稳数据。 这里选择取对数的方式进行处理。


```python

# data_arma = np.log(data_pd['over'].values)
data_arma = data_pd['over'].values
# data_arma_diff = data_arma
# data_arma_diff = np.diff(data_arma)[1:]
# data_arma_diff = data_pd['over'].values
# data_arma_diff = np.diff(data_pd['over'].values)[1:]


plt.plot(data_arma)
```




    [<matplotlib.lines.Line2D at 0x7f48d6277ad0>]




![png](output_41_1.png)


### 进行ADF检验,即使在10%的水平上也不能拒绝原假设，即序列不平稳，存在单位根


```python
# ADF检验，验证是否为平稳数据
print('数据ADF检验：',sm.tsa.stattools.adfuller(data_arma))
```

    数据ADF检验： (-2.04614080742685, 0.2667153375856042, 32, 5097, {'1%': -3.431633617101692, '5%': -2.862107222331948, '10%': -2.5670719327267992}, 53183.61330543149)



```python
data_arma_diff = np.diff(data_arma)[1:]

plt.plot(data_arma_diff)
```




    [<matplotlib.lines.Line2D at 0x7f48d61cc5d0>]




![png](output_44_1.png)



```python
# ADF检验，验证是否为平稳数据
print('数据ADF检验：',sm.tsa.stattools.adfuller(data_arma_diff))
```

    数据ADF检验： (-11.867048399521366, 6.637551809333625e-22, 33, 5094, {'1%': -3.431634373441392, '5%': -2.8621075564815404, '10%': -2.5670721106071963}, 53160.13384388323)


### 进行ADF检验,拒绝了原假设，即差分后序列平稳。但从上图中看出可能存在条件异方差，故可能要用ARCH\GARCH模型进行估计。以下先用ARMA对数据进行拟合，再根据残差的情况决定GARCH的阶数

### 自相关图截尾 q=1


```python
# 画自相关图
fig = sm.graphics.tsa.plot_acf(data_arma_diff,lags=20)
```


![png](output_48_0.png)



```python
# 画偏自相关图
fig = sm.graphics.tsa.plot_pacf(data_arma_diff,lags=20)
```


![png](output_49_0.png)



```python
# ARIMA（p,1,q）模型选择
# p可选4,6；q可选0，1


# ARIMA（4,1,0）模型
arima_mod44 = sm.tsa.arima.ARIMA(data_arma_diff,order=(4,1,0)).fit()
print(arima_mod44.aic,arima_mod44.bic,arima_mod44.hqic)

# ARIMA（4,1,1）模型
arima_mod46 = sm.tsa.arima.ARIMA(data_arma_diff,order=(4,1,1)).fit()
print(arima_mod46.aic,arima_mod46.bic,arima_mod46.hqic)

# ARIMA（6,1,0）模型
arima_mod64 = sm.tsa.arima.ARIMA(data_arma_diff,order=(6,1,0)).fit()
print(arima_mod64.aic,arima_mod64.bic,arima_mod64.hqic)

# ARIMA（6,1,1）模型
arima_mod66 = sm.tsa.arima.ARIMA(data_arma_diff,order=(6,1,1)).fit()
print(arima_mod66.aic,arima_mod66.bic,arima_mod66.hqic)

# 根据AIC、BIC、HQIC最小选择ARIMA（6,1,0）
```

    54373.59481066247 54406.30619052139 54385.04508545776
    53571.4414729961 53610.6951288268 53585.18180275044
    54191.78849253015 54237.584424332636 54207.818877243546
    53558.44047029911 53610.77867807338 53576.76090997156



```python
# 模型检验,检验残差序列
arima_mod=arima_mod66
resid = arima_mod.resid
fig = plt.figure(figsize=(10,7))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(resid,ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(resid,ax=ax2)
print(sm.stats.durbin_watson(arima_mod.resid))
```

    1.99573833322835



![png](output_51_1.png)


### 从AC、PAC及durbin watson检验中发现ARIAM(6,1,0)残差平稳，较好的拟合了数据。下判断是否需要用GARCH及GARCH阶数


```python
# 模型检验,检验残差平方序列
arima_mod=arima_mod66
resid = arima_mod.resid ** 2
fig = plt.figure(figsize=(10,7))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(resid,ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(resid,ax=ax2)
print(sm.stats.durbin_watson(arima_mod.resid))
```

    1.99573833322835



![png](output_53_1.png)


### 可以看出残差是white noise process，但残差平方不是white noise process， 也即存在条件异方差，故需要用GARCH、ARCH模型进行估计


```python
am = arch.arch_model(data_arma_diff ,mean='AR',lags=6 ,vol='GARCH',p=5, q=5) 
res = am.fit()
bm = arch.arch_model(data_arma_diff ,mean='AR',lags=6 ,vol='GARCH',p=10, q=10) 
res1 = bm.fit()
```

    Iteration:      1,   Func. Count:     20,   Neg. LLF: 24963.84732146195
    Iteration:      2,   Func. Count:     44,   Neg. LLF: 24953.057457820454
    Iteration:      3,   Func. Count:     67,   Neg. LLF: 24952.1126356257
    Iteration:      4,   Func. Count:     89,   Neg. LLF: 24946.744239096057
    Iteration:      5,   Func. Count:    112,   Neg. LLF: 24945.793966014273
    Iteration:      6,   Func. Count:    135,   Neg. LLF: 24944.705297421886
    Iteration:      7,   Func. Count:    157,   Neg. LLF: 24936.749839517604
    Iteration:      8,   Func. Count:    180,   Neg. LLF: 24935.680353468106
    Iteration:      9,   Func. Count:    201,   Neg. LLF: 24931.287336890713
    Iteration:     10,   Func. Count:    223,   Neg. LLF: 24929.471473259808
    Iteration:     11,   Func. Count:    244,   Neg. LLF: 24926.87301432819
    Iteration:     12,   Func. Count:    266,   Neg. LLF: 24926.240820181963
    Iteration:     13,   Func. Count:    287,   Neg. LLF: 24923.438741234408
    Iteration:     14,   Func. Count:    309,   Neg. LLF: 24922.609347208458
    Iteration:     15,   Func. Count:    330,   Neg. LLF: 24921.308788653605
    Iteration:     16,   Func. Count:    351,   Neg. LLF: 24920.296593913023
    Iteration:     17,   Func. Count:    372,   Neg. LLF: 24918.13059353007
    Iteration:     18,   Func. Count:    393,   Neg. LLF: 24916.201477674906
    Iteration:     19,   Func. Count:    414,   Neg. LLF: 24915.290073418328
    Iteration:     20,   Func. Count:    434,   Neg. LLF: 24910.74370922013
    Iteration:     21,   Func. Count:    456,   Neg. LLF: 24910.639306825826
    Iteration:     22,   Func. Count:    477,   Neg. LLF: 24909.90375172726
    Iteration:     23,   Func. Count:    497,   Neg. LLF: 24909.474530626925
    Iteration:     24,   Func. Count:    517,   Neg. LLF: 24909.08877834795
    Iteration:     25,   Func. Count:    537,   Neg. LLF: 24909.011777477972
    Iteration:     26,   Func. Count:    557,   Neg. LLF: 24909.00615297546
    Iteration:     27,   Func. Count:    577,   Neg. LLF: 24909.00571139809
    Iteration:     28,   Func. Count:    597,   Neg. LLF: 24909.005701216032
    Iteration:     29,   Func. Count:    621,   Neg. LLF: 24909.005696985063
    Iteration:     30,   Func. Count:    645,   Neg. LLF: 24909.00569597683
    Optimization terminated successfully.    (Exit mode 0)
                Current function value: 24909.005696384924
                Iterations: 31
                Function evaluations: 650
                Gradient evaluations: 30
    Iteration:      1,   Func. Count:     30,   Neg. LLF: 25016.66706436775
    Iteration:      2,   Func. Count:     64,   Neg. LLF: 25005.647021083605
    Iteration:      3,   Func. Count:     97,   Neg. LLF: 25003.60137524762
    Iteration:      4,   Func. Count:    129,   Neg. LLF: 24990.531970423406
    Iteration:      5,   Func. Count:    161,   Neg. LLF: 24978.3943439101
    Iteration:      6,   Func. Count:    193,   Neg. LLF: 24965.950592109257
    Iteration:      7,   Func. Count:    225,   Neg. LLF: 24943.74672127231
    Iteration:      8,   Func. Count:    259,   Neg. LLF: 24943.39193222244
    Iteration:      9,   Func. Count:    291,   Neg. LLF: 24935.036383393475
    Iteration:     10,   Func. Count:    323,   Neg. LLF: 24930.778342224607
    Iteration:     11,   Func. Count:    354,   Neg. LLF: 24922.566818935535
    Iteration:     12,   Func. Count:    386,   Neg. LLF: 24918.843920004285
    Iteration:     13,   Func. Count:    418,   Neg. LLF: 24917.258349241863
    Iteration:     14,   Func. Count:    450,   Neg. LLF: 24916.66031571151
    Iteration:     15,   Func. Count:    482,   Neg. LLF: 24915.577341932803
    Iteration:     16,   Func. Count:    514,   Neg. LLF: 24914.884421560848
    Iteration:     17,   Func. Count:    545,   Neg. LLF: 24911.777873032945
    Iteration:     18,   Func. Count:    577,   Neg. LLF: 24911.06644055379
    Iteration:     19,   Func. Count:    609,   Neg. LLF: 24910.483402733487
    Iteration:     20,   Func. Count:    640,   Neg. LLF: 24908.67662993905
    Iteration:     21,   Func. Count:    671,   Neg. LLF: 24907.706920907465
    Iteration:     22,   Func. Count:    702,   Neg. LLF: 24906.390439621573
    Iteration:     23,   Func. Count:    733,   Neg. LLF: 24905.575784852474
    Iteration:     24,   Func. Count:    765,   Neg. LLF: 24905.50338447447
    Iteration:     25,   Func. Count:    797,   Neg. LLF: 24905.404299919668
    Iteration:     26,   Func. Count:    828,   Neg. LLF: 24904.94235800464
    Iteration:     27,   Func. Count:    858,   Neg. LLF: 24902.423732223353
    Iteration:     28,   Func. Count:    889,   Neg. LLF: 24901.413809372993
    Iteration:     29,   Func. Count:    921,   Neg. LLF: 24901.307428303975
    Iteration:     30,   Func. Count:    952,   Neg. LLF: 24901.093238422523
    Iteration:     31,   Func. Count:    983,   Neg. LLF: 24900.605942028647
    Iteration:     32,   Func. Count: 1.01e+03,   Neg. LLF: 24900.48258238775
    Iteration:     33,   Func. Count: 1.04e+03,   Neg. LLF: 24900.426364952735
    Iteration:     34,   Func. Count: 1.07e+03,   Neg. LLF: 24900.355617075682
    Iteration:     35,   Func. Count: 1.1e+03,   Neg. LLF: 24900.317390899752
    Iteration:     36,   Func. Count: 1.13e+03,   Neg. LLF: 24900.26301688972
    Iteration:     37,   Func. Count: 1.16e+03,   Neg. LLF: 24900.13041975014
    Iteration:     38,   Func. Count: 1.19e+03,   Neg. LLF: 24900.10578352202
    Iteration:     39,   Func. Count: 1.22e+03,   Neg. LLF: 24900.098338892894
    Iteration:     40,   Func. Count: 1.25e+03,   Neg. LLF: 24900.095124257146
    Iteration:     41,   Func. Count: 1.28e+03,   Neg. LLF: 24900.094429287634
    Iteration:     42,   Func. Count: 1.31e+03,   Neg. LLF: 24900.094268230412
    Iteration:     43,   Func. Count: 1.34e+03,   Neg. LLF: 24900.09424052398
    Iteration:     44,   Func. Count: 1.37e+03,   Neg. LLF: 24900.094238770398
    Optimization terminated successfully.    (Exit mode 0)
                Current function value: 24900.094238929323
                Iterations: 44
                Function evaluations: 1385
                Gradient evaluations: 44



```python
res.summary()
```




<table class="simpletable">
<caption>AR - GARCH Model Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>y</td>         <th>  R-squared:         </th>  <td>   0.009</td> 
</tr>
<tr>
  <th>Mean Model:</th>            <td>AR</td>         <th>  Adj. R-squared:    </th>  <td>   0.008</td> 
</tr>
<tr>
  <th>Vol Model:</th>            <td>GARCH</td>       <th>  Log-Likelihood:    </th> <td>  -24909.0</td>
</tr>
<tr>
  <th>Distribution:</th>        <td>Normal</td>       <th>  AIC:               </th> <td>   49854.0</td>
</tr>
<tr>
  <th>Method:</th>        <td>Maximum Likelihood</td> <th>  BIC:               </th> <td>   49971.8</td>
</tr>
<tr>
  <th></th>                        <td></td>          <th>  No. Observations:  </th>    <td>5122</td>   
</tr>
<tr>
  <th>Date:</th>           <td>Fri, Apr 30 2021</td>  <th>  Df Residuals:      </th>    <td>5115</td>   
</tr>
<tr>
  <th>Time:</th>               <td>14:55:07</td>      <th>  Df Model:          </th>      <td>7</td>    
</tr>
</table>
<table class="simpletable">
<caption>Mean Model</caption>
<tr>
    <td></td>       <th>coef</th>     <th>std err</th>      <th>t</th>       <th>P>|t|</th>      <th>95.0% Conf. Int.</th>    
</tr>
<tr>
  <th>Const</th> <td>    0.4067</td> <td>    0.373</td> <td>    1.090</td> <td>    0.276</td>    <td>[ -0.325,  1.138]</td>   
</tr>
<tr>
  <th>y[1]</th>  <td>    0.0210</td> <td>1.617e-02</td> <td>    1.299</td> <td>    0.194</td> <td>[-1.069e-02,5.269e-02]</td> 
</tr>
<tr>
  <th>y[2]</th>  <td>   -0.0139</td> <td>1.608e-02</td> <td>   -0.866</td> <td>    0.386</td> <td>[-4.544e-02,1.759e-02]</td> 
</tr>
<tr>
  <th>y[3]</th>  <td>    0.0326</td> <td>1.579e-02</td> <td>    2.067</td> <td>3.874e-02</td>  <td>[1.689e-03,6.358e-02]</td> 
</tr>
<tr>
  <th>y[4]</th>  <td>8.8470e-03</td> <td>1.705e-02</td> <td>    0.519</td> <td>    0.604</td> <td>[-2.456e-02,4.226e-02]</td> 
</tr>
<tr>
  <th>y[5]</th>  <td>   -0.0107</td> <td>1.600e-02</td> <td>   -0.668</td> <td>    0.504</td> <td>[-4.204e-02,2.067e-02]</td> 
</tr>
<tr>
  <th>y[6]</th>  <td>   -0.0500</td> <td>1.537e-02</td> <td>   -3.254</td> <td>1.136e-03</td> <td>[-8.016e-02,-1.990e-02]</td>
</tr>
</table>
<table class="simpletable">
<caption>Volatility Model</caption>
<tr>
      <td></td>        <th>coef</th>     <th>std err</th>      <th>t</th>       <th>P>|t|</th>     <th>95.0% Conf. Int.</th>  
</tr>
<tr>
  <th>omega</th>    <td>   25.0806</td> <td>    8.693</td> <td>    2.885</td> <td>3.911e-03</td>   <td>[  8.043, 42.118]</td> 
</tr>
<tr>
  <th>alpha[1]</th> <td>    0.0778</td> <td>2.797e-02</td> <td>    2.783</td> <td>5.392e-03</td>  <td>[2.301e-02,  0.133]</td>
</tr>
<tr>
  <th>alpha[2]</th> <td>    0.0833</td> <td>2.816e-02</td> <td>    2.956</td> <td>3.116e-03</td>  <td>[2.806e-02,  0.138]</td>
</tr>
<tr>
  <th>alpha[3]</th> <td>    0.0529</td> <td>3.559e-02</td> <td>    1.487</td> <td>    0.137</td> <td>[-1.685e-02,  0.123]</td>
</tr>
<tr>
  <th>alpha[4]</th> <td>    0.0471</td> <td>4.299e-02</td> <td>    1.095</td> <td>    0.274</td> <td>[-3.719e-02,  0.131]</td>
</tr>
<tr>
  <th>alpha[5]</th> <td>    0.0812</td> <td>2.186e-02</td> <td>    3.716</td> <td>2.020e-04</td>  <td>[3.840e-02,  0.124]</td>
</tr>
<tr>
  <th>beta[1]</th>  <td>4.6458e-15</td> <td>    0.127</td> <td>3.672e-14</td> <td>    1.000</td>   <td>[ -0.248,  0.248]</td> 
</tr>
<tr>
  <th>beta[2]</th>  <td>5.9239e-16</td> <td>    0.138</td> <td>4.305e-15</td> <td>    1.000</td>   <td>[ -0.270,  0.270]</td> 
</tr>
<tr>
  <th>beta[3]</th>  <td>2.4385e-15</td> <td>    0.168</td> <td>1.450e-14</td> <td>    1.000</td>   <td>[ -0.330,  0.330]</td> 
</tr>
<tr>
  <th>beta[4]</th>  <td>1.7422e-15</td> <td>    0.198</td> <td>8.786e-15</td> <td>    1.000</td>   <td>[ -0.389,  0.389]</td> 
</tr>
<tr>
  <th>beta[5]</th>  <td>    0.6577</td> <td>9.498e-02</td> <td>    6.924</td> <td>4.382e-12</td>   <td>[  0.472,  0.844]</td> 
</tr>
</table><br/><br/>Covariance estimator: robust




```python
res1.summary()
```




<table class="simpletable">
<caption>AR - GARCH Model Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>y</td>         <th>  R-squared:         </th>  <td>   0.008</td> 
</tr>
<tr>
  <th>Mean Model:</th>            <td>AR</td>         <th>  Adj. R-squared:    </th>  <td>   0.007</td> 
</tr>
<tr>
  <th>Vol Model:</th>            <td>GARCH</td>       <th>  Log-Likelihood:    </th> <td>  -24900.1</td>
</tr>
<tr>
  <th>Distribution:</th>        <td>Normal</td>       <th>  AIC:               </th> <td>   49856.2</td>
</tr>
<tr>
  <th>Method:</th>        <td>Maximum Likelihood</td> <th>  BIC:               </th> <td>   50039.3</td>
</tr>
<tr>
  <th></th>                        <td></td>          <th>  No. Observations:  </th>    <td>5122</td>   
</tr>
<tr>
  <th>Date:</th>           <td>Fri, Apr 30 2021</td>  <th>  Df Residuals:      </th>    <td>5115</td>   
</tr>
<tr>
  <th>Time:</th>               <td>14:55:08</td>      <th>  Df Model:          </th>      <td>7</td>    
</tr>
</table>
<table class="simpletable">
<caption>Mean Model</caption>
<tr>
    <td></td>       <th>coef</th>     <th>std err</th>      <th>t</th>       <th>P>|t|</th>      <th>95.0% Conf. Int.</th>    
</tr>
<tr>
  <th>Const</th> <td>    0.3371</td> <td>    0.668</td> <td>    0.505</td> <td>    0.614</td>    <td>[ -0.972,  1.646]</td>   
</tr>
<tr>
  <th>y[1]</th>  <td>    0.0183</td> <td>2.228e-02</td> <td>    0.820</td> <td>    0.412</td> <td>[-2.539e-02,6.195e-02]</td> 
</tr>
<tr>
  <th>y[2]</th>  <td>   -0.0142</td> <td>1.610e-02</td> <td>   -0.882</td> <td>    0.378</td> <td>[-4.576e-02,1.735e-02]</td> 
</tr>
<tr>
  <th>y[3]</th>  <td>    0.0284</td> <td>2.264e-02</td> <td>    1.255</td> <td>    0.209</td> <td>[-1.596e-02,7.277e-02]</td> 
</tr>
<tr>
  <th>y[4]</th>  <td>7.1731e-03</td> <td>2.154e-02</td> <td>    0.333</td> <td>    0.739</td> <td>[-3.504e-02,4.939e-02]</td> 
</tr>
<tr>
  <th>y[5]</th>  <td>   -0.0104</td> <td>1.756e-02</td> <td>   -0.591</td> <td>    0.555</td> <td>[-4.480e-02,2.405e-02]</td> 
</tr>
<tr>
  <th>y[6]</th>  <td>   -0.0526</td> <td>1.580e-02</td> <td>   -3.330</td> <td>8.676e-04</td> <td>[-8.359e-02,-2.165e-02]</td>
</tr>
</table>
<table class="simpletable">
<caption>Volatility Model</caption>
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>       <th>P>|t|</th>    <th>95.0% Conf. Int.</th>  
</tr>
<tr>
  <th>omega</th>     <td>   36.2129</td> <td>   43.575</td> <td>    0.831</td> <td>    0.406</td> <td>[-49.192,1.216e+02]</td>
</tr>
<tr>
  <th>alpha[1]</th>  <td>    0.0907</td> <td>4.509e-02</td> <td>    2.013</td> <td>4.415e-02</td> <td>[2.376e-03,  0.179]</td>
</tr>
<tr>
  <th>alpha[2]</th>  <td>    0.0878</td> <td>    0.216</td> <td>    0.407</td> <td>    0.684</td>  <td>[ -0.335,  0.510]</td> 
</tr>
<tr>
  <th>alpha[3]</th>  <td>    0.0683</td> <td>    0.208</td> <td>    0.328</td> <td>    0.743</td>  <td>[ -0.340,  0.476]</td> 
</tr>
<tr>
  <th>alpha[4]</th>  <td>    0.0576</td> <td>    0.102</td> <td>    0.565</td> <td>    0.572</td>  <td>[ -0.142,  0.257]</td> 
</tr>
<tr>
  <th>alpha[5]</th>  <td>    0.0507</td> <td>    0.242</td> <td>    0.209</td> <td>    0.834</td>  <td>[ -0.425,  0.526]</td> 
</tr>
<tr>
  <th>alpha[6]</th>  <td>    0.0105</td> <td>    0.152</td> <td>6.906e-02</td> <td>    0.945</td>  <td>[ -0.287,  0.308]</td> 
</tr>
<tr>
  <th>alpha[7]</th>  <td>    0.0324</td> <td>    0.112</td> <td>    0.291</td> <td>    0.771</td>  <td>[ -0.186,  0.251]</td> 
</tr>
<tr>
  <th>alpha[8]</th>  <td>7.6984e-04</td> <td>    0.253</td> <td>3.039e-03</td> <td>    0.998</td>  <td>[ -0.496,  0.497]</td> 
</tr>
<tr>
  <th>alpha[9]</th>  <td>    0.0108</td> <td>7.088e-02</td> <td>    0.152</td> <td>    0.879</td>  <td>[ -0.128,  0.150]</td> 
</tr>
<tr>
  <th>alpha[10]</th> <td>    0.0787</td> <td>    0.514</td> <td>    0.153</td> <td>    0.878</td>  <td>[ -0.929,  1.087]</td> 
</tr>
<tr>
  <th>beta[1]</th>   <td>1.4416e-16</td> <td>    1.826</td> <td>7.896e-17</td> <td>    1.000</td>  <td>[ -3.578,  3.578]</td> 
</tr>
<tr>
  <th>beta[2]</th>   <td>3.3815e-17</td> <td>    3.071</td> <td>1.101e-17</td> <td>    1.000</td>  <td>[ -6.018,  6.018]</td> 
</tr>
<tr>
  <th>beta[3]</th>   <td>1.4986e-16</td> <td>    1.846</td> <td>8.119e-17</td> <td>    1.000</td>  <td>[ -3.618,  3.618]</td> 
</tr>
<tr>
  <th>beta[4]</th>   <td>1.1193e-16</td> <td>    2.266</td> <td>4.939e-17</td> <td>    1.000</td>  <td>[ -4.442,  4.442]</td> 
</tr>
<tr>
  <th>beta[5]</th>   <td>    0.2037</td> <td>    2.395</td> <td>8.503e-02</td> <td>    0.932</td>  <td>[ -4.491,  4.899]</td> 
</tr>
<tr>
  <th>beta[6]</th>   <td>1.9812e-16</td> <td>    1.587</td> <td>1.249e-16</td> <td>    1.000</td>  <td>[ -3.110,  3.110]</td> 
</tr>
<tr>
  <th>beta[7]</th>   <td>6.9466e-17</td> <td>    2.205</td> <td>3.150e-17</td> <td>    1.000</td>  <td>[ -4.322,  4.322]</td> 
</tr>
<tr>
  <th>beta[8]</th>   <td>2.6104e-16</td> <td>    1.184</td> <td>2.205e-16</td> <td>    1.000</td>  <td>[ -2.320,  2.320]</td> 
</tr>
<tr>
  <th>beta[9]</th>   <td>2.8166e-16</td> <td>    2.371</td> <td>1.188e-16</td> <td>    1.000</td>  <td>[ -4.648,  4.648]</td> 
</tr>
<tr>
  <th>beta[10]</th>  <td>    0.3080</td> <td>    1.989</td> <td>    0.155</td> <td>    0.877</td>  <td>[ -3.591,  4.207]</td> 
</tr>
</table><br/><br/>Covariance estimator: robust




```python
data_arma = np.log(data_pd['over'].values)
arima_mod10 = sm.tsa.arima.ARIMA(data_arma,order=(1,1,0)).fit()
arima_mod=arima_mod10
train_arma = arima_mod.predict(7, data_pd.shape[0])
train_arma = math.e**(train_arma)
```

### 对GARCH(5,0,5) 、GARCH(10,0,10)的比较可以发现，两者的拟合后残差平方自回归的R-squre都很小，也即残差不再存在条件异方差。相比之下GARCH(5,0,5)的AIC和BIC都较小，故选择GARCH(5,0,5)作为波动率模型


```python
res.params
```




    Const       4.067343e-01
    y[1]        2.100077e-02
    y[2]       -1.392534e-02
    y[3]        3.263518e-02
    y[4]        8.846990e-03
    y[5]       -1.068572e-02
    y[6]       -5.002933e-02
    omega       2.508061e+01
    alpha[1]    7.783020e-02
    alpha[2]    8.325697e-02
    alpha[3]    5.291409e-02
    alpha[4]    4.705620e-02
    alpha[5]    8.124777e-02
    beta[1]     4.645844e-15
    beta[2]     5.923945e-16
    beta[3]     2.438490e-15
    beta[4]     1.742171e-15
    beta[5]     6.576948e-01
    Name: params, dtype: float64



$y_t = 0.40673 + 0.02100y_{t-1} -0.01393y_{t-2} + 0.03264y_{t-3}+0.00885y_{t-4}-0.01069y_{t-5}-0.05003y_{t-6}$

$\sigma^2_t = 0.07783 \sigma^2_{t-1} + 0.08326 \sigma^2_{t-2} + 0.05291 \sigma^2_{t-3} + 0.04706 \sigma^2_{t-4} + 0.08125 \sigma^2_{t-5} + 0.65769 h_{t-5}$

## 训练结果

### 预测结果

#### 数据集还原函数

> 开始时标准化的数据需要在实际作图时还原


```python
def inverse_transform(arr):
    zero_arr = np.zeros((arr.shape[0], 3))
    zero_data = np.hstack((arr, zero_arr))
    # print(zero_data)
    zero_data = scaler.inverse_transform(zero_data)
    # print(zero_data)
    
    return zero_data[:, 0]

```

#### 训练集数据预测结果


```python
with fluid.dygraph.guard(place):
    accs = []
    dnn_predict = []
    model_lstm_dict, _ = fluid.load_dygraph('./model/MyLSTMModel.pdopt')
    model_dnn_dict, _ = fluid.load_dygraph('./model/MyDNNModel.pdopt')
    model_lstm = MyLSTMModel()
    model_dnn = MyDNNModel()
    model_lstm.load_dict(model_lstm_dict)
    model_dnn.load_dict(model_dnn_dict) #加载模型参数

    train_reader = fluid.io.batch(reader=switch_reader(is_val=False), batch_size=train.shape[0])
    train_data_x = np.array([x[0] for x in next(train_reader())],np.float32)
    train_data_y = np.array([x[1] for x in next(train_reader())]).astype('float32')

    train_data_x = fluid.dygraph.to_variable(train_data_x)
    train_data_y = fluid.dygraph.to_variable(train_data_y)

    train_dnn=model_dnn(train_data_x)
    train_lstm=model_lstm(train_data_x)

plt.figure(figsize=(10, 6))
title="Train predict"
plt.title(title, fontsize=24)
plt.xlabel("time", fontsize=14)
plt.ylabel("value", fontsize=14)
plt.plot(range(train.shape[0]), inverse_transform(train_dnn.numpy()), color='red', label='dnn')
plt.plot(range(train.shape[0]), inverse_transform(train_lstm.numpy()), color='g', label='LSTM')
plt.plot(range(train.shape[0]), inverse_transform(train_data_y.numpy()), color='b', label='eval')
plt.plot(range(train.shape[0]), train_arma[:split_boundary], label='arima')
plt.legend()
plt.grid()
plt.savefig('./img/train_res.png', dpi=300)
plt.show()
```


![png](output_67_0.png)


#### 验证集预测结果


```python
with fluid.dygraph.guard(place):
    accs = []
    dnn_predict = []
    model_lstm_dict, _ = fluid.load_dygraph('./model/MyLSTMModel.pdopt')
    model_dnn_dict, _ = fluid.load_dygraph('./model/MyDNNModel.pdopt')
    model_lstm = MyLSTMModel()
    model_dnn = MyDNNModel()
    model_lstm.load_dict(model_lstm_dict)
    model_dnn.load_dict(model_dnn_dict) #加载模型参数

    val_reader = fluid.io.batch(reader=switch_reader(is_val=True), batch_size=test.shape[0])
    # print(len(next(val_reader())))
    eval_data_x = np.array([x[0] for x in next(val_reader())],np.float32)
    eval_data_y = np.array([x[1] for x in next(val_reader())]).astype('float32')

    eval_data_x = fluid.dygraph.to_variable(eval_data_x)
    eval_data_y = fluid.dygraph.to_variable(eval_data_y)

    eval_dnn=model_dnn(eval_data_x)
    eval_lstm=model_lstm(eval_data_x)

plt.figure(figsize=(10, 6))
title="Test predict"
plt.title(title, fontsize=24)
plt.xlabel("time", fontsize=14)
plt.ylabel("value", fontsize=14)
plt.plot(range(test.shape[0]), inverse_transform(eval_dnn.numpy()), color='red', label='dnn')
plt.plot(range(test.shape[0]), inverse_transform(eval_lstm.numpy()), color='g', label='LSTM')
plt.plot(range(test.shape[0]), inverse_transform(eval_data_y.numpy()), color='b', label='eval')
plt.plot(range(test.shape[0]), train_arma[split_boundary:], label='arima')
plt.legend()
plt.grid()
plt.savefig('./img/test_res.png', dpi=300)
plt.show()

```


![png](output_69_0.png)


## 模型改进

- 神经网络模型每次训练结果具有差异性(LSTM稳定性较好一点，DNN稳定性较差)，预测结果不理想时可以尝试再次训练
- 同时模型结构未做很多尝试，在训练过程中尝试改变模型参数(模型层数，节点数量等)发现预测效果好上不少，应该还有很大的提升空间，小伙伴们可以尝试一下
- GARCH模型的其他阶数也可以进行更多的尝试

## 关于作者

### Xiyan Li

> 西安交通大学 金禾经济研究中心 本科在读

- Mail: [qq2542725243@stu.xjtu.edu.cn](mailto:qq2542725243@stu.xjtu.edu.cn)

### Asterisk(Fan Guofan)

> 西安交通大学 生物技术(生物信息学方向) 本科在读

- Mail: [fanguofan@asteriscum.cn](mailto:fanguofan@asteriscum.cn)

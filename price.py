# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.preprocessing import MinMaxScaler
import time

# 导入数据
data = pd.read_csv('data/data_stocks.csv')

data.head()
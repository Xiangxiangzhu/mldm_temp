import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# 假设您的Excel文件名为'dataset.xlsx'，并且位于您的工作目录中
file_path = 'data_DM.xlsx'

# 读取Excel文件
df = pd.read_excel(file_path)

## 1. 处理缺失值 - 填充缺失值，这里使用列的中位数
df.fillna(df.median(), inplace=True)

# ## 2. 去除异常值 - 假设异常值定义为某些列的数值超出99.5%分位数或低于0.5%分位数
# for column in df.select_dtypes(include=[np.number]).columns:
#     lower_bound = df[column].quantile(0.005)
#     upper_bound = df[column].quantile(0.995)
#     df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

## 3. 数据归一化 - 使用MinMaxScaler
scaler = MinMaxScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# 查看处理后的数据
df_normalized.head()





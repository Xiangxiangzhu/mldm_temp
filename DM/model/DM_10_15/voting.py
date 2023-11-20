import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. 数据加载与预处理
file_path = 'data_DM.xlsx'
df = pd.read_excel(file_path)

## 选择特征和重新定义标签为二分类问题
features_temp = df.drop(["患病时间-5-10-", "TBA", "PCT", "PLCR", "AFU"], axis=1)
features_temp = features_temp.dropna()

#########
test_file_path_ = 'data_DM_test.xlsx'
df_test_origin_ = pd.read_excel(test_file_path_)
df_test_ = df_test_origin_.drop(["患病时间-5-10-", "LDH1", "MG", "PCT", "PLCR", "AFU"], axis=1)
df_test_ = df_test_.dropna()
#########
feature_concate = pd.concat([features_temp, df_test_], axis=0).reset_index(drop=True)

labels = feature_concate["患病时间-10-15-"].replace({1: 0, 2: 1, 3: 1})
feature_concate = feature_concate.drop(["患病时间-10-15-"], axis=1)

## 数据预处理
scaler = StandardScaler()
features_scaled = scaler.fit_transform(feature_concate)

test_file_path = 'data_DM_test.xlsx'
df_test_origin = pd.read_excel(test_file_path)
df_test = df_test_origin.drop(["患病时间-10-15-", "LDH1", "MG", "PCT", "PLCR", "AFU"], axis=1)
df_test = df_test.dropna()
test_features = df_test.drop(["患病时间-5-10-"], axis=1)
# test_features = test_features[['OSM', 'CRE', 'BUNCREA', 'eGFR（分组指标）']]
test_labels = df_test["患病时间-5-10-"].replace({1: 0, 2: 1, 3: 1})
test_features_scaled = scaler.transform(test_features)

X_test = test_features_scaled
y_test = test_labels

# 加载模型
model1 = load('28-1-0-6.joblib')
model2 = load('29-0-2-4.joblib')

# 假设 model1 和 model2 是已经训练好的分类器模型
# 并且 X_test 是测试数据集

# 获取每个模型对测试集的类别概率预测
probabilities_model1 = model1.predict_proba(X_test)
probabilities_model2 = model2.predict_proba(X_test)

# 计算平均概率
average_probabilities = (probabilities_model1 + probabilities_model2) / 2

# 根据平均概率获得最终预测类别
# 如果是二分类，通常使用 0.5 作为阈值
predictions = np.argmax(average_probabilities, axis=1)

# 打印分类报告
print(classification_report(y_test, predictions))

# 打印准确率
print("Accuracy:", accuracy_score(y_test, predictions))

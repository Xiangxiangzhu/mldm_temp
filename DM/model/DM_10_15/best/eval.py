from joblib import load
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 加载标准化处理器
scaler = load('scaler.joblib')

# 加载特征选择的索引
indices = load('indices.joblib')

# 加载训练好的模型
xgb_model = load('29-0-0-6.joblib')


# 对新的数据集应用同样的预处理和特征选择
def preprocess_and_select_features(X_new):
    # 应用标准化
    X_new_scaled = scaler.transform(X_new)
    # 应用特征选择
    X_new_selected = X_new_scaled[:, indices]
    return X_new_selected


test_file_path = 'data_DM_test.xlsx'
df_test_origin = pd.read_excel(test_file_path)
df_test = df_test_origin.drop(["患病时间-5-10-", "LDH1", "MG", "PCT", "PLCR", "AFU"], axis=1)
df_test = df_test.dropna()
test_features = df_test.drop(["患病时间-10-15-"], axis=1)
test_labels = df_test["患病时间-10-15-"].replace({1: 0, 2: 1, 3: 1})

# 示例：对新数据集进行预处理和预测
# 这里假设 new_data 是一个新的原始数据集，需要与训练集进行相同的预处理
new_data_processed = preprocess_and_select_features(test_features)
new_predictions = xgb_model.predict(new_data_processed)

cm = confusion_matrix(new_predictions, test_labels)

# 打印混淆矩阵
print("Confusion Matrix:")
print(cm)

# 使用Seaborn绘制热图以更直观地显示混淆矩阵
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

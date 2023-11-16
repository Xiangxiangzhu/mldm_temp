import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import matplotlib.pyplot as plt

# 1. 数据加载与预处理
file_path = 'data_DM.xlsx'
df = pd.read_excel(file_path)

## 选择特征和重新定义标签为二分类问题
features = df.drop(["患病时间-10-15-", "患病时间-5-10-", "TBA", "PCT", "PLCR", "AFU"], axis=1)
labels = df["患病时间-10-15-"].replace({1: 0, 2: 1, 3: 1})

## 数据预处理
features.fillna(features.median(), inplace=True)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 加载测试数据并预处理
test_file_path = 'data_DM_test.xlsx'
df_test_origin = pd.read_excel(test_file_path)
df_test = df_test_origin.drop(["患病时间-5-10-", "LDH1", "MG", "PCT", "PLCR", "AFU"], axis=1)
df_test = df_test.dropna()
test_features = df_test.drop(["患病时间-10-15-"], axis=1)
test_labels = df_test["患病时间-10-15-"].replace({1: 0, 2: 1, 3: 1})
test_features_scaled = scaler.transform(test_features)

# 5. 评估模型性能
loaded_model = load('best_model.joblib')
y_pred_test = loaded_model.predict(test_features_scaled)
y_prob_test = loaded_model.predict_proba(test_features_scaled)[:, 1]
print("Classification Report on Test Data:")
print(classification_report(test_labels, y_pred_test))
print("Confusion Matrix:")
print(confusion_matrix(test_labels, y_pred_test))
fpr_test, tpr_test, _ = roc_curve(test_labels, y_prob_test)
roc_auc_test = auc(fpr_test, tpr_test)
plt.figure()
plt.plot(fpr_test, tpr_test, label=f'Test ROC curve (area = {roc_auc_test:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve on Test Data')
plt.legend(loc="lower right")
plt.show()

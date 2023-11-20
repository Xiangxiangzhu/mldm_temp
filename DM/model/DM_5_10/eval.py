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
xgb_model = load('temp.joblib')


# 对新的数据集应用同样的预处理和特征选择
def preprocess_and_select_features(X_new):
    # 应用标准化
    X_new_scaled = scaler.transform(X_new)
    # 应用特征选择
    X_new_selected = X_new_scaled[:, indices]
    return X_new_selected



## 模型预评估
test_file_path = 'data_DM_test.xlsx'
df_test_origin = pd.read_excel(test_file_path)
df_test = df_test_origin.drop(["患病时间-10-15-", "LDH1", "MG", "PCT", "PLCR", "AFU"], axis=1)
df_test = df_test.dropna()
test_features = df_test.drop(["患病时间-5-10-"], axis=1)
# test_features = test_features[['OSM', 'CRE', 'BUNCREA', 'eGFR（分组指标）']]
test_labels = df_test["患病时间-5-10-"].replace({1: 0, 2: 1, 3: 1})
test_features_scaled = scaler.transform(test_features)
test_features_scaled = test_features_scaled[:, indices]


def evaluate_and_plot(model, X_test_, y_test_, title):
    from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
    # 预测并打印分类报告
    y_pred_test = model.predict(X_test_)
    print(f"Classification Report for {title}:")
    print(classification_report(y_test_, y_pred_test))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_, y_pred_test))

    # 绘制ROC曲线
    y_prob_test = model.predict_proba(X_test_)[:, 1]
    fpr_test, tpr_test, _ = roc_curve(y_test_, y_prob_test)
    roc_auc_test = auc(fpr_test, tpr_test)
    plt.figure()
    plt.plot(fpr_test, tpr_test, label=f'ROC curve for {title} (area = {roc_auc_test:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {title}')
    plt.legend(loc="lower right")
    plt.show()


evaluate_and_plot(xgb_model, test_features_scaled, test_labels, "DM-5-10")

# cm = confusion_matrix(new_predictions, test_labels)

# 打印混淆矩阵
# print("Confusion Matrix:")
# print(cm)

# 使用Seaborn绘制热图以更直观地显示混淆矩阵
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.title('Confusion Matrix')
# plt.show()

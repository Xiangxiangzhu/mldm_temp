import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.pipeline import Pipeline
from joblib import dump, load
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

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


def evaluate_and_plot(model, X_test, y_test, title):
    # 预测并打印分类报告
    y_pred_test = model.predict(X_test)
    print(f"Classification Report for {title}:")
    print(classification_report(y_test, y_pred_test))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_test))

    # 绘制ROC曲线
    y_prob_test = model.predict_proba(X_test)[:, 1]
    fpr_test, tpr_test, _ = roc_curve(y_test, y_prob_test)
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


# 4. 加载测试数据并预处理
test_file_path = 'data_DM_test.xlsx'
df_test_origin = pd.read_excel(test_file_path)
df_test = df_test_origin.drop(["患病时间-5-10-", "LDH1", "MG", "PCT", "PLCR", "AFU"], axis=1)
df_test = df_test.dropna()
test_features = df_test.drop(["患病时间-10-15-"], axis=1)
test_labels = df_test["患病时间-10-15-"].replace({1: 0, 2: 1, 3: 1})
test_features_scaled = scaler.transform(test_features)

# 5. 评估模型性能
rf_smoteenn = load('rf_smoteenn.joblib')
# evaluate_and_plot(rf_smoteenn, test_features_scaled, test_labels, 'SMOTEENN Model on Test Data')
rf_smotetomek = load('rf_smotetomek.joblib')
# evaluate_and_plot(rf_smotetomek, test_features_scaled, test_labels, 'SMOTETomek Model on Test Data')


# 获取每个模型的预测概率
prob_smoteenn = rf_smoteenn.predict_proba(test_features_scaled)
prob_smotetomek = rf_smotetomek.predict_proba(test_features_scaled)

# 计算平均概率
avg_prob = (prob_smoteenn + prob_smotetomek) / 2

# 基于平均概率做出最终预测
y_pred_test = np.argmax(avg_prob, axis=1)

# 在测试数据上评估平均概率集成模型
print("Classification Report on Test Data:")
print(classification_report(test_labels, y_pred_test))
print("Confusion Matrix:")
print(confusion_matrix(test_labels, y_pred_test))
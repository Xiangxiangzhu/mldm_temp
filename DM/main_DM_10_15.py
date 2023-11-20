import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import ADASYN, SMOTE
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, LeavePOut
from joblib import dump, load

# 设置是否使用重采样的开关
use_resampling = True  # 将此设置为False以关闭重采样

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
# features.fillna(features.median(), inplace=True)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(feature_concate)

# ## 分割数据集
# X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.01, random_state=42)

X_train, y_train = features_scaled, labels

# 特征选择 - 使用随机森林
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

# 获取特征重要性
importances = rf.feature_importances_

# 获取最重要的特征的索引（例如，选择前20个特征）
indices = np.argsort(importances)[::-1][:10]

# 使用重要特征
X_train_selected = X_train[:, indices]
# X_test_selected = X_test[:, indices]

# X_train_selected = X_train
# X_test_selected = X_test

n_neighbors = 4

# 检查是否使用重采样策略
if use_resampling:
    # 选择重采样策略
    # sampler = SMOTEENN(smote=SMOTE(k_neighbors=min(n_neighbors, len(y_train) // 2)))
    sampler = SMOTETomek(smote=SMOTE(k_neighbors=min(n_neighbors, len(y_train) // 2)))
    X_train_resampled, y_train_resampled = sampler.fit_resample(X_train_selected, y_train)
else:
    X_train_resampled, y_train_resampled = X_train_selected, y_train

# 训练XGBoost模型
xgb_model = XGBClassifier(
    n_estimators=500,  # 减少树的数量
    learning_rate=0.05,  # 增加学习率
    max_depth=10,  # 增加树的深度
    min_child_weight=1,  # 子节点的最小权重
    gamma=0.2,  # 增加分割损失
    subsample=0.7,  # 减少用于训练每棵树的样本比例
    colsample_bytree=0.7,  # 减少用于训练每棵树的特征比例
    reg_lambda=2,  # 增加L2正则化项
    reg_alpha=0.15  # 保持L1正则化项不变
)

#################
# 设置5折交叉验证
cv = StratifiedKFold(n_splits=10)
# cv = LeavePOut(p=2)
# 初始化存储每一折的评估结果
logloss_train, logloss_test = [], []

for train_idx, test_idx in cv.split(X_train_resampled, y_train_resampled):
    # 分割数据
    X_train_fold, X_test_fold = X_train_resampled[train_idx], X_train_resampled[test_idx]
    y_train_fold, y_test_fold = y_train_resampled[train_idx], y_train_resampled[test_idx]

    # 训练模型
    xgb_model.fit(X_train_fold, y_train_fold, eval_set=[(X_train_fold, y_train_fold), (X_test_fold, y_test_fold)],
                  verbose=False)
    # 获取训练历史记录
    results = xgb_model.evals_result()
    logloss_train.append(results['validation_0']['logloss'])
    logloss_test.append(results['validation_1']['logloss'])

# 计算平均logloss
avg_logloss_train = np.mean(logloss_train, axis=0)
avg_logloss_test = np.mean(logloss_test, axis=0)

# 绘制训练和验证的对数损失曲线
epochs = len(avg_logloss_train)
x_axis = range(0, epochs)
fig, ax = plt.subplots()
ax.plot(x_axis, avg_logloss_train, label='Train')
ax.plot(x_axis, avg_logloss_test, label='Test')
ax.legend()
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss')
plt.show()

# # 保存标准化处理器
dump(scaler, 'scaler.joblib')


#
# # 保存特征选择的索引
# dump(indices, 'indices.joblib')
# dump(xgb_model, 'temp.joblib')


def evaluate_and_plot(model, X_test_, y_test_, title):
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
    # plt.show()


## 模型预评估
test_file_path = 'data_DM_test.xlsx'
df_test_origin = pd.read_excel(test_file_path)
df_test = df_test_origin.drop(["患病时间-5-10-", "LDH1", "MG", "PCT", "PLCR", "AFU"], axis=1)
df_test = df_test.dropna()
test_features = df_test.drop(["患病时间-10-15-"], axis=1)
# test_features = test_features[['OSM', 'CRE', 'BUNCREA', 'eGFR（分组指标）']]
test_labels = df_test["患病时间-10-15-"].replace({1: 0, 2: 1, 3: 1})
test_features_scaled = scaler.transform(test_features)
test_features_scaled = test_features_scaled[:, indices]

# # 5. 评估模型性能
# loaded_model = load('best_model.joblib')
evaluate_and_plot(xgb_model, test_features_scaled, test_labels, 'Model on Test Data')

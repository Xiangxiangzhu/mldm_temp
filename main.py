import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from util import evaluate_and_plot
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

print("#################################### switch defined here! ####################################")
use_resampling = True
selected_model = 'random_forest'  # 'logistic_regression', 'svm', 'knn', 'naive_bayes', 'neural_network', 'adaboost', 'gradient_boosting', 'random_forest', 'lightgbm', 'catboost', 'xgboost'
print("#################################### switch defined here! ####################################")

file_path = 'raw.xlsx'
df = pd.read_excel(file_path, sheet_name=None)

arg_data_DM = df["data_DM"]
arg_data_DM["label"] = 0
arg_data_DKD = df["data_DKD"]
arg_data_DKD["label"] = 1

feature_concatenate = (pd.concat([arg_data_DM, arg_data_DKD], axis=0).reset_index(drop=True)
                       .drop(["AFU", "PCT", "PLCR"], axis=1).dropna().reset_index(drop=True))
train_label = feature_concatenate["label"]
train_feature = feature_concatenate.drop(["label"], axis=1)

print("#################################### data is reformed! ####################################")

# noinspection DuplicatedCode
scaler = StandardScaler()
features_scaled = scaler.fit_transform(train_feature)
X_train, y_train = features_scaled, train_label

# 特征选择 - 使用随机森林
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
# 获取特征重要性
importances = rf.feature_importances_
# 获取最重要的特征的索引（例如，选择前20个特征）
indices = np.argsort(importances)[::-1][:20]
# 使用重要特征
X_train = X_train[:, indices]

n_neighbors = 4
# 检查是否使用重采样策略
if use_resampling:
    # 选择重采样策略
    # sampler = SMOTEENN(smote=SMOTE(k_neighbors=min(n_neighbors, len(y_train) // 2)))
    sampler = SMOTETomek(smote=SMOTE(k_neighbors=min(n_neighbors, len(y_train) // 2)))
    X_train, y_train = sampler.fit_resample(X_train, y_train)
else:
    pass

print("#################################### data is preprocessed! ####################################")

# 训练XGBoost模型
# noinspection DuplicatedCode
models = {
    'logistic_regression': LogisticRegression(max_iter=100, solver='lbfgs', class_weight='balanced'),
    'svm': SVC(kernel='rbf', C=10.0, gamma='auto', probability=True, class_weight='balanced'),
    'knn': KNeighborsClassifier(n_neighbors=5, weights='uniform'),
    'naive_bayes': GaussianNB(var_smoothing=1e-15),
    'neural_network': MLPClassifier(
        hidden_layer_sizes=(30, 30, 30),
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        alpha=0.001,
        max_iter=5000
    ),
    'adaboost': AdaBoostClassifier(n_estimators=100, learning_rate=1.0),
    'gradient_boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, subsample=0.8),
    'random_forest': RandomForestClassifier(
        n_estimators=500,
        max_depth=2,
        min_samples_split=2,
        min_samples_leaf=3,
        max_features="sqrt",
        bootstrap=True,
        class_weight='balanced'
    ),
    'lightgbm': LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=-1, num_leaves=31),
    'catboost': CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6),
    'xgboost': XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, min_child_weight=1, subsample=0.8,
                             colsample_bytree=0.8)
}
#################
# 设置5折交叉验证
cv = StratifiedKFold(n_splits=20)
# cv = LeavePOut(p=2)
# 初始化存储每一折的评估结果
# noinspection DuplicatedCode
logloss_train, logloss_test = [], []

model = models[selected_model]

for train_idx, test_idx in cv.split(X_train, y_train):
    # 分割数据
    X_train_fold, X_test_fold = X_train[train_idx], X_train[test_idx]
    y_train_fold, y_test_fold = y_train[train_idx], y_train[test_idx]

    # 训练模型
    model.fit(X_train_fold, y_train_fold)
    # model.fit(X_train_fold, y_train_fold, eval_set=[(X_train_fold, y_train_fold), (X_test_fold, y_test_fold)],
    #           verbose=False)
    # 获取训练历史记录
    if selected_model == 'xgboost':
        results = model.evals_result()
        logloss_train.append(results['validation_0']['logloss'])
        logloss_test.append(results['validation_1']['logloss'])

# 计算平均logloss（仅适用于XGBoost）
if selected_model == 'xgboost':
    # noinspection DuplicatedCode
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

# 保存标准化处理器
dump(scaler, 'scaler.joblib')
# 保存特征选择的索引
dump(indices, 'indices.joblib')
# 保存模型
dump(model, 'temp.joblib')

print("#################################### train finished! ####################################")

## 模型预评估
test_file_path = 'test_data.xlsx'
test_origin = pd.read_excel(test_file_path)
test_labels = test_origin["label"]
test_features = test_origin.drop(["label"], axis=1)
test_features_scaled = scaler.transform(test_features)
test_features_scaled = test_features_scaled[:, indices]
# # 5. 评估模型性能
# loaded_model = load('best_model.joblib')
evaluate_and_plot(model, test_features_scaled, test_labels, 'Model on Test Data')

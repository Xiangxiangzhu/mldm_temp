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
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import LeaveOneOut

# 设置是否使用重采样的开关
use_resampling = True  # 将此设置为False以关闭重采样

# 1. 数据加载与预处理
file_path = 'data_DM.xlsx'
df = pd.read_excel(file_path)

## 选择特征和重新定义标签为二分类问题
features = df.drop(["患病时间-10-15-", "患病时间-5-10-", "TBA", "PCT", "PLCR", "AFU"], axis=1)
# features = features[['OSM', 'CRE', 'BUNCREA', 'eGFR（分组指标）']]
labels = df["患病时间-5-10-"].replace({1: 0, 2: 1, 3: 1})

## 数据预处理
features.fillna(features.median(), inplace=True)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

## 分割数据集
X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.3, random_state=42)

# 特征选择 - 使用随机森林
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

# 获取特征重要性
importances = rf.feature_importances_

# 获取最重要的特征的索引（例如，选择前20个特征）
indices = np.argsort(importances)[::-1][:20]

# 使用重要特征
X_train_selected = X_train[:, indices]
X_test_selected = X_test[:, indices]

# 2. 模型训练与选择
models = {
    # "RandomForest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(),
    # "LightGBM": LGBMClassifier(),
    # "CatBoost": CatBoostClassifier()
}

if use_resampling:
    param_grids = {
        # "RandomForest": {
        #     "RandomForest__n_estimators": [50, 100, 150],
        #     "RandomForest__max_depth": [3, 5, 10, None],
        #     "RandomForest__min_samples_split": [2, 4, 6],
        #     "RandomForest__min_samples_leaf": [1, 2, 4],
        #     "RandomForest__bootstrap": [True, False],
        #     "RandomForest__class_weight": ["balanced", None]
        # },
        "XGBoost": {
            "XGBoost__n_estimators": [100],  # 较少的树的数量
            "XGBoost__learning_rate": [0.05],  # 更详细的学习率设置
            "XGBoost__max_depth": [3],  # 调整最大深度
            "XGBoost__min_child_weight": [1, 2, 4],  # 默认值
            "XGBoost__gamma": [0, 0.1, 0.2],  # 轻微的分裂损失阈值
            "XGBoost__subsample": [0.8],  # 子样本比例
            "XGBoost__colsample_bytree": [0.8],  # 特征采样比例
            "XGBoost__reg_lambda": [1, 1.5],  # L2正则化
            "XGBoost__reg_alpha": [0, 0.1]  # L1正则化
        },
        # "LightGBM": {
        #     "LightGBM__n_estimators": [50, 100, 150],
        #     "LightGBM__learning_rate": [0.01, 0.1, 0.2],
        #     "LightGBM__max_depth": [3, 4, 5],
        #     "LightGBM__reg_lambda": [1, 2, 3]
        # },
    }
else:
    param_grids = {
        # "RandomForest": {
        #     "n_estimators": [50, 100, 150],
        #     "max_depth": [3, 5, 10, None],
        #     "min_samples_split": [2, 4, 6],
        #     "min_samples_leaf": [1, 2, 4],
        #     "bootstrap": [True, False],
        #     "class_weight": ["balanced", None]
        # },
    }

## 数据处理和模型训练
cv = StratifiedKFold(n_splits=5)
# cv = LeaveOneOut(5)
best_models = {}
best_scores = {}

n_neighbors = 3

for model_name, model in models.items():
    if use_resampling:
        resampling_strategies = {
            # "ADASYN": ADASYN(n_neighbors=n_neighbors),
            # "SMOTE": SMOTE(k_neighbors=n_neighbors),
            "SMOTEENN": SMOTEENN(smote=SMOTE(k_neighbors=min(n_neighbors, len(y_train) // 2))),
            "SMOTETomek": SMOTETomek(smote=SMOTE(k_neighbors=min(n_neighbors, len(y_train) // 2)))
        }
    else:
        resampling_strategies = {"None": None}  # 不使用重采样

    for sampler_name, sampler in resampling_strategies.items():
        if sampler is None:
            estimator = model
        else:
            estimator = Pipeline([(sampler_name, sampler), (model_name, model)])

        grid_search = GridSearchCV(estimator, param_grids[model_name], cv=cv, scoring='roc_auc', verbose=3)
        # grid_search = GridSearchCV(estimator, param_grids[model_name], cv=cv, scoring='f1', verbose=3)
        # grid_search = GridSearchCV(estimator, param_grids[model_name], cv=cv, scoring='f1')

        grid_search.fit(X_train, y_train)

        best_models[f"{model_name}_{sampler_name if sampler else 'NoResampling'}"] = grid_search.best_estimator_
        best_scores[f"{model_name}_{sampler_name if sampler else 'NoResampling'}"] = grid_search.best_score_

# 3. 保存最佳模型
## 输出最佳模型和分数
print("the best model is: ", best_models)
print("the best scores are: ", best_scores)
best_model_name = max(best_scores, key=best_scores.get)
best_model = best_models[best_model_name]
dump(best_model, 'best_model.joblib')


# ## 用于模型融合
# rf_smoteenn = best_models['RandomForest_SMOTEENN']
# rf_smotetomek = best_models['RandomForest_SMOTETomek']
# dump(rf_smoteenn, 'rf_smoteenn.joblib')
# dump(rf_smotetomek, 'rf_smotetomek.joblib')
# ## 创建投票集成模型
# voting_clf = VotingClassifier(estimators=[
#     ('rf_smoteenn', rf_smoteenn),
#     ('rf_smotetomek', rf_smotetomek)
# ], voting='soft')
# ## 训练集成模型（可选，如果模型已经训练过可以跳过）
# voting_clf.fit(X_train, y_train)
# ## 保存集成模型
# dump(voting_clf, 'voting_clf.joblib')
#
# ### 创建堆叠集成模型
# stacking_clf = StackingClassifier(estimators=[
#     ('rf_smoteenn', rf_smoteenn),
#     ('rf_smotetomek', rf_smotetomek)
# ], final_estimator=RandomForestClassifier())
# # 训练堆叠集成模型
# stacking_clf.fit(X_train, y_train)
# # 保存堆叠集成模型
# dump(stacking_clf, 'stacking_clf.joblib')


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


## 模型预评估
for name, model in best_models.items():
    evaluate_and_plot(model, X_test, y_test, name)

# 4. 加载测试数据并预处理
test_file_path = 'data_DM_test.xlsx'
df_test_origin = pd.read_excel(test_file_path)
df_test = df_test_origin.drop(["患病时间-10-15-", "LDH1", "MG", "PCT", "PLCR", "AFU"], axis=1)
df_test = df_test.dropna()
test_features = df_test.drop(["患病时间-5-10-"], axis=1)
# test_features = test_features[['OSM', 'CRE', 'BUNCREA', 'eGFR（分组指标）']]
test_labels = df_test["患病时间-5-10-"].replace({1: 0, 2: 1, 3: 1})
test_features_scaled = scaler.transform(test_features)
test_features_scaled = test_features_scaled[:, indices]

# # 5. 评估模型性能
# loaded_model = load('best_model.joblib')
evaluate_and_plot(best_model, test_features_scaled, test_labels, 'Best Model on Test Data')
# loaded_model = load('rf_smoteenn.joblib')
# evaluate_and_plot(rf_smoteenn, test_features_scaled, test_labels, 'SMOTEENN Model on Test Data')
# loaded_model = load('rf_smotetomek.joblib')
# evaluate_and_plot(rf_smotetomek, test_features_scaled, test_labels, 'SMOTETomek Model on Test Data')
# loaded_model = load('voting_clf.joblib')
# evaluate_and_plot(loaded_model, test_features_scaled, test_labels, 'voting_clf Model on Test Data')

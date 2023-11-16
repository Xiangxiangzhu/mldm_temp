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
import matplotlib.pyplot as plt
import wandb

# 加载数据集
file_path = 'data_DM.xlsx'
df = pd.read_excel(file_path)

# 选择特征和重新定义标签为二分类问题
features = df.drop("患病时间-10-15-", axis=1)
labels = df["患病时间-10-15-"].replace({1: 0, 2: 1, 3: 1})

# 数据预处理
features.fillna(features.median(), inplace=True)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.3, random_state=42)

# 定义模型和参数
models = {
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(),
    "SVM": SVC(probability=True),
    "BP_NeuralNetwork": MLPClassifier()
}

param_grids = {
    "DecisionTree": {"DecisionTree__max_depth": [10, 20, 30, None]},
    "RandomForest": {"RandomForest__n_estimators": [10, 50, 100], "RandomForest__max_depth": [10, 20, 30, None]},
    "SVM": {"SVM__C": [0.1, 1, 10], "SVM__kernel": ["linear", "rbf"]},
    "BP_NeuralNetwork": {"BP_NeuralNetwork__hidden_layer_sizes": [(50,), (100,), (50, 50)],
                         "BP_NeuralNetwork__activation": ["tanh", "relu"]}
}

# 数据处理和模型训练
cv = StratifiedKFold(n_splits=5)
best_models = {}
best_scores = {}

# 使用WandB记录
wandb.init(project="diabetes_prediction")

n_neighbors = 3  # 或者根据您数据集的具体情况选择合适的值

for model_name, model in models.items():
    for sampler_name, sampler in {"ADASYN": ADASYN(n_neighbors=n_neighbors),
                                  "SMOTE": SMOTE(k_neighbors=n_neighbors)}.items():
        pipeline = Pipeline([
            (sampler_name, sampler),
            (model_name, model)
        ])

        grid_search = GridSearchCV(pipeline, param_grids[model_name], cv=cv, scoring='roc_auc')
        grid_search.fit(X_train, y_train)

        best_models[f"{model_name}_{sampler_name}"] = grid_search.best_estimator_
        best_scores[f"{model_name}_{sampler_name}"] = grid_search.best_score_

        wandb.log({f"{model_name}_{sampler_name}_best_score": grid_search.best_score_})

# 输出最佳模型和分数
print("the best model is: ", best_models)
print("the best scores are: ", best_scores)

# 模型评估
for name, model in best_models.items():
    y_pred = model.predict(X_test)
    print(f"Model: {name}")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {name}')
    plt.legend(loc="lower right")
    plt.show()

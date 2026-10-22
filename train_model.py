from pathlib import Path
import joblib
import pandas as pd

df = pd.read_csv("obesity_data.csv")
print(df.head())
print("Dataset shape:", df.shape)




# 2. Check required columns
features = [
    "Age",
    "Gender",
    "Height",
    "Weight",
    "PhysicalActivityLevel"
]

target = "ObesityCategory"

required_columns = features + [target]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in dataset: {missing_columns}"
    )

if df[required_columns].isnull().any().any():
    raise ValueError(
        "Missing values found. Clean the dataset before training."
    )

# 3. Separate features and target
X = df[features].copy()
y = df[target].copy()

print("\nTarget distribution:")
print(y.value_counts())

# 4. Split dataset
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. Preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

numeric_features = [
    "Age",
    "Height",
    "Weight",
    "PhysicalActivityLevel"
]

categorical_features = ["Gender"]

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), numeric_features),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)

# 6. Define models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=1
    ),

    "Extra Trees": ExtraTreesClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=1
    ),

    "SVM": SVC(
        class_weight="balanced",
        probability=True,
        random_state=42
    )
}

# 7. Compare models using cross-validation
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = {}
best_score = -1
best_name = None
best_model = None

print("\nComparing models...")

for name, classifier in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="f1_weighted",
        n_jobs=1
    )

    mean_score = scores.mean()
    results[name] = mean_score

    print(f"{name}: Mean weighted F1 = {mean_score:.4f}")

    if mean_score > best_score:
        best_score = mean_score
        best_name = name
        best_model = pipeline

# 8. Train selected model
best_model.fit(X_train, y_train)

print(f"\nBest model: {best_name}")
print(f"Cross-validation weighted F1: {best_score:.4f}")

# 9. Evaluate on test dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

y_pred = best_model.predict(X_test)

print("\nTest Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Test Weighted F1:", round(f1_score(
    y_test, y_pred, average="weighted"
), 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 10. Save complete pipeline
MODEL_PATH = "obesity_model.pkl"
joblib.dump(best_model, MODEL_PATH)

print(f"\nModel saved successfully at: {MODEL_PATH}")



# Import standard libraries
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import model selection
from sklearn.model_selection import train_test_split

# Import preprocessing and pipelines
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Import classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Import metrics
from sklearn.metrics import accuracy_score

# Import encoder
from gpc_encoder import gpc_encoder
from sklearn.preprocessing import LabelEncoder
from category_encoders import TargetEncoder, OneHotEncoder

# CONFIG
TARGET_COL = 'deposit' # Target feature's name
CSV_FILE_NAME = 'bank.csv' # CSV file's name
DATA_LEN_LIMIT = 10000 # Limits the number of rows in the data
RANDOM_SEED = 42
MODELS = {"LogReg": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))]),
          "DecisionTree": DecisionTreeClassifier(random_state=42),
          "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
          "KNN": Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=15))]),
          "SVM(RBF)": Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf"))]),
          "MLP": Pipeline([("scaler", StandardScaler()), ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42))])}
ENCODERS= ["GPCE", "Label", "OneHot", "Target"]
D = 16 # Targeted dimension
R = 1.0 # Distance
TEST_SIZE = 0.2 # Test data size

# Load data
df = pd.read_csv(f"./datasets/{CSV_FILE_NAME}")

if DATA_LEN_LIMIT:
    df = df[:DATA_LEN_LIMIT]

tar_dtype = df[TARGET_COL].dtype
if tar_dtype == 'object' or tar_dtype == 'category':
    df[TARGET_COL] = LabelEncoder().fit_transform(df[TARGET_COL])

cat_cols = df.drop(TARGET_COL, axis=1).select_dtypes(include=['object', 'category']).columns.tolist()

df = df.dropna().copy()

# Split X and y
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL].to_numpy()

# Split train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y if len(np.unique(y)) > 1 else None)

# Encode categorical features
def get_encoded_data(method, X_train, X_test, y_train, cat_cols):
    if method == "GPCE":
        X_train_enc, X_test_enc = gpc_encoder(X_train=X_train, X_test=X_test, cat_cols=cat_cols, d=D, r=R, seed=RANDOM_SEED)
        return X_train_enc, X_test_enc

    elif method == "Label":
        X_train_enc, X_test_enc = X_train.copy(), X_test.copy()
        for col in cat_cols:
            le = LabelEncoder()
            full_data = pd.concat([X_train_enc[col], X_test_enc[col]]).astype(str)
            le.fit(full_data)
            X_train_enc[col] = le.transform(X_train_enc[col].astype(str))
            X_test_enc[col] = le.transform(X_test_enc[col].astype(str))
        return X_train_enc, X_test_enc

    elif method == "OneHot":
        ohe = OneHotEncoder(cols=cat_cols, use_cat_names=True, handle_unknown='value')
        X_train_enc = ohe.fit_transform(X_train, y_train)
        X_test_enc = ohe.transform(X_test)
        return X_train_enc, X_test_enc

    elif method == "Target":
        te = TargetEncoder(cols=cat_cols)
        X_train_enc = te.fit_transform(X_train, y_train)
        X_test_enc = te.transform(X_test)
        return X_train_enc, X_test_enc

    return X_train, X_test

results = {}
for encoder in tqdm(
    ENCODERS,
    desc="Encoding method",
    position=0
):
    results[encoder] = {}
    X_train_enc, X_test_enc = get_encoded_data(method=encoder, X_train=X_train, X_test=X_test, y_train=y_train, cat_cols=cat_cols)

    for name, model in tqdm(
        MODELS.items(),
        desc=f"Models ({encoder})",
        position=1,
        leave=False
    ):
        # Train the model
        model.fit(X_train_enc, y_train)
        # Evaluate the model
        pred = model.predict(X_test_enc)
        accuracy = accuracy_score(y_test, pred)
        # Save the result
        results[encoder][name] = float(accuracy)

results_df = pd.DataFrame(results).T
print("========================== Accuracy Result =========================")
print(results_df)

# Visualize
models = results_df.columns.tolist()
encoders = results_df.index.tolist()

n_models = len(models)
n_encoders = len(encoders)

x = np.arange(n_models)
bar_width = 0.8 / n_encoders

gray_levels = np.linspace(0.85, 0.25, len(encoders))

min_y = max(results_df.min().min() - 0.1, 0)
max_y = min(results_df.max().max() + 0.1, 1)

plt.figure(figsize=(12, 5))

for i, (enc, gray) in enumerate(zip(encoders, gray_levels)):
    offsets = x - 0.4 + (i + 0.5) * bar_width
    plt.bar(
        offsets,
        results_df.loc[enc].values,
        width=bar_width,
        color=str(gray_levels[i]),
        edgecolor="black",
        label=enc
    )

plt.xticks(x, models, rotation=0)
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title(f"Comparison of Model Accuracy Across Encoders ({CSV_FILE_NAME})")
plt.ylim(min_y, max_y)
plt.legend(title="Encoder", ncol=min(n_encoders, 4))
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

output_dir = "results"
file_name = f"{CSV_FILE_NAME.replace('.csv', '')}.png"
full_path = os.path.join(output_dir, file_name)
os.makedirs(output_dir, exist_ok=True)
plt.savefig(full_path, dpi=500)

plt.show()
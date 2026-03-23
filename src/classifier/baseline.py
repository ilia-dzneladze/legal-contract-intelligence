import os
import sys
import pickle

# Allow running as a script from any CWD (Colab, repo root, etc.)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_script_dir, "../.."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.classifier.data_loader import load_data
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict, StratifiedKFold

contracts = load_data("CUADv1.json")
test_data = load_data("test.json")

train_X = contracts["text"]
train_y = contracts["contract_type"]
test_X = test_data["text"]
test_y = test_data["contract_type"]

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=20000)),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

pipeline.fit(train_X, train_y)

# cross-validation on training data
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred_cv = cross_val_predict(pipeline, train_X, train_y, cv=cv) # type: ignore
print("Cross-validation results:")
print(classification_report(train_y, y_pred_cv))

# test set evaluation
y_pred_test = pipeline.predict(test_X)
print("Test set results:")
print(classification_report(test_y, y_pred_test))

# save after everything checks out
_models_dir = os.path.join(_repo_root, "models")
os.makedirs(_models_dir, exist_ok=True)
with open(os.path.join(_models_dir, "baseline_tfidf_logreg.pkl"), "wb") as f:
    pickle.dump(pipeline, f)
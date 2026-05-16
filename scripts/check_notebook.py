"""Full end-to-end check of every notebook cell logic."""
import warnings, sys
warnings.filterwarnings("error", category=FutureWarning)
warnings.filterwarnings("error", category=DeprecationWarning)

import pandas as pd
import numpy as np
from pathlib import Path
from math import pi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay,
    classification_report,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

SEED = 42
features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# ── load ──────────────────────────────────────────────────────────────────────
here = Path(__file__).parent.parent
csv = here / "document" / "Atividade_Cap10_produtos_agricolas.csv"
df = pd.read_csv(csv)
assert df.isnull().sum().sum() == 0, "nulos encontrados"
assert df.duplicated().sum() == 0, "duplicatas encontradas"
label_counts = df["label"].value_counts()
assert label_counts.nunique() == 1, "dataset nao balanceado"
print(f"[OK] load  shape={df.shape}  culturas={df['label'].nunique()}  amostras/cultura={label_counts.iloc[0]}")

# ── perfil ideal ──────────────────────────────────────────────────────────────
ideal = df[features].median().rename("ideal")
profiles = df.groupby("label")[features].median()
CHOSEN = ["rice", "coffee", "apple"]
comparison = profiles.loc[CHOSEN].T.copy()
comparison["ideal"] = ideal
desvio = (
    comparison[CHOSEN]
    .sub(comparison["ideal"], axis=0)
    .div(comparison["ideal"], axis=0)
    .mul(100)
    .round(1)
)
assert not desvio.isnull().any().any(), "desvio tem NaN"
print(f"[OK] perfil ideal  N={ideal['N']}  P={ideal['P']}  K={ideal['K']}  hum={ideal['humidity']:.1f}  rain={ideal['rainfall']:.1f}")
for crop in CHOSEN:
    row = desvio[crop]
    print(f"     {crop:8s}  N={row['N']:+.1f}%  P={row['P']:+.1f}%  K={row['K']:+.1f}%  temp={row['temperature']:+.1f}%  hum={row['humidity']:+.1f}%  ph={row['ph']:+.1f}%  rain={row['rainfall']:+.1f}%")

# ── preprocessing ─────────────────────────────────────────────────────────────
le = LabelEncoder()
X = df[features].values
y = le.fit_transform(df["label"].values)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# check no leakage: scaler fit only on train
assert abs(X_train_sc.mean()) < 1e-10, "scaler vazamento: media nao zero"
assert abs(X_train_sc.std() - 1.0) < 1e-6, "scaler vazamento: std nao um"
# test set should NOT be zero-mean (confirming no fit on test)
# (it can be close but not exact)
print(f"[OK] preprocessing  treino={X_train.shape[0]}  teste={X_test.shape[0]}  scaler sem leakage")

# ── modelos ───────────────────────────────────────────────────────────────────
results = {}

def run(name, clf, Xtr, Xte):
    clf.fit(Xtr, y_train)
    yp = clf.predict(Xte)
    acc   = accuracy_score(y_test, yp)
    f1    = f1_score(y_test, yp, average="macro")
    cvacc = cross_val_score(clf, Xtr, y_train, cv=cv, scoring="accuracy").mean()
    results[name] = {"Acuracia Teste": acc, "F1-Macro": f1, "Acuracia CV": cvacc}
    print(f"[OK] {name:25s}  acc={acc:.4f}  f1={f1:.4f}  cv={cvacc:.4f}")
    return clf, yp

lr_clf,  lr_pred  = run("Regressao Logistica",  LogisticRegression(max_iter=1000, random_state=SEED),                                             X_train_sc, X_test_sc)
dt_clf,  dt_pred  = run("Arvore de Decisao",    DecisionTreeClassifier(random_state=SEED),                                                        X_train,    X_test)
rf_clf,  rf_pred  = run("Random Forest",        RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),                           X_train,    X_test)
knn_clf, knn_pred = run("KNN k=5",              KNeighborsClassifier(n_neighbors=5, metric="euclidean"),                                          X_train_sc, X_test_sc)
gb_clf,  gb_pred  = run("Gradient Boosting",    GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=4, random_state=SEED),  X_train,    X_test)

# check: RF must be best
best = max(results, key=lambda k: results[k]["Acuracia Teste"])
assert best == "Random Forest", f"melhor modelo inesperado: {best}"
print(f"[OK] melhor modelo = {best}")

# ── graficos ──────────────────────────────────────────────────────────────────

# g1
fig, ax = plt.subplots(figsize=(12, 5))
ax.barh(label_counts.index, label_counts.values, color=plt.get_cmap("tab20").colors[:len(label_counts)])
plt.close(); print("[OK] g1 distribuicao")

# g2
corr = df[features].corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
            mask=np.triu(np.ones_like(corr, dtype=bool)), ax=ax)
plt.close(); print("[OK] g2 heatmap")

# g3 - boxplot com hue= (seaborn >= 0.13)
fig, axes = plt.subplots(2, 4, figsize=(20, 9))
for i, feat in enumerate(features):
    order = df.groupby("label")[feat].median().sort_values().index
    sns.boxplot(data=df, y="label", x=feat, order=order,
                hue="label", palette="tab20", legend=False, ax=axes.flatten()[i])
axes.flatten()[-1].set_visible(False)
plt.close(); print("[OK] g3 boxplot")

# g4 - histogramas + KDE
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for i, feat in enumerate(features):
    ax = axes.flatten()[i]
    ax.hist(df[feat], bins=40, density=True, color="steelblue", alpha=0.6)
    df[feat].plot.kde(ax=ax)
axes.flatten()[-1].set_visible(False)
plt.close(); print("[OK] g4 histogramas")

# g5 - scatter
fig, ax = plt.subplots(figsize=(12, 7))
cmap = plt.get_cmap("tab20")
for i, crop in enumerate(sorted(df["label"].unique())):
    m = df["label"] == crop
    ax.scatter(df.loc[m, "temperature"], df.loc[m, "rainfall"],
               alpha=0.55, s=25, color=cmap(i / df["label"].nunique()))
plt.close(); print("[OK] g5 scatter")

# g6 - radar
feat_min = df[features].min(); feat_max = df[features].max()
def normalize(s): return ((s - feat_min) / (feat_max - feat_min)).values
N_cats = len(features)
angles = [n / float(N_cats) * 2 * pi for n in range(N_cats)] + [0]
fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
for s in [ideal] + [profiles.loc[c] for c in CHOSEN]:
    v = normalize(s); vals = np.concatenate([v, [v[0]]])
    ax.plot(angles, vals); ax.fill(angles, vals, alpha=0.08)
plt.close(); print("[OK] g6 radar")

# g7 - compare bar
res_df = pd.DataFrame(results).T
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(res_df)); w = 0.28
ax.bar(x - w, res_df["Acuracia Teste"], w)
ax.bar(x,     res_df["F1-Macro"],       w)
ax.bar(x + w, res_df["Acuracia CV"],    w)
plt.close(); print("[OK] g7 compare bar")

# g8 - matriz de confusao
pred_map = {"Regressao Logistica": lr_pred, "Arvore de Decisao": dt_pred,
            "Random Forest": rf_pred, "KNN k=5": knn_pred, "Gradient Boosting": gb_pred}
cm = confusion_matrix(y_test, pred_map[best])
fig, ax = plt.subplots(figsize=(14, 12))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_).plot(
    ax=ax, xticks_rotation=45, colorbar=False, cmap="Blues")
plt.close(); print(f"[OK] g8 confmatrix  melhor={best}")

# g9 - feat importance
feat_imp = pd.Series(rf_clf.feature_importances_, index=features).sort_values()
fig, ax = plt.subplots(figsize=(8, 4))
feat_imp.plot.barh(ax=ax)
plt.close()
print(f"[OK] g9 feat_importance  top={feat_imp.index[-1]}")

# pandas .map (nao applymap)
rdf = pd.DataFrame(results).T
rdf.sort_values("Acuracia Teste", ascending=False).map(lambda x: f"{x:.4f}")
print("[OK] pandas .map sem applymap")

print()
print("=" * 50)
print("TODOS OS CHECKS PASSARAM SEM ERROS OU WARNINGS")
print("=" * 50)

# !pip install rdkit pandas matplotlib scikit-learn 

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Draw
from rdkit.Chem import rdMolDescriptors
from IPython.display import display

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FEATURES = [
    "mol_weight",
    "logp",
    "h_donors",
    "h_acceptors",
    "tpsa",
    "rotatable_bonds",
    "heavy_atoms",
    "rings",
    "aromatic_rings",
]

RF_PARAMS = {"n_estimators": 300, "random_state": 42}
RANDOM_STATE = 42


def make_forest_model():
    return RandomForestRegressor(**RF_PARAMS)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

!wget -O /tmp/delaney-processed.csv "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv"

df = pd.read_csv("/tmp/delaney-processed.csv")

df = df.rename(columns={
    "measured log solubility in mols per litre": "log_solubility"
})


# ---------------------------------------------------------------------------
# Descriptor generation
# ---------------------------------------------------------------------------

def get_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    return {
        "mol_weight": Descriptors.MolWt(mol),
        "logp": Descriptors.MolLogP(mol),
        "h_donors": Descriptors.NumHDonors(mol),
        "h_acceptors": Descriptors.NumHAcceptors(mol),
        "tpsa": Descriptors.TPSA(mol),
        "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
        "heavy_atoms": Descriptors.HeavyAtomCount(mol),
        "rings": Descriptors.RingCount(mol),
        "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
    }


def build_descriptor_table(names, smiles_list):
    """Builds a dataframe of {name, smiles, **descriptors}, skipping
    unparseable SMILES cleanly so names/descriptors never misalign."""
    rows = []
    for name, smiles in zip(names, smiles_list):
        descriptors = get_descriptors(smiles)
        if descriptors is None:
            print("skipping invalid smiles:", name, smiles)
            continue
        rows.append({"name": name, "smiles": smiles, **descriptors})
    return pd.DataFrame(rows)


descriptor_table = build_descriptor_table(df["Compound ID"], df["smiles"])

# Merge on name (not positional concat) so a skipped/invalid SMILES can never
# silently shift every row below it out of alignment.
df = df.merge(
    descriptor_table.rename(columns={"name": "Compound ID"}),
    on="Compound ID",
    how="inner",
    suffixes=('_original', '') # Keep the descriptor_table's smiles as 'smiles', original df's smiles as 'smiles_original'
)
# Drop the 'smiles_original' column to avoid ambiguity, keeping the 'smiles' from descriptor_table
df = df.drop(columns=['smiles_original'])

if len(df) < len(descriptor_table):
    print(f"warning: {len(descriptor_table) - len(df)} compounds dropped "
          f"during merge (duplicate or mismatched IDs)")

x = df[FEATURES]
y = df["log_solubility"]


# ---------------------------------------------------------------------------
# Cross-validation first — this is what actually decides the best model,
# since a single train/test split's MAE is noisy on a ~1100-row dataset.
# ---------------------------------------------------------------------------

linear_cv_mae = -cross_val_score(
    LinearRegression(), x, y, cv=5, scoring="neg_mean_absolute_error"
)
linear_cv_r2 = cross_val_score(LinearRegression(), x, y, cv=5, scoring="r2")

forest_cv_mae = -cross_val_score(
    make_forest_model(), x, y, cv=5, scoring="neg_mean_absolute_error"
)
forest_cv_r2 = cross_val_score(make_forest_model(), x, y, cv=5, scoring="r2")

print("cross validation (5-fold)")
print()
print("linear regression")
print("mae scores:", linear_cv_mae)
print("average mae:", linear_cv_mae.mean())
print("average r2:", linear_cv_r2.mean())
print()
print("random forest")
print("mae scores:", forest_cv_mae)
print("average mae:", forest_cv_mae.mean())
print("average r2:", forest_cv_r2.mean())
print()

use_forest = forest_cv_mae.mean() < linear_cv_mae.mean()
best_model_name = "random forest" if use_forest else "linear regression"
print("best model (by CV mae):", best_model_name)
print()


# ---------------------------------------------------------------------------
# Fit the chosen model on a single train/test split for diagnostics
# (residuals, worst predictions, scatter plot).
# ---------------------------------------------------------------------------

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=RANDOM_STATE
)

best_model = make_forest_model() if use_forest else LinearRegression()
best_model.fit(x_train, y_train)
y_pred_best = best_model.predict(x_test)

best_mae = mean_absolute_error(y_test, y_pred_best)
best_r2 = r2_score(y_test, y_pred_best)

print(f"{best_model_name} on held-out 20% split")
print("mae:", best_mae)
print("r2:", best_r2)


# ---------------------------------------------------------------------------
# Diagnostics: predicted vs actual, worst errors, residuals
# ---------------------------------------------------------------------------

plt.scatter(y_test, y_pred_best)

min_value = min(y_test.min(), y_pred_best.min())
max_value = max(y_test.max(), y_pred_best.max())
plt.plot([min_value, max_value], [min_value, max_value])

plt.xlabel("actual log solubility")
plt.ylabel("predicted log solubility")
plt.title(best_model_name + " predicted vs actual solubility")
plt.show()

results = x_test.copy()
results["name"] = df.loc[x_test.index, "Compound ID"].values
results["smiles"] = df.loc[x_test.index, "smiles"].values # This will now correctly access the single 'smiles' column
results["actual_log_solubility"] = y_test.values
results["predicted_log_solubility"] = y_pred_best
results["error"] = results["actual_log_solubility"] - results["predicted_log_solubility"]
results["absolute_error"] = results["error"].abs()

biggest_errors = results.sort_values("absolute_error", ascending=False)

print("\nbiggest prediction errors for the best model:")
print(
    biggest_errors[
        [
            "name", "smiles", "actual_log_solubility",
            "predicted_log_solubility", "error", "absolute_error",
            "mol_weight", "logp", "h_donors", "h_acceptors",
        ]
    ].head(10)
)

worst = biggest_errors.head(8)
worst_mols = [Chem.MolFromSmiles(s) for s in worst["smiles"]]
labels = [
    row["name"]
    + "\nactual: " + str(round(row["actual_log_solubility"], 2))
    + "\npred: " + str(round(row["predicted_log_solubility"], 2))
    + "\nerror: " + str(round(row["error"], 2))
    for _, row in worst.iterrows()
]
img = Draw.MolsToGridImage(
    worst_mols, legends=labels, molsPerRow=4, subImgSize=(300, 250)
)
display(img)

plt.scatter(results["predicted_log_solubility"], results["error"])
plt.axhline(0)
plt.xlabel("predicted log solubility")
plt.ylabel("error")
plt.title("residual plot for " + best_model_name)
plt.show()


# ---------------------------------------------------------------------------
# Predict on new molecules
# ---------------------------------------------------------------------------

def predict_many(molecules):
    """molecules: list of {"name": ..., "smiles": ...} dicts."""
    names = [m["name"] for m in molecules]
    smiles_list = [m["smiles"] for m in molecules]
    new_df = build_descriptor_table(names, smiles_list)

    if new_df.empty:
        print("no valid molecules to predict")
        return new_df

    x_new = new_df[FEATURES]

    # Sanity check: flag molecules whose descriptors fall outside the
    # training distribution, since predictions there are low-confidence
    # extrapolations rather than interpolations.
    train_desc = x_train.describe()
    out_of_range = []
    for name, row in zip(new_df["name"], x_new.itertuples(index=False)):
        for feat, val in zip(FEATURES, row):
            lo, hi = train_desc.loc["min", feat], train_desc.loc["max", feat]
            if val < lo or val > hi:
                out_of_range.append((name, feat, val, lo, hi))
    if out_of_range:
        print("warning: some molecules fall outside the training feature "
              "range (predictions there are extrapolations):")
        for name, feat, val, lo, hi in out_of_range:
            print(f"  {name}: {feat}={val:.2f} (train range {lo:.2f}-{hi:.2f})")
        print()

    new_df["predicted_log_solubility"] = best_model.predict(x_new)
    new_df["predicted_mol_per_L"] = 10 ** new_df["predicted_log_solubility"]
    new_df["predicted_mg_per_L"] = (
        new_df["predicted_mol_per_L"] * new_df["mol_weight"] * 1000
    )

    return new_df.sort_values("predicted_log_solubility", ascending=False)


new_molecules_list = [
    {"name": "ethanol", "smiles": "CCO"},
    {"name": "benzene", "smiles": "c1ccccc1"},
    {"name": "phenol", "smiles": "c1ccc(cc1)O"},
    {"name": "toluene", "smiles": "Cc1ccccc1"},
    {"name": "aniline", "smiles": "Nc1ccccc1"},
    {"name": "aspirin", "smiles": "CC(=O)Oc1ccccc1C(=O)O"},
]

new_predictions = predict_many(new_molecules_list)

print("\npredicted solubilities for new molecules:")
print(
    new_predictions[
        [
            "name", "smiles", "predicted_log_solubility",
            "predicted_mg_per_L", "mol_weight", "logp",
            "h_donors", "h_acceptors", "tpsa",
        ]
    ]
)

plt.figure(figsize=(10, 6))
plt.bar(new_predictions["name"], new_predictions["predicted_log_solubility"])
plt.xlabel("molecule")
plt.ylabel("predicted log solubility")
plt.title("predicted solubility of new molecules")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

new_predictions.to_csv("new_molecule_predictions.csv", index=False)
print("\nnew molecule predictions saved to new_molecule_predictions.csv")

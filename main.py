from rdkit import Chem #kinda broken for now :(
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

!wget -O /tmp/delaney-processed.csv "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv"

df = pd.read_csv("/tmp/delaney-processed.csv")

df = df.rename(columns={
    "measured log solubility in mols per litre": "log_solubility"
})

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
        "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol)
    }

descriptor_rows = []

for smiles in df["smiles"]:
    descriptor_rows.append(get_descriptors(smiles))

descriptor_df = pd.DataFrame(descriptor_rows)

df = pd.concat([df, descriptor_df], axis=1)

features = [
    "mol_weight",
    "logp",
    "h_donors",
    "h_acceptors",
    "tpsa",
    "rotatable_bonds",
    "heavy_atoms",
    "rings",
    "aromatic_rings"
]

x = df[features]
y = df["log_solubility"]

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

linear_model = LinearRegression()
linear_model.fit(x_train, y_train)

linear_pred = linear_model.predict(x_test)

linear_mae = mean_absolute_error(y_test, linear_pred)
linear_r2 = r2_score(y_test, linear_pred)

print("linear regression")
print("mae:", linear_mae)
print("r2:", linear_r2)
print()

forest_model = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

forest_model.fit(x_train, y_train)

forest_pred = forest_model.predict(x_test)

forest_mae = mean_absolute_error(y_test, forest_pred)
forest_r2 = r2_score(y_test, forest_pred)

print("random forest")
print("mae:", forest_mae)
print("r2:", forest_r2)
print()

if forest_mae < linear_mae:
    best_model_name = "random forest"
    best_model = forest_model
    y_pred_best = forest_pred
    best_mae = forest_mae
    best_r2 = forest_r2
else:
    best_model_name = "linear regression"
    best_model = linear_model
    y_pred_best = linear_pred
    best_mae = linear_mae
    best_r2 = linear_r2

print("best model:", best_model_name)
print("best mae:", best_mae)
print("best r2:", best_r2)

def predict_solubility(name, smiles):
    descriptor_dict = get_descriptors(smiles)

    if descriptor_dict is None:
        print("invalid smiles:", smiles)
        return None

    molecule_df = pd.DataFrame([{
        "name": name,
        "smiles": smiles,
        **descriptor_dict
    }])

    x_new = molecule_df[features]

    molecule_df["predicted_log_solubility"] = best_model.predict(x_new)

    return molecule_df

def predict_many(molecules):
    rows = []

    for molecule in molecules:
        name = molecule["name"]
        smiles = molecule["smiles"]

        descriptor_dict = get_descriptors(smiles)

        if descriptor_dict is None:
            print("skipping invalid smiles:", name, smiles)
            continue

        rows.append({
            "name": name,
            "smiles": smiles,
            **descriptor_dict
        })

    new_df = pd.DataFrame(rows)

    x_new = new_df[features]

    new_df["predicted_log_solubility"] = best_model.predict(x_new)
    new_df["predicted_mol_per_L"] = 10 ** new_df["predicted_log_solubility"]
    new_df["predicted_mg_per_L"] = new_df["predicted_mol_per_L"] * new_df["mol_weight"] * 1000

    new_df = new_df.sort_values(
        "predicted_log_solubility",
        ascending=False
    )

    return new_df


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
results["smiles"] = df.loc[x_test.index, "smiles"].values
results["actual_log_solubility"] = y_test.values
results["predicted_log_solubility"] = y_pred_best
results["error"] = results["actual_log_solubility"] - results["predicted_log_solubility"]
results["absolute_error"] = results["error"].abs()
biggest_errors = results.sort_values("absolute_error", ascending=False)

print("biggest prediction errors for the best model:")
print(
    biggest_errors[
        [
            "name",
            "smiles",
            "actual_log_solubility",
            "predicted_log_solubility",
            "error",
            "absolute_error",
            "mol_weight",
            "logp",
            "h_donors",
            "h_acceptors"
        ]
    ].head(10)
)

worst = biggest_errors.head(8)
worst_mols = []
for smiles in worst["smiles"]:
    mol = Chem.MolFromSmiles(smiles)
    worst_mols.append(mol)
labels = []
for index, row in worst.iterrows():
    label = (
        row["name"]
        + "\nactual: " + str(round(row["actual_log_solubility"], 2))
        + "\npred: " + str(round(row["predicted_log_solubility"], 2))
        + "\nerror: " + str(round(row["error"], 2))
    )
    labels.append(label)
img = Draw.MolsToGridImage(
    worst_mols,
    legends=labels,
    molsPerRow=4,
    subImgSize=(300, 250)
)
display(img)

plt.scatter(results["predicted_log_solubility"], results["error"])
plt.axhline(0)
plt.xlabel("predicted log solubility")
plt.ylabel("error")
plt.title("residual plot for " + best_model_name)
plt.show()

linear_cv_mae = -cross_val_score(
    LinearRegression(),
    x,
    y,
    cv=5,
    scoring="neg_mean_absolute_error"
)

linear_cv_r2 = cross_val_score(
    LinearRegression(),
    x,
    y,
    cv=5,
    scoring="r2"
)

forest_cv_mae = -cross_val_score(
    RandomForestRegressor(n_estimators=300, random_state=42),
    x,
    y,
    cv=5,
    scoring="neg_mean_absolute_error"
)

forest_cv_r2 = cross_val_score(
    RandomForestRegressor(n_estimators=300, random_state=42),
    x,
    y,
    cv=5,
    scoring="r2"
)

print("\ncross validation")
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

# Predict solubility for new molecules using the old method
new_molecules = pd.DataFrame([
    {"name": "ethanol", "smiles": "CCO"},
    {"name": "benzene", "smiles": "c1ccccc1"},
    {"name": "phenol", "smiles": "c1ccc(cc1)O"},
    {"name": "toluene", "smiles": "Cc1ccccc1"},
    {"name": "aniline", "smiles": "Nc1ccccc1"},
    {"name": "aspirin", "smiles": "CC(=O)Oc1ccccc1C(=O)O"}
])

new_descriptor_rows = []

for smiles in new_molecules["smiles"]:
    descriptor = get_descriptors(smiles)
    if descriptor is not None:
        new_descriptor_rows.append(descriptor)

new_descriptor_df = pd.DataFrame(new_descriptor_rows)

new_molecules = pd.concat([new_molecules, new_descriptor_df], axis=1)

new_x = new_molecules[features]

new_molecules["predicted_log_solubility"] = best_model.predict(new_x)

print("\nPredicted solubilities for new molecules (old method):")
print(new_molecules[["name", "smiles", "predicted_log_solubility"]])

new_molecules = new_molecules.sort_values(
    "predicted_log_solubility",
    ascending=False
)

print("\nnew molecule predictions (old method, sorted by predicted solubility):")
print(
    new_molecules[
        [
            "name",
            "smiles",
            "predicted_log_solubility",
            "mol_weight",
            "logp",
            "h_donors",
            "h_acceptors",
            "tpsa"
        ]
    ]
)

plt.figure(figsize=(10, 6))
plt.bar(
    new_molecules["name"],
    new_molecules["predicted_log_solubility"]
)

plt.xlabel("molecule")
plt.ylabel("predicted log solubility")
plt.title("predicted solubility of new molecules (old method)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

new_molecules.to_csv("new_molecule_predictions.csv", index=False)
print("\nNew molecule predictions (old method) saved to new_molecule_predictions.csv")

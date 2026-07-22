# Log

(PS: July 25-July 22 is an approximation tracker, I forgot to keep a log! But I remembered the explicit details)

## June 25, 2026

While drinking milk with ice cream, I noticed that the ice cream did not simply "dissolve" as I intuitively expected. From prior chemistry knowledge, I realized that ice cream is a complex mixture containing both polar and nonpolar components and behaves as an emulsion rather than a single dissolved substance. This observation led me to wonder why some substances readily dissolve in water while others do not, and whether molecular structure alone could be used to predict aqueous solubility. Over the following days, I read about computational solubility prediction and began planning a machine-learning project around this question.

## July 1, 2026

Began development in Google Colab.

After exploring available datasets, I selected the Delaney (ESOL) dataset because it provides experimentally measured aqueous solubilities together with molecular structures represented as SMILES strings.

I chose RDKit to generate molecular descriptors and used AI as a programming assistant to accelerate environment setup, imports, and exploration of possible descriptor sets. The scientific design decisions including descriptor selection, model comparison, and evaluation strategy were made during the project's development.

## July 3, 2026

Implemented the descriptor-generation pipeline using RDKit and verified that valid molecular descriptors could be extracted from the dataset. Added handling for invalid SMILES entries and prepared the feature table for machine-learning experiments.

## July 7, 2026

Completed the first working version of the molecular solubility prediction pipeline.

Implemented descriptor extraction using RDKit, trained and evaluated both Linear Regression and Random Forest models, and compared their performance using five-fold cross-validation. Added visualizations, including predicted-versus-observed scatter plots and residual plots, to better understand model behavior.

To investigate the model beyond overall accuracy, I implemented analysis of the largest prediction errors and added an applicability check that warns when new molecules fall outside the descriptor ranges represented in the training data.

With the core pipeline complete, I began identifying questions for future investigation, including descriptor importance, systematic prediction errors across different molecular classes, and model reliability for molecules outside the training distribution.

## July 8, 2026

Completed the first full version of the project after several days of implementation and testing.

The project was able to generate molecular descriptors from SMILES structures, train and compare multiple machine-learning models, evaluate predictive performance through cross-validation, visualize prediction errors, and make solubility predictions for new molecules. After verifying that the pipeline functioned as intended, I made minor code cleanup and usability improvements before considering the first version complete.

Following completion, I shifted my focus to other commitments. Future revisions are planned to investigate descriptor importance, systematic prediction errors across molecular classes, and prediction reliability outside the training distribution.

## July 9-26 

Visiting my village, no progress other than minor tweaks there and here.
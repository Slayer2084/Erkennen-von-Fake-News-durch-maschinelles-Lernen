import pickle
import re
from functools import partial

import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split

from CombineDatasets import get_combined_features_dataset
from PreprocessPipe import get_feature_union

MODELNAME = "PAC"
NUM_TRIALS = 50

df = get_combined_features_dataset()
feature_union = get_feature_union(df)


def optimize(trial, X, y):
    if MODELNAME == "PAC":
        pas_agr_tol = trial.suggest_categorical("pas_agr_tol", ["float", None])
        if pas_agr_tol == "float":
            pas_agr_tol = trial.suggest_float("pas_agr_tol_float", 0.00001, 0.01)

        model = PassiveAggressiveClassifier(loss=trial.suggest_categorical("pas_agr_loss", ["hinge", "squared_hinge"]),
                                            validation_fraction=trial.suggest_float("pas_agr_validation_fraction", 0.1,
                                                                                    1),
                                            C=trial.suggest_float("pas_agr_C", 0.1, 1),
                                            max_iter=trial.suggest_int("pas_agr_max_iter", 500, 10000),
                                            tol=pas_agr_tol,
                                            n_jobs=-1)
    elif MODELNAME == "RFC":
        RFC_min_samples_split_int_or_float = trial.suggest_categorical("RFC_min_samples_split_int_or_float",
                                                                       ["int", "float"])
        if RFC_min_samples_split_int_or_float == "int":
            RFC_min_samples_split = trial.suggest_int("RFC_min_samples_split", 1, 5)
        else:
            RFC_min_samples_split = trial.suggest_float("RFC_min_samples_split", 2, 5)

        RFC_min_samples_leaf_int_or_float = trial.suggest_categorical("RFC_min_samples_leaf_int_or_float",
                                                                      ["int", "float"])
        if RFC_min_samples_leaf_int_or_float == "int":
            RFC_min_samples_leaf = trial.suggest_int("RFC_min_samples_leaf", 1, 5)
        else:
            RFC_min_samples_leaf = trial.suggest_float("RFC_min_samples_leaf", 2, 5)

        model = RandomForestClassifier(n_estimators=trial.suggest_int("RFC_n_estimators", 10, 500),
                                       criterion=trial.suggest_categorical("RFC_criterion", ["gini", "entropy"]),
                                       min_samples_split=RFC_min_samples_split,
                                       min_samples_leaf=RFC_min_samples_leaf,
                                       max_features=trial.suggest_categorical("RFC_max_features", ["auto", "log2"]),
                                       n_jobs=-1)
    else:
        print("Invalid Model Name!")
        return None

    train_x, valid_x, train_y, valid_y = train_test_split(
        X, y, test_size=0.1, random_state=0
    )
    model.fit(train_x, train_y)
    score = model.score(valid_x, valid_y)
    with open("{}Models/{}Score {}.pickle".format(MODELNAME, trial.number, score), "wb") as fout:
        pickle.dump(model, fout)
    return score


X, y = df.drop("label", axis="columns"), df["label"]
X = feature_union.fit_transform(X)
optimization_function = partial(optimize, X=X, y=y)
study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
study.optimize(optimization_function, n_trials=NUM_TRIALS)
file = "{}Models/{}Score {}.pickle".format(MODELNAME, study.best_trial.number, study.best_value)
with open(file, "rb") as fin:
    best_model = pickle.load(fin)
score = re.search(r"Score(.*?).pickle", file).group(1)
with open("{}Models/Best/Score {}.pickle".format(MODELNAME, score), "wb") as fout:
    pickle.dump(best_model, fout)

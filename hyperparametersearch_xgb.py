from functools import partial
import optuna
import pickle
import numpy as np
import xgboost as xgb
from CombineDatasets import get_combined_features_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PreprocessPipe import get_feature_union
import re

MODELNAME = "XGBC"
NUM_TRIALS = 3

df = get_combined_features_dataset()
feature_union = get_feature_union(df)


def optimize(trial, X, y):
    if MODELNAME == "XGBC":
        train_x, valid_x, train_y, valid_y = train_test_split(
            X, y, test_size=0.1, random_state=0
        )
        dtrain = xgb.DMatrix(train_x, label=train_y)
        dvalid = xgb.DMatrix(valid_x, label=valid_y)

        param = {
            "verbosity": 0,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        }

        if param["booster"] == "gbtree" or param["booster"] == "dart":
            param["max_depth"] = trial.suggest_int("max_depth", 1, 9)
            param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
            param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
            param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
        if param["booster"] == "dart":
            param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
            param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
            param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
            param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

        pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-auc")
        model = xgb.train(params=param, dtrain=dtrain, evals=[(dvalid, "validation")], callbacks=[pruning_callback])
        preds = model.predict(dvalid)
        pred_labels = np.rint(preds)
        accuracy = accuracy_score(valid_y, pred_labels)
        with open("{}Models/{}Score {}.pickle".format(MODELNAME, trial.number, accuracy), "wb") as fout:
            pickle.dump(model, fout)
        return accuracy
    else:
        print("Invalid Model Name!")
        return


X, y = df.drop("label", axis="columns"), df["label"]
X = feature_union.fit_transform(X)
optimization_function = partial(optimize, X=X, y=y)
study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
study.optimize(optimization_function, n_trials=NUM_TRIALS)
file = "{}Models/{}Score {}.pickle".format(MODELNAME, study.best_trial.number, study.best_value)
with open(file, "rb") as fin:
    best_model = pickle.load(fin)
score = re.search(r"Score(.*?).pickle", file).group(1)
with open("{}Models/Best/Score {}.pickle".format(MODELNAME, score), "wb") as fout:
    pickle.dump(best_model, fout)

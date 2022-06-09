from functools import partial
import optuna
import pickle
from sklearn.linear_model import SGDClassifier
from CombineDatasets import get_combined_features_dataset
from sklearn.model_selection import train_test_split
from PreprocessPipe import get_feature_union
import re


MODELNAME = "SGD"
NUM_TRIALS = 50

df = get_combined_features_dataset()
feature_union = get_feature_union(df)


def optimize(trial, X, y):
    classes = list(set(df['label'].values))

    if MODELNAME == "SGD":
        sgd_l1_ratio = 0.15
        sgd_penalty = trial.suggest_categorical("sgd_penalty", ["l2", "l1", "elasticnet"])
        if sgd_penalty == "elasticnet":
            sgd_l1_ratio = trial.suggest_float("sgd_l1_ratio", 0, 1)

        model = SGDClassifier(loss=trial.suggest_categorical("sgd_loss", ["hinge", "log", "squared_hinge",
                                                                          "perceptron", "squared_error",
                                                                          "epsilon_insensitive"]),
                              alpha=trial.suggest_float("sgd_alpha", 0.00001, 0.001),
                              power_t=trial.suggest_float("sgd_power_t", 0.1, 1),
                              penalty=sgd_penalty,
                              l1_ratio=sgd_l1_ratio,
                              max_iter=trial.suggest_int("sgd_max_iter", 500, 10000),
                              learning_rate=trial.suggest_categorical("sgd_learning_rate",
                                                                      ["constant", "optimal",
                                                                       "invscaling", "adaptive"]),
                              eta0=trial.suggest_float("sgd_eta0", 0, 10),
                              n_jobs=-1)
    else:
        print("Invalid Model Name!")
        return None

    train_x, valid_x, train_y, valid_y = train_test_split(
        X, y, test_size=0.1, random_state=0
    )
    for step in range(100):
        model.partial_fit(train_x, train_y, classes=classes)

        intermediate_value = model.score(valid_x, valid_y)
        trial.report(intermediate_value, step)

        if trial.should_prune():
            raise optuna.TrialPruned()
    score = model.score(valid_x, valid_y)
    with open("{}Models/{}Score {}.pickle".format(MODELNAME, trial.number, score), "wb") as fout:
        pickle.dump(model, fout)
    return score


X, y = df.drop("label", axis="columns"), df["label"]
X = feature_union.fit_transform(X)
optimization_function = partial(optimize, X=X, y=y)
study = optuna.create_study(direction="maximize")
study.optimize(optimization_function, n_trials=NUM_TRIALS)
file = "{}Models/{}Score {}.pickle".format(MODELNAME, study.best_trial.number, study.best_value)
with open(file, "rb") as fin:
    best_model = pickle.load(fin)
score = re.search(r"Score(.*?).pickle", file).group(1)
with open("{}Models/Best/Score {}.pickle".format(MODELNAME, score), "wb") as fout:
    pickle.dump(best_model, fout)

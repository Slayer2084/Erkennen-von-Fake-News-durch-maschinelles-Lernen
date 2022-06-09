import logging
import sys
from xgboost import XGBClassifier
import matplotlib.pyplot
import pandas as pd
import numpy as np
import optuna
from TweetPreprocessing import Preprocessor
from functools import partial
import matplotlib as plt
from sklearn.model_selection import cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, make_scorer, precision_score, recall_score
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion, make_union
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier, Perceptron

df = pd.read_csv("NonPullable/2Cleaned_Fake_News_Dataset.csv", index_col='index', sep=";").reset_index(drop=True)
preprocessor = Preprocessor(df)
df = preprocessor.add_removed_names().remove_chars().add_lemmatized_to_df().add_convert_emoji().remove_spaces(chain=False)
preprocessor = Preprocessor(df)
preprocessor.get_report()
n_cv = 3
X, Y = df.drop('label', axis="columns"), df["label"]


def optimize(trial, x, y):
    analyzer = trial.suggest_categorical("analyzer", ["word", "char", "char_wb"])
    binary = trial.suggest_categorical("binary", [True, False])
    sublinear_tf = trial.suggest_categorical("sublinear_tf", [True, False])
    norm = trial.suggest_categorical("norm", ["l1", "l2"])
    use_idf = trial.suggest_categorical("use_idf", [True, False])
    smooth_idf = trial.suggest_categorical("smooth_idf", [True, False])
    lowercase = trial.suggest_categorical("lowercase", [True, False])
    strip_accents = trial.suggest_categorical("strip_accents", ["ascii", "unicode", None])
    ngram_range = trial.suggest_categorical("ngram_range", ["1, 1", "1, 2", "2, 2", "1, 3", "2, 3", "3, 3"])
    if ngram_range == "1, 1": ngram_range = [1, 1]
    if ngram_range == "1, 2": ngram_range = [1, 2]
    if ngram_range == "1, 3": ngram_range = [1, 3]
    if ngram_range == "2, 2": ngram_range = [2, 2]
    if ngram_range == "2, 3": ngram_range = [2, 3]
    if ngram_range == "3, 3": ngram_range = [3, 3]
    pas_agr_C = trial.suggest_float("pas_agr_C", 0.1, 1)
    pas_agr_max_iter = trial.suggest_int("pas_agr_max_iter", 500, 10000)
    pas_agr_tol = trial.suggest_categorical("pas_agr_tol", ["float", None])
    if pas_agr_tol == "float":
        pas_agr_tol = trial.suggest_float("pas_agr_tol_float", 0.00001, 0.01)
    pas_agr_validation_fraction = trial.suggest_float("pas_agr_validation_fraction", 0.1, 1)
    pas_agr_loss = trial.suggest_categorical("pas_agr_loss", ["hinge", "squared_hinge"])

    '''
    n_estimator = trial.suggest_int("n_estimator", 100, 2500)
    max_depth = trial.suggest_int("max_depth", 1, 100)
    learning_rate = trial.suggest_float("learning_rate", 0.0001, 2)
    reg_alpha = trial.suggest_float("reg_alpha", 0.00001, 0.01)
    reg_lambda = trial.suggest_float("reg_lambda", 0.00001, 0.01)
    base_score = trial.suggest_float("base_score", 0.01, 2)
    '''

    classes = list(set(df['label'].values))
    train_x, valid_x, train_y, valid_y = train_test_split(
        x, y, test_size=0.1, random_state=0
    )

    tweet_data = ColumnTransformer([
        ("transform", TfidfVectorizer(stop_words="english", ngram_range=ngram_range, strip_accents=strip_accents,
                                      lowercase=lowercase, smooth_idf=smooth_idf, analyzer=analyzer, use_idf=use_idf,
                                      sublinear_tf=sublinear_tf, norm=norm, binary=binary), "tweet")
    ])

    sentiment_data = ColumnTransformer([
        ("scaler", RobustScaler(), ["polarity", "subjectivity"])
    ])

    feature_union = FeatureUnion([
        ("tweet_data", tweet_data),
        ("sentiment_data", sentiment_data)
    ])
    feature_union.fit(train_x)
    train_x, valid_x = feature_union.transform(train_x), feature_union.transform(valid_x)

    '''model = XGBClassifier(n_jobs=-1, n_estimators=n_estimator, max_depth=max_depth, learning_rate=learning_rate, reg_alpha=reg_alpha, reg_lambda=reg_lambda, base_score=base_score)
'''
    model = PassiveAggressiveClassifier(C=pas_agr_C, loss=pas_agr_loss, validation_fraction=pas_agr_validation_fraction, max_iter=pas_agr_max_iter,
                                        tol=pas_agr_tol)
    for step in range(100):
        model.partial_fit(train_x, train_y, classes=classes)

        intermediate_value = model.score(valid_x, valid_y)
        trial.report(intermediate_value, step)

        if trial.should_prune():
            raise optuna.TrialPruned()

    return model.score(valid_x, valid_y)


optimization_function = partial(optimize, x=X, y=Y)
study = optuna.create_study(pruner=optuna.pruners.MedianPruner(), direction="maximize")
study.optimize(optimization_function, n_trials=300)
print(study.best_trial)

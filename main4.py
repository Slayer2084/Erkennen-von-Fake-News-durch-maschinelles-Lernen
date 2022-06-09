import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import optuna
from functools import partial
import matplotlib as plt
from sklearn.model_selection import cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, make_scorer, precision_score, recall_score
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion, make_union
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier, Perceptron
from TweetTextProcessor import TweetTextProcessor, DataFrameColumnExtracter
le = LabelEncoder()
df = pd.read_csv("NonPullable/2Cleaned_Fake_News_Dataset.csv", index_col='index', sep=";").reset_index(drop=True)
n_cv = 3
X, y = df.drop('label', axis="columns"), df["label"]
y = le.fit_transform(y)




def optimize(trial, x, y):
    '''sgd_alpha = trial.suggest_float("sgd_alpha", 0.00001, 0.001)
    pas_agr_loss = trial.suggest_categorical("pas_agr_loss", ["hinge", "squared_hinge"])
    perc_alpha = trial.suggest_float("perc_alpha", 0.00001, 0.001)
    sgd_weight = trial.suggest_float("sgd_weight", 0.01, 10)
    pas_agr_weight = trial.suggest_float("pas_agr_weight", 0.01, 10)
    perc_weight = trial.suggest_float("perc_weight", 0.01, 10)'''
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
    '''perc_eta0 = trial.suggest_float("perc_eta0", 0.1, 1)
    # sgd_loss = trial.suggest_categorical("sgd_loss", ["hinge", "squared_hinge"])
    perc_validation_fraction = trial.suggest_float("perc_validation_fraction", 0.1, 1)
    power_t = trial.suggest_float("sgd_power_t", 0.1, 1)'''
    n_estimator = trial.suggest_int("n_estimator", 100, 2500)
    max_depth = trial.suggest_int("max_depth", 1, 100)
    learning_rate = trial.suggest_float("learning_rate", 0.0001, 2)
    reg_alpha = trial.suggest_float("reg_alpha", 0.00001, 0.01)
    reg_lambda = trial.suggest_float("reg_lambda", 0.00001, 0.01)
    base_score = trial.suggest_float("base_score", 0.01, 1)

    tweet_data = ColumnTransformer([
        ("transform", TfidfVectorizer(stop_words="english", ngram_range=ngram_range, strip_accents=strip_accents, lowercase=lowercase, smooth_idf=smooth_idf, analyzer=analyzer, use_idf=use_idf, sublinear_tf=sublinear_tf, norm=norm, binary=binary), "tweet")
    ])

    sentiment_data = ColumnTransformer([
        ("pol", RobustScaler(), ["polarity", "subjectivity"])
    ])

    feature_union = FeatureUnion([
        ("tweet_data", tweet_data),
        ("sentiment_data", sentiment_data)
    ])
    train_x, valid_x, train_y, valid_y = train_test_split(
        x, y, test_size=(0.1), random_state=0
    )
    feature_union.fit(train_x)
    train_x, valid_x = feature_union.transform(train_x), feature_union.transform(valid_x)
    model = XGBClassifier(n_jobs=-1, n_estimators=n_estimator, max_depth=max_depth, learning_rate=learning_rate,
                          reg_alpha=reg_alpha, reg_lambda=reg_lambda, base_score=base_score, use_label_encoder=False)
    model.fit(train_x, train_y)

    return model.score(valid_x, valid_y)


optimization_function = partial(optimize, x=X, y=y)
study = optuna.create_study(direction="maximize")
study.optimize(optimization_function, n_trials=400)
print(study.best_trial)

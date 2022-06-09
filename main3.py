import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.calibration import CalibratedClassifierCV
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
df = pd.read_csv("NonPullable/2Cleaned_Fake_News_Dataset.csv", index_col='index', sep=";").reset_index(drop=True)
n_cv = 3
X, y = df.drop('label', axis="columns"), df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

print(X_train)

tweet_data = ColumnTransformer([
    ("transform", TfidfVectorizer(stop_words='english', use_idf=True, smooth_idf=True), 'tweet')
])

sentiment_data = ColumnTransformer([
    ("pol", RobustScaler(), ['polarity', 'subjectivity'])
])

feature_union = FeatureUnion([
    ("tweet_data", tweet_data),
    ("sentiment_data", sentiment_data)
])

pas_agr_clf = Pipeline([
    ("data", feature_union),
    ("clf", CalibratedClassifierCV(PassiveAggressiveClassifier(), cv=n_cv))
])

sgd_clf = Pipeline([
    ("data", feature_union),
    ("clf", CalibratedClassifierCV(SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42,
                                             max_iter=5, tol=None)))
])

perc_clf = Pipeline([
    ("data", feature_union),
    ("clf", CalibratedClassifierCV(Perceptron()))
])


clf = VotingClassifier([
    ("sgd", sgd_clf),
    ("pas_agr", pas_agr_clf),
    ("perc", perc_clf)
], voting='soft')


print(clf.get_params().keys())

parameters = {
    'sgd__data__tweet_data__transform__lowercase': (True, False),
    'sgd__data__tweet_data__transform__ngram_range': ([1, 1], [1, 2], [1, 3], [2, 3], [3, 3], [1, 4], [2, 4], [3, 4], [4, 4]),
    'sgd__data__tweet_data__transform__strip_accents': ('ascii', 'unicode', None),
    'sgd__data__tweet_data__transform__use_idf': (True, False),
    'sgd__data__tweet_data__transform__smooth_idf': (True, False),

    'pas_agr__data__tweet_data__transform__lowercase': (True, False),
    'pas_agr__data__tweet_data__transform__ngram_range': ([1, 1], [1, 2], [1, 3], [2, 3], [3, 3], [1, 4], [2, 4], [3, 4], [4, 4]),
    'pas_agr__data__tweet_data__transform__strip_accents': ('ascii', 'unicode', None),
    'pas_agr__data__tweet_data__transform__use_idf': (True, False),
    'pas_agr__data__tweet_data__transform__smooth_idf': (True, False),

    'perc__data__tweet_data__transform__lowercase': (True, False),
    'perc__data__tweet_data__transform__ngram_range': ([1, 1], [1, 2], [1, 3], [2, 3], [3, 3], [1, 4], [2, 4], [3, 4], [4, 4]),
    'perc__data__tweet_data__transform__strip_accents': ('ascii', 'unicode', None),
    'perc__data__tweet_data__transform__use_idf': (True, False),
    'perc__data__tweet_data__transform__smooth_idf': (True, False),

    # 'sgd__clf__base_estimator__alpha': np.linspace(0.0001, 0.000001, 100),
    'sgd__clf__base_estimator__loss': ('hinge', 'modified_huber', 'squared_hinge',
                       'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'),
    # 'sgd__clf__base_estimator__power_t': np.linspace(0.1, 1, 10, dtype=int),

    # 'pas_agr__clf__base_estimator__max_iter': np.linspace(100, 5000, 900, dtype=int),
    # 'pas_agr__clf__base_estimator__validation_fraction': np.linspace(0.1, 1, 20),
    'pas_agr__clf__base_estimator__loss': ('hinge', 'squared_hinge'),

    'perc__clf__base_estimator__penalty': ('l2', 'l1', 'elasticnet'),
    # 'perc__clf__base_estimator__alpha': np.linspace(0.001, 0.00001, 100),
    # 'perc__clf__base_estimator__max_iter': np.linspace(100, 5000, 1000, dtype=int),
    # 'perc__clf__base_estimator__eta0': np.linspace(0.1, 3, 100),
    'perc__clf__base_estimator__validation_fraction': np.linspace(0.1, 0.5, 10),

}

mod = RandomizedSearchCV(
    clf,
    param_distributions=parameters,
    n_iter=1000000000,
    cv=n_cv,
    n_jobs=-1,
    verbose=3
)

mod.fit(X_train, y_train)
print(mod.best_score_)
print(mod.best_params_)
grid_results = pd.DataFrame(data=[
    [mod.best_score_, str(mod.best_params_)]
])
grid_results.to_csv(path_or_buf="NonPullable/grid_results.csv", sep=";", mode="a", header=False, index=False)

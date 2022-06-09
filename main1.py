import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn import metrics
from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, make_scorer, accuracy_score
from sklearn.model_selection import train_test_split
from sklego.preprocessing import ColumnSelector
from sklego.datasets import load_chicken
from sklearn.linear_model import PassiveAggressiveClassifier

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

df = pd.read_csv('NonPullable/2Cleaned_Fake_News_Dataset.csv', index_col='index', sep=';')
X, y = df.drop('label', axis="columns"), df[['label']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

feature_pipeline = Pipeline([
    ("datagrab", FeatureUnion([
        ("discrete", Pipeline([
            ("grab", ColumnSelector("polarity")),
        ])),
        ("continuous", Pipeline([
            ("grab", ColumnSelector("tweet")),
            ("vect", CountVectorizer(stop_words='english')),
            ("tf-idf", TfidfTransformer())
        ]))
    ]))
])

pipe = Pipeline([
    ("transform", feature_pipeline),
    ("clf", SGDClassifier(loss='hinge', penalty='l2',
                          alpha=1e-3, random_state=42,
                          max_iter=5, tol=None))
])
print("Pipe Params:", pipe.get_params().keys())
# pipe.fit(X_train, y_train)
# predicted = pipe.predict(X_test)
# print(np.mean(predicted == y_test))
# print(metrics.classification_report(y_test, predicted))

parameters = {
    'transform__datagrab__continuous__vect__ngram_range': [(1, 1), (1, 2)],
    'transform__datagrab__continuous__tf-idf__use_idf': (True, False),
    'clf__alpha': (1e-2, 1e-3),

}

mod = GridSearchCV(estimator=pipe,
                   param_grid=parameters,
                   scoring={'precision': make_scorer(precision_score), 'recall': make_scorer(recall_score), 'accuracy': make_scorer(accuracy_score)},
                   refit='recall',
                   return_train_score=True,
                   cv=3,
                   n_jobs=-1)

mod.fit(X, y)
print(pd.DataFrame(mod.cv_results_))

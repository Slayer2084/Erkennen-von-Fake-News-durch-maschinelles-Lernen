import pandas as pd
import numpy as np
from sklearn import *
from sklearn.base import TransformerMixin, BaseEstimator
from textblob import TextBlob
import matplotlib.pylab as plt
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn import metrics
from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, make_scorer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.compose import ColumnTransformer
from sklego.preprocessing import ColumnSelector
pd.set_option("display.max_rows", 10, "display.max_columns", None)
enc = LabelBinarizer()


df = pd.read_csv('NonPullable/2Cleaned_Fake_News_Dataset.csv', index_col='index', sep=';')

X, y = df.drop("label", axis="columns"), df['label']
print(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)


class TweetTextProcessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.tweet_text_transformer = Pipeline(steps=[
            ('count_vectoriser', CountVectorizer(stop_words='english')),
            ('tfidf', TfidfTransformer())])

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return self.tweet_text_transformer.fit_transform(X.squeeze()).toarray()


feature_pipeline = ColumnTransformer(transformers=[
    ("tf-idf-vect", TweetTextProcessor(),['tweet']),
    ], remainder="passthrough")





pipe = Pipeline([
    ('transform', feature_pipeline),
    ('clf', SGDClassifier(loss='hinge', penalty='l2',
                          alpha=1e-3, random_state=42,
                          max_iter=5, tol=None))
])
print(pipe.get_params().keys())
# pipe.fit(X_train, y_train)
# predicted = pipe.predict(X_test)
# print(np.mean(predicted == y_test))
# print(metrics.classification_report(y_test, predicted))

parameters = {
    'clf__alpha': (1e-2, 1e-3),
    'clf__loss': ('hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive')

}

mod = GridSearchCV(estimator=pipe,
                   param_grid=parameters,
                   # scoring={'precision': make_scorer(precision_score), 'recall': make_scorer(recall_score)},
                   # refit='recall',
                   # return_train_score=True,
                   cv=3,
                   n_jobs=-1)

mod.fit(X_train, y_train)
print(np.mean(mod.predict(X_test) == y_test))
print(pd.DataFrame(mod.cv_results_))


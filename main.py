import pandas as pd
import xgboost
import pickle
from FeatureEngineering import get_features
from PreprocessPipe import get_feature_union
import time

INPUT = "The single most effective, safe and simple way to protect yourself and your loved ones from covid-19 is to wear your mask properly."
MODEL = "XGBC"
time1 = time.time()
model = pickle.load(open(MODEL + "Models/Best/Score  0.9153754469606674.pickle", "rb"))
input_df = pd.DataFrame(data={"content": [INPUT]})
time3 = time.time()
df_ft = get_features(input_df)
time4 = time.time()

pipe = get_feature_union(df_ft)
X = pipe.fit_transform(df_ft)
if MODEL == "XGBC":
    X = xgboost.DMatrix(X)

predict = model.predict(X)
time2 = time.time()
if predict:
    print(MODEL + "-Model has declared this tweet as real.")
else:
    print(MODEL + "-Model has declared this tweet as fake.")
print("Whole thing:", time2 - time1)
print("Time for features", time4 - time3)

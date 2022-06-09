import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
import numpy as np
import time


class CorrectLabels:

    def __init__(self,
                 dataset: pd.DataFrame,
                 label_column_name: str,
                 epochs: int,
                 threshold: float,
                 preprocessing_pipe,
                 repeats: int = 500,
                 split_rate: int = 4,
                 ):
        self.dataset = dataset
        self.threshold = threshold
        self.preprocessing_pipe = preprocessing_pipe
        self.label_column_name = label_column_name
        self.index_column_name = "index"
        self.repeats = repeats
        self.epochs = epochs
        self.split_rate = split_rate
        self.models = self.form_models()
        self.dataset.index = range(len(self.dataset.index))
        self.dataset["index"] = self.dataset.index

    @staticmethod
    def shuffle_dataset(dataset):
        print("Shuffling Data...")
        shuffled_dataset = shuffle(dataset)
        return shuffled_dataset

    def get_train_test(self, dataset):
        X = dataset.drop([self.label_column_name, self.index_column_name], axis="columns")
        y = dataset[self.label_column_name]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1 / self.split_rate), random_state=7)
        print("Transforming Data...")
        self.preprocessing_pipe.fit(X)
        X_train = self.preprocessing_pipe.transform(X_train)
        return X_train, X_test, y_train, y_test

    @staticmethod
    def form_models():
        models = {}
        print("Forming Models...")
        models["LogisticRegression"] = LogisticRegression(n_jobs=-1, max_iter=10000)
        # models["PassiveAggressiveClassifier"] = PassiveAggressiveClassifier(n_jobs=-1) No predict_proba
        models["KNN"] = KNeighborsClassifier(n_jobs=-1)
        models["SGDClassifier"] = SGDClassifier(loss="modified_huber", n_jobs=-1)
        models["DecisionTreeClassifier"] = DecisionTreeClassifier()
        models["RandomForestClassifier"] = RandomForestClassifier(n_jobs=-1)
        models["AdaBoostClassifier"] = AdaBoostClassifier()
        # models["GradientBoostingClassifier"] = GradientBoostingClassifier()
        # models["GaussianNB"] = GaussianNB()
        # models["MultinomialNB"] = MultinomialNB()
        # models["Perceptron"] = Perceptron(n_jobs=-1) No predict_proba
        models["XGBoost"] = XGBClassifier(n_jobs=-1, use_label_encoder=False)

        return models

    def get_trained_models(self, X_train, y_train):
        fitted_models = {}
        print("Starting to train models...")
        time3 = time.time()
        for idx, (model_name, model) in enumerate(self.models.items()):
            time1 = time.time()
            model.fit(X_train, y_train)
            fitted_models[model_name] = model
            time2 = time.time()
            # print("Successfully trained ", model_name, "in ", ((time2-time1)*1000.0), "ms, only ",
            # (len(self.models) - idx - 1), " more to go!")
        time4 = time.time()
        print("Successfully trained all models in", ((time4 - time3) * 1000.0), "ms!")
        return fitted_models

    def get_predict(self, X_test, fitted_models: dict):
        print("Starting predictions...")
        X_test = self.preprocessing_pipe.transform(X_test)
        preds = {}
        for model_name, model in fitted_models.items():
            predict = model.predict(X_test)
            predict_proba = model.predict_proba(X_test)
            mask = np.max(predict_proba, axis=1) > self.threshold
            preds[model_name] = [predict, mask]
        return preds

    def repeat(self, dataset):
        index_counter_dict = {}
        # Füllen des index_counter_dict mit "Reihe" : [0, 0]
        for row in range(len(self.dataset)):
            index_counter_dict[row] = [0, 0]

        for idx in range(self.repeats):
            shuffled_dataset = self.shuffle_dataset(dataset)
            X_train, X_test, y_train, y_test = self.get_train_test(shuffled_dataset)
            fitted_models = self.get_trained_models(X_train, y_train)
            predictions = self.get_predict(X_test, fitted_models)
            index = X_test.index
            for model_name, preds in predictions.items():
                i = 0
                for row in index:  # Für jede Reihe
                    one_or_zero_prediction = preds[0]
                    mask = preds[1]

                    if mask[i]:
                        if one_or_zero_prediction[i] == 0:
                            index_counter_dict[row][0] += 1
                        else:
                            index_counter_dict[row][1] += 1

                    i += 1
            print("Completed Repetition ", idx + 1, "out of ", self.repeats, "!")
        return index_counter_dict

    def clean_up_bad_labels(self):
        dataset = self.dataset
        for epoch in range(self.epochs):
            time1 = time.time()
            index_counter_dict = self.repeat(dataset)
            for row, result in index_counter_dict.items():
                if result[0] > 3 and result[1] > 3:
                    dataset.copy()[row] = np.max(result)
            time2 = time.time()
            print("Completed Epoch ", epoch + 1, "out of ", self.epochs, "in ", ((time2 - time1) * 1000.0), "ms!")
        return dataset


if __name__ == "__main__":
    from CombineDatasets import get_combined_features_dataset
    from PreprocessPipe import get_feature_union

    df = get_combined_features_dataset().sample(100)
    pipe = get_feature_union(df)
    label_cleaner = CorrectLabels(df, "label", 1, 0.9, pipe, 4, split_rate=8)
    refined_df = label_cleaner.clean_up_bad_labels()
    print(refined_df)
    # refined_df.to_csv(path_or_buf="/data/RefinedCombinedDataset.csv", sep=";")

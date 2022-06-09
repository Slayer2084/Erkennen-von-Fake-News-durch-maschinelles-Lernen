import time
import datetime
import pandas as pd


def get_combined_features_dataset():
    df = pd.read_csv(filepath_or_buffer="data/CombinedWithFeatures2.csv", sep=";")
    df = df.drop(["index2", "index", "tweet_object", "op_object", "opProtected", "num_rare_words"], axis="columns")
    df["opCreated"] = df["opCreated"].apply(lambda x: time.mktime(datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S+00:00").timetuple()))
    df["time"] = df["time"].apply(lambda x: time.mktime(datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S+00:00").timetuple()))
    # Dropping Features because OutOfMemory Errors
    df = df.drop(["converted_emojis", "removed_rare_words", "content", "lemmatized", "removed_names"], axis="columns")
    return df


def get_combined_dataset():
    mickes = pd.read_csv("data/MickesClubhouseCOVID19RumourDataset/MickesClubhouseCOVID19RumourDataset.csv", sep=",")
    constraint_test = pd.read_csv("data/COVID19 Fake News Detection in English Dataset/Constraint_Test.csv", sep=",")
    constraint_train = pd.read_csv("data/COVID19 Fake News Detection in English Dataset/Constraint_Train.csv", sep=",")
    constraint_val = pd.read_csv("data/COVID19 Fake News Detection in English Dataset/Constraint_Val.csv", sep=",")
    constraint = pd.concat([constraint_test, constraint_train, constraint_val], ignore_index=True, sort=False)
    constraint = constraint.drop(columns="id")
    constraint["label"] = constraint["label"].map({"real": 1, "fake": 0})
    constraint = constraint.rename({"tweet": "content"}, axis=1)
    mickes = mickes[mickes.label != "U"]
    mickes["label"] = mickes["label"].map({"T": 1, "F": 0})
    combined = pd.concat([mickes, constraint], ignore_index=True, sort=False)
    return combined


if __name__ == "__main__":
    print(get_combined_features_dataset().columns)

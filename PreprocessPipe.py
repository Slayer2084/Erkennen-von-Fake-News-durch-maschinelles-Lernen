from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn_pandas import DataFrameMapper

text_column = ["content", "lemmatized", "removed_names", "converted_emojis", "removed_rare_words"]

num_column = ["likeCount", "quoteCount", "retweetCount", "replyCount", "nHashtags",
              "nMentionedUsers", "opFollowerCount", "opFollowingCount", "opPostedTweetsCount",
              "opFavouritesCount", "opListedCount", "polarity", "subjectivity", "char_count",
              "verb_count", "noun_count", "adj_count", "word_count", "sent_count", "avg_word_len",
              "avg_sent_len", "num_lemmatized_words"]

hot_encoded_column = ["source"]  # ["source", "hashtags", "shortened_outlinks"]

# Removed because OutOfMemory
"""("removed_rare_words", TfidfVectorizer(stop_words="english", ngram_range=[1, 3], strip_accents=None,
                                               lowercase=False, smooth_idf=False, analyzer='char', use_idf=True,
                                               sublinear_tf=True, norm='l2', binary=True)),"""
"""("converted_emojis", TfidfVectorizer(stop_words="english", ngram_range=[1, 3], strip_accents=None,
                                             lowercase=False, smooth_idf=False, analyzer='char', use_idf=True,
                                             sublinear_tf=True, norm='l2', binary=True)),"""
"""("content", TfidfVectorizer(stop_words="english", ngram_range=[1, 3], strip_accents=None,
                                    lowercase=False, smooth_idf=False, analyzer='char', use_idf=True,
                                    sublinear_tf=True, norm='l2', binary=True)),"""
"""("removed_names", TfidfVectorizer(stop_words="english", ngram_range=[1, 3], strip_accents=None,
                                          lowercase=False, smooth_idf=False, analyzer='char', use_idf=True,
                                          sublinear_tf=True, norm='l2', binary=True)),"""
"""("lemmatized", TfidfVectorizer(stop_words="english", ngram_range=[1, 3], strip_accents=None,
                                       lowercase=False, smooth_idf=False, analyzer='char', use_idf=True,
                                       sublinear_tf=True, norm='l2', binary=True)),"""


def get_feature_union(df):
    tag_column = [k for k in df.columns if 'tags_' in k]

    mapper = DataFrameMapper([
        ("lemm_removed_names", TfidfVectorizer(stop_words=None, ngram_range=[1, 3], strip_accents=None,
                                               lowercase=False, smooth_idf=False, analyzer='char', use_idf=True,
                                               sublinear_tf=True, norm='l2', binary=True)),
        (num_column, RobustScaler()),
        (tag_column, RobustScaler()),
        (hot_encoded_column, OneHotEncoder())
    ])

    return mapper


if __name__ == "__main__":
    from CombineDatasets import get_combined_features_dataset

    df_features = get_combined_features_dataset()
    pipe = get_feature_union(df_features)
    X = df_features.drop("label", axis="columns")
    y = df_features["label"]
    print(X)
    print(y)
    transformed_X = pipe.fit_transform(X)
    print(transformed_X)

import pandas as pd
import ScrapeTweets
import spacy
from usefull import time_it
import collections
from urllib.parse import urlsplit
from TweetPreprocessing import Preprocessor, remove_names


def shorten_links(link_list):
    shortened_link_list = []
    for link in link_list:
        shortened_link_list.append("{0.scheme}://{0.netloc}/".format(urlsplit(link)))
    return shortened_link_list


# Function from https://towardsdatascience.com/text-analysis-feature-engineering-with-nlp-502d6ea9225d
def utils_ner_features(lst_dics_tuples, tag):
    if len(lst_dics_tuples) > 0:
        tag_type = []
        for dic_tuples in lst_dics_tuples:
            for tuple in dic_tuples:
                type, n = tuple[1], dic_tuples[tuple]
                tag_type = tag_type + [type] * n
                dic_counter = collections.Counter()
                for x in tag_type:
                    dic_counter[x] += 1
        return dic_counter[tag]
    else:
        return 0


def get_num_changed_words(string1: str, string2: str):
    word_list_string1 = string1.split()
    word_list_string2 = string2.split()
    count = 0
    if len(word_list_string1) != len(word_list_string2):
        return 0
    for i in range(len(word_list_string1)):
        if word_list_string1[i] != word_list_string2[i]:
            count += 1
    return count


def utils_lst_count(
        lst):  # Function from https://towardsdatascience.com/text-analysis-feature-engineering-with-nlp-502d6ea9225d
    dic_counter = collections.Counter()
    for x in lst:
        dic_counter[x] += 1
    dic_counter = collections.OrderedDict(
        sorted(dic_counter.items(),
               key=lambda x: x[1], reverse=True))
    lst_count = [{key: value} for key, value in dic_counter.items()]
    return lst_count


def get_length(obj):
    if obj is None:
        return 0
    else:
        return len(obj)


def list_none_to_string(liste: list):
    if liste is None:
        return ["None"]
    new_list = []
    for item in liste:
        if item is None:
            item = "None"
        new_list.append(item)
    return new_list


@time_it
def get_features(df_ft):
    df_ft["tweet_object"] = df_ft["content"].apply(lambda text: ScrapeTweets.get_tweet_object(text))
    df_ft = df_ft[df_ft["tweet_object"].notna()]
    df_ft["time"] = df_ft["tweet_object"].apply(lambda tweet: None if tweet is None else tweet.date)
    df_ft["likeCount"] = df_ft["tweet_object"].apply(lambda tweet: None if tweet is None else tweet.likeCount)
    df_ft["quoteCount"] = df_ft["tweet_object"].apply(lambda tweet: None if tweet is None else tweet.quoteCount)
    df_ft["source"] = df_ft["tweet_object"].apply(lambda tweet: None if tweet is None else tweet.sourceLabel)
    df_ft["retweetCount"] = df_ft["tweet_object"].apply(lambda tweet: None if tweet is None else tweet.retweetCount)
    df_ft["replyCount"] = df_ft["tweet_object"].apply(lambda tweet: None if tweet is None else tweet.replyCount)
    df_ft["op_object"] = df_ft["tweet_object"].apply(lambda tweet: None if tweet is None else tweet.user)
    df_ft["hashtags"] = df_ft["tweet_object"].apply(lambda tweet: None if tweet is None else tweet.hashtags)
    df_ft["nHashtags"] = df_ft["tweet_object"].apply(
        lambda tweet: None if tweet is None else get_length(tweet.hashtags))
    df_ft = df_ft.drop("hashtags", axis="columns")
    df_ft["mentionedUsers"] = df_ft["tweet_object"].apply(lambda tweet: None if tweet is None else tweet.mentionedUsers)
    df_ft["nMentionedUsers"] = df_ft["tweet_object"].apply(
        lambda tweet: None if tweet is None else get_length(tweet.mentionedUsers))
    df_ft = df_ft.drop("mentionedUsers", axis="columns")
    df_ft["outlinks"] = df_ft["tweet_object"].apply(lambda tweet: None if tweet is None else tweet.outlinks)
    df_ft["shortened_outlinks"] = df_ft["outlinks"].apply(lambda links: None if links is None else shorten_links(links))
    df_ft["shortened_outlinks"] = df_ft["shortened_outlinks"].apply(lambda outlinks: list_none_to_string(outlinks))
    df_ft = df_ft.drop("outlinks", axis="columns")
    df_ft = df_ft.drop("shortened_outlinks", axis="columns")
    # https://stackoverflow.com/questions/47786822/how-do-you-one-hot-encode-columns-with-a-list-of-strings-as-values
    df_ft["author"] = df_ft["op_object"].apply(lambda user: None if user is None else user.username)
    df_ft = df_ft.drop("author", axis="columns")  # No real use case, yet
    df_ft["opVerified"] = df_ft["op_object"].apply(lambda user: None if user is None else user.verified)
    df_ft["opCreated"] = df_ft["op_object"].apply(lambda user: None if user is None else user.created)
    df_ft["opFollowerCount"] = df_ft["op_object"].apply(lambda user: None if user is None else user.followersCount)
    df_ft["opFollowingCount"] = df_ft["op_object"].apply(lambda user: None if user is None else user.friendsCount)
    df_ft["opPostedTweetsCount"] = df_ft["op_object"].apply(lambda user: None if user is None else user.statusesCount)
    df_ft["opFavouritesCount"] = df_ft["op_object"].apply(lambda user: None if user is None else user.favouritesCount)
    df_ft["opListedCount"] = df_ft["op_object"].apply(lambda user: None if user is None else user.listedCount)
    df_ft["opProtected"] = df_ft["op_object"].apply(lambda user: None if user is None else user.protected)
    preprocessor = Preprocessor(df_ft, "content")
    df_ft = preprocessor.add_lemmatized_to_df().add_removed_names().add_pol_subj_to_df().add_convert_emoji() \
        .add_remove_rare_words().remove_chars().remove_spaces(chain=False)
    df_ft["lemm_removed_names"] = df_ft["lemmatized"].apply(lambda x: remove_names(x))
    df_ft["char_count"] = df_ft["content"].apply(lambda x: sum(len(word) for word in str(x).split(" ")))
    ner = spacy.load("en_core_web_lg")
    df_ft["tags"] = df_ft["content"].apply(lambda text: [(tag.text, tag.label_) for tag in ner(text).ents])
    df_ft["tags"] = df_ft["tags"].apply(lambda x: utils_lst_count(x))
    tags_set = []
    for lst in df_ft["tags"].tolist():
        for dic in lst:
            for k in dic.keys():
                tags_set.append(k[1])
    tags_set = list(set(tags_set))
    for feature in tags_set:
        df_ft["tags_" + feature] = df_ft["tags"].apply(lambda x: utils_ner_features(x, feature))
    df_ft = df_ft.drop("tags", axis="columns")
    df_ft["verb_count"] = df_ft["content"].apply(
        lambda text: len([token for token in ner(text) if token.pos_ == 'VERB']))
    df_ft["noun_count"] = df_ft["content"].apply(
        lambda text: len([token for token in ner(text) if token.pos_ == 'NOUN']))
    df_ft["adj_count"] = df_ft["content"].apply(
        lambda text: len([token for token in ner(text) if token.pos_ == 'ADJ']))
    df_ft["word_count"] = df_ft["content"].apply(
        lambda text: len([token for token in ner(text)]))
    df_ft["sent_count"] = df_ft["content"].apply(lambda text: len([[token.text for token in sent]
                                                                   for sent in ner(text).sents]))
    df_ft["avg_word_len"] = df_ft["char_count"] / df_ft["sent_count"]
    df_ft["avg_sent_len"] = df_ft["word_count"] / df_ft["sent_count"]
    df_ft["num_rare_words"] = len(df_ft["removed_rare_words"]) - len(df_ft["content"])
    df_ft["num_lemmatized_words"] = df_ft.apply(lambda x: get_num_changed_words(x['lemmatized'], x["content"]), axis=1)
    df_ft["num_rare_words"] = df_ft.apply(lambda x: get_num_changed_words(x["removed_rare_words"], x["content"]),
                                          axis=1)

    return df_ft


if __name__ == "__main__":
    from CombineDatasets import get_combined_dataset

    pd.options.mode.chained_assignment = None
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.max_columns", None)

    df = get_combined_dataset().sample(10)
    print(get_features(df))

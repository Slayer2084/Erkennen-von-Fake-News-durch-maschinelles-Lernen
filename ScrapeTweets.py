import snscrape.modules.twitter as sntwitter
import snscrape
from datetime import datetime


def get_mentioned_users(text):
    tweet_list = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(text).get_items()):
        if i > 500:
            break
        tweet_list.append(tweet.mentionedUsers)
    return tweet_list[-1]


def get_date(text):
    tweet_list = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(text).get_items()):
        if i > 500:
            break
        tweet_list.append(tweet.date)
    return tweet_list[-1]


def get_reply_count(text):
    tweet_list = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(text).get_items()):
        if i > 500:
            break
        tweet_list.append(tweet.replyCount)
    return tweet_list[-1]


def get_retweet_count(text):
    tweet_list = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(text).get_items()):
        if i > 500:
            break
        tweet_list.append(tweet.retweetCount)
    return tweet_list[-1]


def get_quote_count(text):
    tweet_list = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(text).get_items()):
        if i > 500:
            break
        tweet_list.append(tweet.quoteCount)
    return tweet_list[-1]


def get_source(text):
    tweet_list = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(text).get_items()):
        if i > 500:
            break
        tweet_list.append(tweet.source)
    return tweet_list[-1]


def get_like_count(text):
    tweet_list = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(text).get_items()):
        if i > 500:
            break
        tweet_list.append(tweet.likeCount)
    return tweet_list[-1]


def get_op_created_date(text):
    tweet_list = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(text).get_items()):
        if i > 500:
            break
        tweet_list.append(tweet.user.created)
    return tweet_list[-1]


def get_op_friends_count(text):
    tweet_list = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(text).get_items()):
        if i > 500:
            break
        tweet_list.append(tweet.user.friendsCount)
    return tweet_list[-1]


def get_op_followers_count(text):
    tweet_list = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(text).get_items()):
        if i > 500:
            break
        tweet_list.append(tweet.user.followersCount)
    return tweet_list[-1]


def get_op_description(text):
    tweet_list = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(text).get_items()):
        if i > 500:
            break
        tweet_list.append(tweet.user.description)
    return tweet_list[-1]


def get_op_displayname(text):
    tweet_list = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(text).get_items()):
        if i > 500:
            break
        tweet_list.append(tweet.user.displayname)
    return tweet_list[-1]


def get_op_verified(text):
    tweet_list = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(text).get_items()):
        if i > 500:
            break
        tweet_list.append(tweet.user.verified)
    return tweet_list[-1]


def get_timestamp(text):
    tweet_list = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(text).get_items()):
        if i > 500:
            break
        tweet_list.append(tweet.date)
    print("done")
    return datetime.timestamp(tweet_list[-1])


def get_tweet_object(text):
    try:
        tweet_list = []
        for i, tweet in enumerate(sntwitter.TwitterSearchScraper(text).get_items()):
            if i > 500:
                break
            tweet_list.append(tweet)
        print("done")
        return tweet_list[-1]
    except IndexError:
        print("Not from Twitter")
        return None
    except snscrape.base.ScraperException:
        print("BaseScraperException")
        return None


if __name__ == '__main__':
    print(get_tweet_object(
        "Quotes by Finnish national health authorities between 20th of January and 3rd of March taken out of context").likeCount)

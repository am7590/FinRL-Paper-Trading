from ntscraper import Nitter

'''
Nitter is a scaping library. 
It seems like it's keyword seaching is broken, but we are still able to obtain posts from specific users.
TODO: Find a working scraping library for hashtags or keywords
'''

scraper = Nitter()

# Scrape tweets from a specific user
tweets_data = scraper.get_tweets("elonmusk", mode='user', number=10)
print(tweets_data)

# Inspecting the structure of the data
print(tweets_data.keys())
print(tweets_data['tweets'][0])

# Getting profile information
profile_info = scraper.get_profile_info(username='elonmusk')
print(profile_info)

# Creating a dictionary to store the tweet data
data = {
    'link': [],
    'text': [],
    'user': [],
    'likes': [],
    'retweets': [],
    'comments': []
}

# Extracting data from tweets
for tweet in tweets_data['tweets']:
    data['link'].append(tweet['link'])
    data['text'].append(tweet['text'])
    data['user'].append(tweet['user']['name'])
    data['likes'].append(tweet['stats']['likes'])
    data['retweets'].append(tweet['stats']['retweets'])
    data['comments'].append(tweet['stats']['comments'])

# Creating a DataFrame and saving to CSV
import pandas as pd
df = pd.DataFrame(data)
df.to_csv('main/scraping/results/musk.csv', index=False)
print("Tweets saved to musk.csv")



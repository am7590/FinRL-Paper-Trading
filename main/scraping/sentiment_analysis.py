import pandas as pd
from textblob import TextBlob

'''
This is a super simple sentiment anlysis tool.
We should look into other NLP solutions to get more accurate results.
'''

df = pd.read_csv('main/scraping/results/musk.csv')

df['Sentiment'] = df['text'].apply(lambda tweet: TextBlob(tweet).sentiment.polarity)

print(df.head())

df['Sentiment'].plot(kind='hist', title='Sentiment Analysis of Tweets')
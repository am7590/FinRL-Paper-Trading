import feedparser
import time

'''
This file streams the output from RSS feeds.
I made the RSS streams on https://rss.app/.
This may be our best option for real-time social media post streaming.
rss.app supports Reddit, facebook, LinkedIn and other integrations.
'''

class RSSFeedStreamer:
    def __init__(self, feed_url, check_interval=60):
        self.feed_url = feed_url
        self.check_interval = check_interval
        self.seen_entries = set()

    def fetch_feed(self):
        """Fetch the RSS feed and return new entries."""
        feed = feedparser.parse(self.feed_url)
        new_entries = []
        
        for entry in feed.entries:
            if entry.id not in self.seen_entries:
                self.seen_entries.add(entry.id)
                new_entries.append(entry)
        
        return new_entries

    def stream(self):
        """Continuously fetch the RSS feed at intervals and print new entries."""
        print("Starting RSS feed stream...")
        while True:
            new_entries = self.fetch_feed()
            for entry in new_entries:
                print(f"Title: {entry.title}")
                print(f"Link: {entry.link}")
                print(f"Published: {entry.published}")
                print("----")

            time.sleep(self.check_interval)

if __name__ == "__main__":
    feed_url = "https://rss.app/feeds/sELiL1zwWPUe2XcW.xml"
    streamer = RSSFeedStreamer(feed_url, check_interval=60)
    streamer.stream()

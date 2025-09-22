import praw
import pandas as pd
import time
from datetime import datetime
import os

# Reddit API Setup
# You'll need to get these from https://www.reddit.com/prefs/apps
REDDIT_CONFIG = {
    'client_id': '4E_Zv9Bnf3kL10-auGVjkg',
    'client_secret': 'HWls7zV8MabgKFJeSHO1apT4ppExRw', 
    'user_agent': 'fallacy_collector_v1.0'
}

class RedditCommentCollector:
    def __init__(self):
        """Initialize Reddit API connection"""
        try:
            self.reddit = praw.Reddit(
                client_id=REDDIT_CONFIG['client_id'],
                client_secret=REDDIT_CONFIG['client_secret'],
                user_agent=REDDIT_CONFIG['user_agent']
            )
            print("✅ Successfully connected to Reddit API")
        except Exception as e:
            print(f"❌ Failed to connect to Reddit: {e}")
            print("Make sure to update your API credentials in the script!")
    
    def collect_comments_from_subreddit(self, subreddit_name, limit=100, min_length=50):
        """
        Collect comments from a specific subreddit
        
        Args:
            subreddit_name: Name of subreddit (without r/)
            limit: Number of posts to check
            min_length: Minimum character length for comments
        """
        comments_data = []
        
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            print(f"🔍 Collecting comments from r/{subreddit_name}...")
            
            # Get hot posts from the subreddit
            for post in subreddit.hot(limit=limit):
                print(f"📄 Processing post: {post.title[:50]}...")
                
                # Expand all comments (this might take time for large threads)
                post.comments.replace_more(limit=5)
                
                # Extract comments
                for comment in post.comments.list():
                    if len(comment.body) >= min_length and comment.body != '[deleted]':
                        comments_data.append({
                            'comment_id': comment.id,
                            'subreddit': subreddit_name,
                            'post_title': post.title,
                            'post_url': f"https://reddit.com{post.permalink}",
                            'comment_url': f"https://reddit.com{comment.permalink}",
                            'comment_text': comment.body,
                            'comment_score': comment.score,
                            'author': str(comment.author) if comment.author else '[deleted]',
                            'created_utc': datetime.fromtimestamp(comment.created_utc).isoformat(),
                            'collected_at': datetime.now().isoformat()
                        })
                
                # Add small delay to be respectful to Reddit's servers
                time.sleep(0.5)
            
            print(f"✅ Collected {len(comments_data)} comments from r/{subreddit_name}")
            return comments_data
            
        except Exception as e:
            print(f"❌ Error collecting from r/{subreddit_name}: {e}")
            return []
    
    def collect_from_multiple_subreddits(self, subreddits, limit_per_sub=50):
        """Collect comments from multiple subreddits"""
        all_comments = []
        
        for subreddit in subreddits:
            print(f"\n{'='*50}")
            comments = self.collect_comments_from_subreddit(subreddit, limit_per_sub)
            all_comments.extend(comments)
            
            # Add delay between subreddits
            time.sleep(2)
        
        return all_comments
    
    def save_comments(self, comments_data, filename=None):
        """Save collected comments to CSV file"""
        if not comments_data:
            print("❌ No comments to save!")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reddit_comments_{timestamp}.csv"
        
        df = pd.DataFrame(comments_data)
        df.to_csv(filename, index=False)
        print(f"💾 Saved {len(comments_data)} comments to {filename}")
        
        # Show basic stats
        print(f"\n📊 Collection Summary:")
        print(f"Total comments: {len(df)}")
        print(f"Average length: {df['comment_text'].str.len().mean():.0f} characters")
        print(f"Subreddits: {df['subreddit'].value_counts().to_dict()}")
        
        return filename

def main():
    """Main function to run the collector"""
    collector = RedditCommentCollector()
    
    # Define subreddits known for debates and discussions
    # These tend to have comments with logical fallacies
    target_subreddits = [
        'changemyview',      # Structured debates
        'unpopularopinion',  # Controversial opinions
        'politics',          # Political discussions (can be heated)
        'AmItheAsshole',     # Moral judgments
        'relationship_advice', # Personal advice with reasoning
        'conspiracy',        # Often contains flawed logic
        'AskReddit'          # Popular discussions
    ]
    
    print("🚀 Starting Reddit comment collection...")
    print("Target subreddits:", target_subreddits)
    
    # Collect comments
    comments = collector.collect_from_multiple_subreddits(
        subreddits=target_subreddits[:3],  # Start with first 3 subreddits
        limit_per_sub=30  # 30 posts per subreddit
    )
    
    # Save to file
    if comments:
        filename = collector.save_comments(comments)
        print(f"\n🎉 Collection complete! Check your {filename}")
        print("\nNext steps:")
        print("1. Review the collected comments")
        print("2. Filter for quality comments")
        print("3. Start manual annotation for logical fallacies")
    else:
        print("❌ No comments collected. Check your API credentials!")

# Example usage
if __name__ == "__main__":
    print("Reddit Comment Collector for Logical Fallacy Dataset")
    print("="*50)
    
    # Check if API credentials are set
    if REDDIT_CONFIG['client_id'] == 'YOUR_CLIENT_ID':
        print("⚠️  SETUP REQUIRED:")
        print("1. Go to https://www.reddit.com/prefs/apps")
        print("2. Create a new app (choose 'script')")
        print("3. Update REDDIT_CONFIG in this script with your credentials")
        print("4. Install required packages: pip install praw pandas")
    else:
        main()
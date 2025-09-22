import praw
import pandas as pd
import datetime
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def setup_reddit_api():
    """
    Set up and authenticate with the Reddit API using PRAW
    """
    reddit = praw.Reddit(
        client_id=os.getenv('REDDIT_CLIENT_ID'),
        client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
        user_agent=os.getenv('REDDIT_USER_AGENT', 'Comment Extraction Script v1.0 by YourUsername'),
    )
    return reddit

def get_comments_from_post(reddit, post_url, limit=None):
    """
    Extract comments from a specific Reddit post
    
    Args:
        reddit: Authenticated PRAW Reddit instance
        post_url: URL of the Reddit post
        limit: Maximum number of comments to retrieve (None for all)
        
    Returns:
        DataFrame containing comment data
    """
    # Get the submission object from the URL
    submission = reddit.submission(url=post_url)
    
    # Replace "More Comments" links with actual comments
    submission.comments.replace_more(limit=0)  # Set to None to get all comments
    
    comments_data = []
    
    # Get all comments (or up to the limit)
    comments = submission.comments.list()
    if limit:
        comments = comments[:min(limit, len(comments))]
    
    for comment in comments:
        # Extract comment data
        comment_data = {
            'comment_id': comment.id,
            'author': str(comment.author) if comment.author else '[deleted]',
            'body': comment.body,
            'score': comment.score,
            'created_utc': datetime.datetime.fromtimestamp(comment.created_utc),
            'permalink': f"https://www.reddit.com{comment.permalink}",
            'parent_id': comment.parent_id,
            'is_submitter': comment.is_submitter,
            'depth': comment.depth
        }
        comments_data.append(comment_data)
    
    # Convert to DataFrame
    df = pd.DataFrame(comments_data)
    return df

def save_comments_to_csv(df, filename='reddit_comments.csv'):
    """
    Save comments DataFrame to CSV file
    """
    df.to_csv(filename, index=False, encoding='utf-8')
    print(f"Saved {len(df)} comments to {filename}")

def main():
    # Example usage
    post_url = input("Enter the Reddit post URL: ")
    comment_limit = input("Enter maximum number of comments to retrieve (press Enter for all): ")
    
    # Convert to int or None
    if comment_limit.strip():
        try:
            comment_limit = int(comment_limit)
        except ValueError:
            print("Invalid limit, using default (all comments)")
            comment_limit = None
    else:
        comment_limit = None
    
    # Set up Reddit API
    reddit = setup_reddit_api()
    
    # Get comments
    try:
        print(f"Fetching comments from: {post_url}")
        comments_df = get_comments_from_post(reddit, post_url, limit=comment_limit)
        
        # Display some stats
        print(f"Retrieved {len(comments_df)} comments")
        
        # Save to CSV
        output_filename = f"reddit_comments_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        save_comments_to_csv(comments_df, output_filename)
        
        # Show a sample
        print("\nSample comments:")
        sample_df = comments_df[['author', 'body', 'score', 'created_utc']].head(5)
        print(sample_df)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
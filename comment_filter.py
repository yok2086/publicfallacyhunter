import pandas as pd
import re
from collections import Counter

class CommentFilter:
    def __init__(self, csv_file):
        """Load the CSV file with Reddit comments"""
        try:
            self.df = pd.read_csv(csv_file)
            print(f"✅ Loaded {len(self.df)} comments from {csv_file}")
            self.show_basic_stats()
        except Exception as e:
            print(f"❌ Error loading file: {e}")
            return
    
    def show_basic_stats(self):
        """Show basic statistics about the collected comments"""
        print(f"\n📊 Dataset Overview:")
        print(f"Total comments: {len(self.df)}")
        print(f"Average length: {self.df['comment_text'].str.len().mean():.0f} characters")
        print(f"Subreddits: {list(self.df['subreddit'].value_counts().head())}")
        print(f"Date range: {self.df['created_utc'].min()} to {self.df['created_utc'].max()}")
    
    def filter_quality_comments(self, min_length=100, min_score=1, max_length=2000):
        """
        Filter comments based on quality criteria
        
        Args:
            min_length: Minimum character length
            min_score: Minimum Reddit score (upvotes - downvotes)
            max_length: Maximum length (to avoid very long posts)
        """
        print(f"\n🔍 Filtering comments...")
        print(f"Criteria: {min_length}-{max_length} chars, score >= {min_score}")
        
        # Start with original dataset
        filtered = self.df.copy()
        print(f"Starting with: {len(filtered)} comments")
        
        # Filter by length
        filtered = filtered[
            (filtered['comment_text'].str.len() >= min_length) & 
            (filtered['comment_text'].str.len() <= max_length)
        ]
        print(f"After length filter: {len(filtered)} comments")
        
        # Filter by score (popularity)
        filtered = filtered[filtered['comment_score'] >= min_score]
        print(f"After score filter: {len(filtered)} comments")
        
        # Remove deleted/removed comments
        filtered = filtered[
            (~filtered['comment_text'].str.contains('\[deleted\]', na=False)) &
            (~filtered['comment_text'].str.contains('\[removed\]', na=False))
        ]
        print(f"After cleanup: {len(filtered)} comments")
        
        # Look for argumentative language (indicates potential fallacies)
        argumentative_keywords = [
            'because', 'therefore', 'however', 'but', 'although', 'since', 
            'obviously', 'clearly', 'everyone knows', 'always', 'never',
            'you\'re wrong', 'that\'s stupid', 'you people', 'typical'
        ]
        
        # Score comments based on argumentative content
        def argument_score(text):
            text_lower = text.lower()
            return sum(1 for keyword in argumentative_keywords if keyword in text_lower)
        
        filtered['argument_score'] = filtered['comment_text'].apply(argument_score)
        
        # Prioritize comments with argumentative language
        filtered = filtered.sort_values(['argument_score', 'comment_score'], ascending=[False, False])
        
        self.filtered_df = filtered
        print(f"✅ Filtered to {len(filtered)} quality comments")
        return filtered
    
    def find_potential_fallacies(self):
        """Look for comments that might contain specific fallacies"""
        if not hasattr(self, 'filtered_df'):
            print("❌ Run filter_quality_comments() first!")
            return
        
        fallacy_patterns = {
            'Ad hominem': [
                r'you\'re (stupid|dumb|idiot)', r'typical (liberal|conservative)', 
                r'you people', r'coming from someone who'
            ],
            'Straw man': [
                r'so you\'re saying', r'what you\'re really saying', 
                r'you basically want', r'your logic means'
            ],
            'Hasty generalization': [
                r'all (women|men|liberals|conservatives)', r'every single', 
                r'always does', r'never fails'
            ],
            'Appeal to authority': [
                r'experts say', r'studies show', r'scientists agree', 
                r'the media says', r'everyone knows'
            ],
            'False dichotomy': [
                r'either .+ or', r'only two options', r'you\'re either', 
                r'if not .+ then'
            ],
            'Bandwagon': [
                r'everyone (thinks|believes|knows)', r'most people', 
                r'the majority', r'popular opinion'
            ]
        }
        
        print(f"\n🎯 Scanning for potential fallacy patterns...")
        
        fallacy_matches = []
        
        for fallacy_name, patterns in fallacy_patterns.items():
            matches = 0
            for _, comment in self.filtered_df.iterrows():
                text = comment['comment_text'].lower()
                if any(re.search(pattern, text) for pattern in patterns):
                    matches += 1
                    if matches <= 3:  # Show first 3 examples
                        fallacy_matches.append({
                            'fallacy': fallacy_name,
                            'comment': comment['comment_text'][:200] + '...',
                            'score': comment['comment_score'],
                            'url': comment['comment_url']
                        })
            
            if matches > 0:
                print(f"{fallacy_name}: {matches} potential matches")
        
        return fallacy_matches
    
    def preview_comments(self, n=10):
        """Show a sample of filtered comments for manual review"""
        if not hasattr(self, 'filtered_df'):
            print("❌ Run filter_quality_comments() first!")
            return
        
        print(f"\n👀 Preview of top {n} comments for annotation:")
        print("="*80)
        
        for i, (_, comment) in enumerate(self.filtered_df.head(n).iterrows()):
            print(f"\n[{i+1}] Score: {comment['comment_score']} | r/{comment['subreddit']}")
            print(f"Text: {comment['comment_text'][:300]}...")
            print(f"URL: {comment['comment_url']}")
            print("-" * 60)
    
    def save_filtered_comments(self, filename=None):
        """Save filtered comments to a new CSV"""
        if not hasattr(self, 'filtered_df'):
            print("❌ Run filter_quality_comments() first!")
            return
        
        if filename is None:
            filename = "filtered_comments_for_annotation.csv"
        
        self.filtered_df.to_csv(filename, index=False)
        print(f"💾 Saved {len(self.filtered_df)} filtered comments to {filename}")
        return filename

def main():
    print("Reddit Comment Filter for Fallacy Detection")
    print("="*50)
    
    # Get CSV filename from user
    csv_file = input("Enter your CSV filename (e.g., reddit_comments_20241124_123456.csv): ")
    
    # Load and filter comments
    filter_tool = CommentFilter(csv_file)
    
    if hasattr(filter_tool, 'df'):
        # Filter for quality
        filtered = filter_tool.filter_quality_comments(
            min_length=150,  # Longer comments more likely to have arguments
            min_score=2,     # At least somewhat popular
            max_length=1500  # Not too long
        )
        
        # Look for potential fallacies
        potential_fallacies = filter_tool.find_potential_fallacies()
        
        # Preview comments
        filter_tool.preview_comments(5)
        
        # Save filtered results
        output_file = filter_tool.save_filtered_comments()
        
        print(f"\n🎉 Next Steps:")
        print(f"1. Review {output_file} in Excel")
        print(f"2. Pick 50-100 interesting comments to start with")
        print(f"3. Begin manual annotation for logical fallacies")
        print(f"4. Look for the fallacy patterns we identified above")

if __name__ == "__main__":
    main()
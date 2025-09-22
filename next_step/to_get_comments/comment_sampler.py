import pandas as pd
import numpy as np
import re
from collections import defaultdict, Counter
import random

class SmartAnnotationSampler:
    def __init__(self, csv_file):
        """Load filtered comments CSV"""
        try:
            self.df = pd.read_csv(csv_file)
            print(f"✅ Loaded {len(self.df)} comments for sampling")
        except Exception as e:
            print(f"❌ Error loading file: {e}")
            return
    
    def create_fallacy_patterns(self):
        """Define patterns to identify potential fallacies"""
        self.fallacy_patterns = {
            'Ad hominem': [
                r'you\'re (stupid|dumb|idiot|moron)', r'typical (liberal|conservative|democrat|republican)',
                r'you people', r'coming from someone who', r'says the (guy|person) who',
                r'what do you expect from', r'look who\'s talking'
            ],
            
            'Straw man': [
                r'so you\'re saying', r'what you\'re really saying', r'you basically want',
                r'your logic means', r'by that logic', r'so according to you'
            ],
            
            'Red herring': [
                r'but what about', r'speaking of which', r'that reminds me',
                r'on a related note', r'while we\'re on the topic'
            ],
            
            'Hasty generalization': [
                r'all (women|men|liberals|conservatives|millennials|boomers)', 
                r'every single', r'always does', r'never fails', r'every time',
                r'(women|men|people) always', r'typical (woman|man|person)'
            ],
            
            'Appeal to authority': [
                r'experts say', r'studies show', r'scientists agree', r'research proves',
                r'the media says', r'doctors recommend', r'professionals know'
            ],
            
            'Bandwagon fallacy': [
                r'everyone (thinks|believes|knows)', r'most people', r'the majority',
                r'popular opinion', r'common knowledge', r'everybody agrees'
            ],
            
            'Appeal to ignorance': [
                r'no one has proven', r'can\'t prove', r'there\'s no evidence',
                r'show me proof', r'until you can prove', r'absence of evidence'
            ],
            
            'Circular argument': [
                r'because it is', r'that\'s just how it is', r'it\'s true because',
                r'obviously true', r'self-evident', r'by definition'
            ],
            
            'Appeal to pity': [
                r'feel sorry', r'think of the children', r'poor (guy|person|people)',
                r'have sympathy', r'it\'s sad that', r'heartbreaking'
            ],
            
            'Causal fallacy': [
                r'caused by', r'because of', r'due to', r'results from',
                r'leads to', r'correlation', r'linked to'
            ],
            
            'Appeal to hypocrisy': [
                r'you also', r'you do the same', r'hypocrite', r'pot calling kettle',
                r'look at yourself', r'you\'re guilty of'
            ],
            
            'Equivocation': [
                r'depends on what you mean', r'define', r'interpretation',
                r'could mean', r'different ways'
            ]
        }
    
    def score_comments_for_fallacies(self):
        """Score each comment for likelihood of containing each fallacy"""
        self.create_fallacy_patterns()
        
        print("🔍 Scoring comments for fallacy patterns...")
        
        # Create fallacy score columns
        for fallacy in self.fallacy_patterns.keys():
            self.df[f'{fallacy}_score'] = 0
        
        # Score each comment
        for idx, row in self.df.iterrows():
            text_lower = row['comment_text'].lower()
            
            for fallacy, patterns in self.fallacy_patterns.items():
                score = sum(1 for pattern in patterns if re.search(pattern, text_lower))
                self.df.at[idx, f'{fallacy}_score'] = score
        
        # Calculate total fallacy score
        fallacy_cols = [f'{fallacy}_score' for fallacy in self.fallacy_patterns.keys()]
        self.df['total_fallacy_score'] = self.df[fallacy_cols].sum(axis=1)
        
        print("✅ Fallacy scoring complete!")
    
    def sample_for_annotation(self, target_per_fallacy=30, no_fallacy_samples=100):
        """
        Sample comments strategically for annotation
        
        Args:
            target_per_fallacy: Target number of samples per fallacy type
            no_fallacy_samples: Number of comments with no apparent fallacies
        """
        if 'total_fallacy_score' not in self.df.columns:
            self.score_comments_for_fallacies()
        
        sampled_comments = []
        sampled_indices = set()
        
        print(f"🎯 Sampling comments for annotation...")
        
        # Sample high-scoring comments for each fallacy
        for fallacy in self.fallacy_patterns.keys():
            fallacy_col = f'{fallacy}_score'
            
            # Get comments with high scores for this fallacy
            candidates = self.df[
                (self.df[fallacy_col] > 0) & 
                (~self.df.index.isin(sampled_indices))
            ].copy()
            
            if len(candidates) == 0:
                print(f"⚠️  No candidates found for {fallacy}")
                continue
            
            # Sort by fallacy score and comment score
            candidates = candidates.sort_values([fallacy_col, 'comment_score'], ascending=[False, False])
            
            # Sample top candidates
            n_sample = min(target_per_fallacy, len(candidates))
            selected = candidates.head(n_sample)
            
            for idx, row in selected.iterrows():
                sampled_comments.append({
                    'comment_id': row['comment_id'],
                    'comment_text': row['comment_text'],
                    'comment_url': row['comment_url'],
                    'subreddit': row['subreddit'],
                    'comment_score': row['comment_score'],
                    'suspected_fallacy': fallacy,
                    'fallacy_score': row[fallacy_col],
                    'sampling_reason': f'High {fallacy} score'
                })
                sampled_indices.add(idx)
            
            print(f"{fallacy}: {n_sample} samples")
        
        # Sample comments with no apparent fallacies
        no_fallacy_candidates = self.df[
            (self.df['total_fallacy_score'] == 0) & 
            (~self.df.index.isin(sampled_indices))
        ].copy()
        
        if len(no_fallacy_candidates) > 0:
            # Sort by comment quality (score, length)
            no_fallacy_candidates['length'] = no_fallacy_candidates['comment_text'].str.len()
            no_fallacy_candidates = no_fallacy_candidates.sort_values(['comment_score', 'length'], ascending=[False, False])
            
            n_sample = min(no_fallacy_samples, len(no_fallacy_candidates))
            selected = no_fallacy_candidates.head(n_sample)
            
            for idx, row in selected.iterrows():
                sampled_comments.append({
                    'comment_id': row['comment_id'],
                    'comment_text': row['comment_text'],
                    'comment_url': row['comment_url'],
                    'subreddit': row['subreddit'],
                    'comment_score': row['comment_score'],
                    'suspected_fallacy': 'None',
                    'fallacy_score': 0,
                    'sampling_reason': 'No fallacy patterns detected'
                })
            
            print(f"No fallacy: {n_sample} samples")
        
        self.annotation_sample = pd.DataFrame(sampled_comments)
        print(f"\n✅ Total sampled: {len(sampled_comments)} comments")
        return self.annotation_sample
    
    def preview_samples(self, n=5):
        """Preview sampled comments"""
        if not hasattr(self, 'annotation_sample'):
            print("❌ Run sample_for_annotation() first!")
            return
        
        print(f"\n👀 Preview of annotation samples:")
        print("="*80)
        
        for fallacy in ['Ad hominem', 'Straw man', 'None']:
            fallacy_samples = self.annotation_sample[
                self.annotation_sample['suspected_fallacy'] == fallacy
            ].head(2)
            
            if len(fallacy_samples) > 0:
                print(f"\n🎯 {fallacy} examples:")
                for _, row in fallacy_samples.iterrows():
                    print(f"Score: {row['comment_score']} | r/{row['subreddit']}")
                    print(f"Text: {row['comment_text'][:200]}...")
                    print(f"URL: {row['comment_url']}")
                    print("-" * 40)
    
    def save_annotation_batch(self, filename="annotation_batch.csv"):
        """Save the sampled comments for annotation"""
        if not hasattr(self, 'annotation_sample'):
            print("❌ Run sample_for_annotation() first!")
            return
        
        self.annotation_sample.to_csv(filename, index=False)
        
        print(f"💾 Saved {len(self.annotation_sample)} comments to {filename}")
        print(f"\n📊 Sampling Summary:")
        
        fallacy_counts = self.annotation_sample['suspected_fallacy'].value_counts()
        for fallacy, count in fallacy_counts.items():
            print(f"{fallacy}: {count} samples")
        
        return filename
    
    def create_annotation_instructions(self):
        """Generate annotation guidelines"""
        instructions = """
ANNOTATION INSTRUCTIONS
======================

For each comment, mark ALL fallacies present (can be multiple):

1. Ad hominem - Attacking the person instead of their argument
2. Red herring - Introducing irrelevant information to distract  
3. Straw man - Misrepresenting someone's position to attack it
4. Equivocation - Using ambiguous language to mislead
5. Hasty generalization - Drawing broad conclusions from limited examples
6. Bandwagon fallacy - Arguing something is true because many believe it
7. Appeal to ignorance - Claiming something is true because it hasn't been proven false
8. Circular argument - Using the conclusion as evidence for itself
9. Sunk cost fallacy - Continuing something because of past investment
10. Appeal to pity - Using emotion rather than logic to persuade
11. Causal fallacy - Incorrectly assuming causation from correlation
12. Appeal to hypocrisy - Dismissing argument by pointing out inconsistency

TIPS:
- One comment can have multiple fallacies
- Focus on the main argument, not side comments
- When unsure, mark as "No fallacy"
- Quality over quantity - accurate labels are crucial
        """
        
        with open("annotation_instructions.txt", "w") as f:
            f.write(instructions)
        
        print("📋 Created annotation_instructions.txt")

def main():
    print("Smart Comment Sampler for Fallacy Annotation")
    print("="*50)
    
    # Use the filtered CSV from previous step
    csv_file = input("Enter your filtered CSV filename (e.g., filtered_comments_for_annotation.csv): ")
    
    sampler = SmartAnnotationSampler(csv_file)
    
    if hasattr(sampler, 'df'):
        # Sample comments strategically
        samples = sampler.sample_for_annotation(
            target_per_fallacy=25,  # 25 per fallacy = 300 fallacy examples
            no_fallacy_samples=100  # 100 clean examples
        )
        
        # Preview the samples
        sampler.preview_samples()
        
        # Save for annotation
        output_file = sampler.save_annotation_batch()
        
        # Create instructions
        sampler.create_annotation_instructions()
        
        print(f"\n🎉 Ready for annotation!")
        print(f"1. Open {output_file} in Excel or annotation tool")
        print(f"2. Follow annotation_instructions.txt")
        print(f"3. Start with obvious examples first")
        print(f"4. Target 300-400 high-quality annotations")

if __name__ == "__main__":
    main()
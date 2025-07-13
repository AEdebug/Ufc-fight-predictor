import pandas as pd
import numpy as np
import os

def load_processed_data():
    """Load the preprocessed data"""
    fighters_df = pd.read_csv("data/processed_fighters.csv")
    fights_df = pd.read_csv("data/processed_fights.csv")
    return fighters_df, fights_df

def calculate_fighter_stats(fights_df, fighters_df):
    """Calculate additional statistics for each fighter based on their fight history"""
    print("Calculating fighter statistics...")
    
    # Create a copy of the fighters dataframe to add new features
    enhanced_fighters = fighters_df.copy()
    
    # Ensure image_path is carried over if it exists
    if 'image_path' in fighters_df.columns:
        enhanced_fighters['image_path'] = fighters_df['image_path']

    # Initialize new columns for fighter statistics
    enhanced_fighters['total_fights'] = 0
    enhanced_fighters['wins'] = 0
    enhanced_fighters['losses'] = 0
    enhanced_fighters['ko_wins'] = 0
    enhanced_fighters['submission_wins'] = 0
    enhanced_fighters['decision_wins'] = 0
    
    # Calculate statistics for each fighter
    for _, fight in fights_df.iterrows():
        # Update fighter 1 stats
        fighter1_id = fight['fighter1_id']
        if fighter1_id in enhanced_fighters['fighter_id'].values:
            idx1 = enhanced_fighters.index[enhanced_fighters['fighter_id'] == fighter1_id][0]
            enhanced_fighters.loc[idx1, 'total_fights'] += 1
            if fight['fighter1_won'] == 1:
                enhanced_fighters.loc[idx1, 'wins'] += 1
                if fight['win_method'] == 'knockout':
                    enhanced_fighters.loc[idx1, 'ko_wins'] += 1
                elif fight['win_method'] == 'submission':
                    enhanced_fighters.loc[idx1, 'submission_wins'] += 1
                elif fight['win_method'] == 'decision':
                    enhanced_fighters.loc[idx1, 'decision_wins'] += 1
            else:
                enhanced_fighters.loc[idx1, 'losses'] += 1
        
        # Update fighter 2 stats
        fighter2_id = fight['fighter2_id']
        if fighter2_id in enhanced_fighters['fighter_id'].values:
            idx2 = enhanced_fighters.index[enhanced_fighters['fighter_id'] == fighter2_id][0]
            enhanced_fighters.loc[idx2, 'total_fights'] += 1
            if fight['fighter1_won'] == 0:
                enhanced_fighters.loc[idx2, 'wins'] += 1
                if fight['win_method'] == 'knockout':
                    enhanced_fighters.loc[idx2, 'ko_wins'] += 1
                elif fight['win_method'] == 'submission':
                    enhanced_fighters.loc[idx2, 'submission_wins'] += 1
                elif fight['win_method'] == 'decision':
                    enhanced_fighters.loc[idx2, 'decision_wins'] += 1
            else:
                enhanced_fighters.loc[idx2, 'losses'] += 1
    
    # Calculate derived statistics
    enhanced_fighters['win_pct'] = enhanced_fighters['wins'] / enhanced_fighters['total_fights'].replace(0, 1)
    enhanced_fighters['ko_rate'] = enhanced_fighters['ko_wins'] / enhanced_fighters['wins'].replace(0, 1)
    enhanced_fighters['sub_rate'] = enhanced_fighters['submission_wins'] / enhanced_fighters['wins'].replace(0, 1)
    enhanced_fighters['dec_rate'] = enhanced_fighters['decision_wins'] / enhanced_fighters['wins'].replace(0, 1)
    
    return enhanced_fighters

def create_fight_pairs(fights_df, enhanced_fighters):
    """Create feature pairs for each fight by combining fighter statistics"""
    print("Creating fight pair features...")
    
    # Initialize list to store fight pairs
    fight_pairs = []
    
    # Process each fight
    for _, fight in fights_df.iterrows():
        fighter1_id = fight['fighter1_id']
        fighter2_id = fight['fighter2_id']
        
        # Get fighter data
        fighter1 = enhanced_fighters[enhanced_fighters['fighter_id'] == fighter1_id].iloc[0]
        fighter2 = enhanced_fighters[enhanced_fighters['fighter_id'] == fighter2_id].iloc[0]
        
        # Create feature pair
        pair = {
            'fight_id': fight['fight_id'],
            'fighter1_id': fighter1_id,
            'fighter2_id': fighter2_id,
            'fighter1_name': fighter1['name'],
            'fighter2_name': fighter2['name'],
            
            # Physical attributes
            'height_diff': fighter1['height'] - fighter2['height'],
            'weight_diff': fighter1['weight'] - fighter2['weight'],
            'reach_diff': fighter1['reach'] - fighter2['reach'],
            'age_diff': fighter1['age'] - fighter2['age'],
            
            # Performance metrics
            'win_pct_diff': fighter1['win_pct'] - fighter2['win_pct'],
            'ko_rate_diff': fighter1['ko_rate'] - fighter2['ko_rate'],
            'sub_rate_diff': fighter1['sub_rate'] - fighter2['sub_rate'],
            'dec_rate_diff': fighter1['dec_rate'] - fighter2['dec_rate'],
            'experience_diff': fighter1['total_fights'] - fighter2['total_fights'],
            
            # Target variables
            'fighter1_won': fight['fighter1_won'],
            'win_method': fight['win_method'],
            'win_round': fight['win_round']
        }
        
        # Add stance information (one-hot encoded)
        stance_cols = [col for col in fighter1.index if col.startswith('stance_')]
        for col in stance_cols:
            pair[f'fighter1_{col}'] = fighter1[col]
            pair[f'fighter2_{col}'] = fighter2[col]
        
        # Add weight class information (one-hot encoded)
        weight_class_cols = [col for col in fighter1.index if col.startswith('weight_class_')]
        for col in weight_class_cols:
            pair[f'fighter1_{col}'] = fighter1[col]
            pair[f'fighter2_{col}'] = fighter2[col]
        
        fight_pairs.append(pair)
    
    # Convert to DataFrame
    fight_pairs_df = pd.DataFrame(fight_pairs)
    
    return fight_pairs_df

if __name__ == "__main__":
    print("Starting feature engineering...")
    
    # Load preprocessed data
    fighters_df, fights_df = load_processed_data()
    
    # Calculate fighter statistics
    enhanced_fighters = calculate_fighter_stats(fights_df, fighters_df)
    
    # Create fight pairs with features
    fight_pairs_df = create_fight_pairs(fights_df, enhanced_fighters)
    
    # Save enhanced data
    enhanced_fighters.to_csv("data/enhanced_fighters.csv", index=False)
    fight_pairs_df.to_csv("data/fight_pairs.csv", index=False)
    
    print(f"Feature engineering complete! Saved enhanced data for {len(enhanced_fighters)} fighters and {len(fight_pairs_df)} fight pairs.")
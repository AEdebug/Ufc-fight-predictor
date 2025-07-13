import pandas as pd
import numpy as np
import os

def load_data():
    """Load the raw data from CSV files"""
    fighters_df = pd.read_csv("data/fighters.csv")
    fights_df = pd.read_csv("data/fights.csv")
    return fighters_df, fights_df

def preprocess_fighters(fighters_df):
    """Preprocess the fighters data"""
    print("Preprocessing fighter data...")
    
    # Create a copy to avoid modifying the original
    processed_df = fighters_df.copy()
    
    # Handle missing values (if any)
    for col in processed_df.columns:
        if processed_df[col].dtype in [np.float64, np.int64]:
            processed_df[col].fillna(processed_df[col].median(), inplace=True)
        else:
            processed_df[col].fillna('unknown', inplace=True)
    
    # Convert categorical variables to one-hot encoding
    stance_dummies = pd.get_dummies(processed_df['stance'], prefix='stance')
    weight_class_dummies = pd.get_dummies(processed_df['weight_class'], prefix='weight_class')
    
    # Drop original categorical columns and join the dummy variables
    # Ensure 'image_path' is not dropped if it exists
    columns_to_drop = [col for col in ['stance', 'weight_class'] if col in processed_df.columns]
    processed_df = processed_df.drop(columns=columns_to_drop, axis=1)
    processed_df = pd.concat([processed_df, stance_dummies, weight_class_dummies], axis=1)
    
    return processed_df

def preprocess_fights(fights_df, fighters_df):
    """Preprocess the fights data"""
    print("Preprocessing fight data...")
    
    # Create a copy to avoid modifying the original
    processed_df = fights_df.copy()
    
    # Add fighter names for easier reference
    fighter_names = {row['fighter_id']: row['name'] for _, row in fighters_df.iterrows()}
    processed_df['fighter1_name'] = processed_df['fighter1_id'].map(fighter_names)
    processed_df['fighter2_name'] = processed_df['fighter2_id'].map(fighter_names)
    processed_df['winner_name'] = processed_df['winner_id'].map(fighter_names)
    
    # Create binary outcome (1 if fighter1 won, 0 if fighter2 won)
    processed_df['fighter1_won'] = (processed_df['winner_id'] == processed_df['fighter1_id']).astype(int)
    
    # Convert win method to categorical
    method_dummies = pd.get_dummies(processed_df['win_method'], prefix='win_method')
    processed_df = pd.concat([processed_df, method_dummies], axis=1)
    
    return processed_df

if __name__ == "__main__":
    print("Starting data preprocessing...")
    
    # Load data
    fighters_df, fights_df = load_data()
    
    # Preprocess data
    processed_fighters = preprocess_fighters(fighters_df)
    processed_fights = preprocess_fights(fights_df, fighters_df)
    
    # Save processed data
    processed_fighters.to_csv("data/processed_fighters.csv", index=False)
    processed_fights.to_csv("data/processed_fights.csv", index=False)
    
    print(f"Preprocessing complete! Saved processed data for {len(processed_fighters)} fighters and {len(processed_fights)} fights.")
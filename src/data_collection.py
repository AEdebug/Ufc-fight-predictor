import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import os
import json

# Create necessary directories
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("src\\templates", exist_ok=True)  # Use Windows path separator

def fetch_ufc_data():
    """
    Fetch UFC data from a public dataset or API
    For this implementation, we'll use a simplified approach with sample data
    """
    # In a real implementation, you would scrape or use an API
    # For now, we'll create sample data
    
    # Sample fighter data
    fighters = [
        {"fighter_id": 1, "name": "Conor McGregor", "height": 1.75, "weight": 70.3, "reach": 1.88, "stance": "southpaw", "age": 33, "weight_class": "lightweight", "image_path": "static/images/pngimg.com - conor_mcgregor_PNG54.png"},
        {"fighter_id": 2, "name": "Khabib Nurmagomedov", "height": 1.78, "weight": 70.3, "reach": 1.78, "stance": "orthodox", "age": 33, "weight_class": "lightweight", "image_path": "static/images/default_fighter.jpg"},
        {"fighter_id": 3, "name": "Jon Jones", "height": 1.93, "weight": 93.0, "reach": 2.15, "stance": "orthodox", "age": 34, "weight_class": "light_heavyweight", "image_path": "static/images/2335639.png"},
        {"fighter_id": 4, "name": "Israel Adesanya", "height": 1.93, "weight": 84.0, "reach": 2.03, "stance": "orthodox", "age": 32, "weight_class": "middleweight", "image_path": "static/images/default_fighter.jpg"},
        {"fighter_id": 5, "name": "Francis Ngannou", "height": 1.93, "weight": 117.0, "reach": 2.03, "stance": "orthodox", "age": 35, "weight_class": "heavyweight", "image_path": "static/images/default_fighter.jpg"},
        {"fighter_id": 6, "name": "Dustin Poirier", "height": 1.75, "weight": 70.3, "reach": 1.83, "stance": "southpaw", "age": 33, "weight_class": "lightweight", "image_path": "static/images/default_fighter.jpg"},
        {"fighter_id": 7, "name": "Charles Oliveira", "height": 1.78, "weight": 70.3, "reach": 1.85, "stance": "orthodox", "age": 32, "weight_class": "lightweight", "image_path": "static/images/default_fighter.jpg"},
        {"fighter_id": 8, "name": "Kamaru Usman", "height": 1.83, "weight": 77.1, "reach": 1.93, "stance": "orthodox", "age": 34, "weight_class": "welterweight", "image_path": "static/images/default_fighter.jpg"},
        {"fighter_id": 9, "name": "Alexander Volkanovski", "height": 1.68, "weight": 65.8, "reach": 1.82, "stance": "orthodox", "age": 33, "weight_class": "featherweight", "image_path": "static/images/default_fighter.jpg"},
        {"fighter_id": 10, "name": "Max Holloway", "height": 1.80, "weight": 65.8, "reach": 1.82, "stance": "orthodox", "age": 30, "weight_class": "featherweight", "image_path": "static/images/default_fighter.jpg"},
    ]
    
    # Sample fight data
    fights = [
        {"fight_id": 1, "fighter1_id": 1, "fighter2_id": 2, "winner_id": 2, "win_method": "submission", "win_round": 4},
        {"fight_id": 2, "fighter1_id": 1, "fighter2_id": 6, "winner_id": 6, "win_method": "knockout", "win_round": 2},
        {"fight_id": 3, "fighter1_id": 6, "fighter2_id": 7, "winner_id": 7, "win_method": "submission", "win_round": 3},
        {"fight_id": 4, "fighter1_id": 3, "fighter2_id": 5, "winner_id": 3, "win_method": "decision", "win_round": 5},
        {"fight_id": 5, "fighter1_id": 4, "fighter2_id": 8, "winner_id": 8, "win_method": "decision", "win_round": 5},
        {"fight_id": 6, "fighter1_id": 9, "fighter2_id": 10, "winner_id": 9, "win_method": "decision", "win_round": 5},
        {"fight_id": 7, "fighter1_id": 2, "fighter2_id": 7, "winner_id": 2, "win_method": "submission", "win_round": 2},
        {"fight_id": 8, "fighter1_id": 3, "fighter2_id": 4, "winner_id": 3, "win_method": "knockout", "win_round": 3},
        {"fight_id": 9, "fighter1_id": 5, "fighter2_id": 8, "winner_id": 5, "win_method": "knockout", "win_round": 1},
        {"fight_id": 10, "fighter1_id": 9, "fighter2_id": 6, "winner_id": 9, "win_method": "decision", "win_round": 5},
        {"fight_id": 11, "fighter1_id": 10, "fighter2_id": 6, "winner_id": 10, "win_method": "decision", "win_round": 5},
        {"fight_id": 12, "fighter1_id": 1, "fighter2_id": 7, "winner_id": 7, "win_method": "submission", "win_round": 3},
        {"fight_id": 13, "fighter1_id": 2, "fighter2_id": 6, "winner_id": 2, "win_method": "submission", "win_round": 3},
        {"fight_id": 14, "fighter1_id": 3, "fighter2_id": 8, "winner_id": 3, "win_method": "knockout", "win_round": 2},
        {"fight_id": 15, "fighter1_id": 4, "fighter2_id": 5, "winner_id": 4, "win_method": "decision", "win_round": 5},
    ]
    
    # Convert to DataFrames
    fighters_df = pd.DataFrame(fighters)
    fights_df = pd.DataFrame(fights)
    
    return fighters_df, fights_df

if __name__ == "__main__":
    print("Collecting UFC data...")
    fighters_df, fights_df = fetch_ufc_data()
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Save to CSV
    fighters_df.to_csv("data/fighters.csv", index=False)
    fights_df.to_csv("data/fights.csv", index=False)
    
    print(f"Data collection complete! Saved {len(fighters_df)} fighters and {len(fights_df)} fights.")
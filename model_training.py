import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

def load_fight_pairs():
    """Load the fight pairs data with engineered features"""
    return pd.read_csv("data/fight_pairs.csv")

def prepare_features_and_targets(fight_pairs_df):
    """Prepare feature matrices and target vectors for training"""
    print("Preparing features and targets...")
    
    # Features for winner prediction
    winner_features = fight_pairs_df.drop(['fight_id', 'fighter1_id', 'fighter2_id', 
                                          'fighter1_name', 'fighter2_name', 
                                          'fighter1_won', 'win_method', 'win_round'], axis=1)
    
    # Target for winner prediction
    winner_target = fight_pairs_df['fighter1_won']
    
    # Features for method prediction (only use fights where we know the method)
    method_features = fight_pairs_df.drop(['fight_id', 'fighter1_id', 'fighter2_id', 
                                          'fighter1_name', 'fighter2_name', 
                                          'fighter1_won', 'win_method', 'win_round'], axis=1)
    
    # Target for method prediction
    method_target = fight_pairs_df['win_method']
    
    # Features for round prediction
    round_features = fight_pairs_df.drop(['fight_id', 'fighter1_id', 'fighter2_id', 
                                         'fighter1_name', 'fighter2_name', 
                                         'fighter1_won', 'win_method', 'win_round'], axis=1)
    
    # Target for round prediction
    round_target = fight_pairs_df['win_round']
    
    return winner_features, winner_target, method_features, method_target, round_features, round_target

def train_models(winner_features, winner_target, method_features, method_target, round_features, round_target):
    """Train models for winner, method, and round prediction"""
    print("Training models...")
    
    # Split data into training and testing sets
    X_winner_train, X_winner_test, y_winner_train, y_winner_test = train_test_split(
        winner_features, winner_target, test_size=0.2, random_state=42)
    
    X_method_train, X_method_test, y_method_train, y_method_test = train_test_split(
        method_features, method_target, test_size=0.2, random_state=42)
    
    X_round_train, X_round_test, y_round_train, y_round_test = train_test_split(
        round_features, round_target, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_winner_train_scaled = scaler.fit_transform(X_winner_train)
    X_winner_test_scaled = scaler.transform(X_winner_test)
    
    X_method_train_scaled = scaler.fit_transform(X_method_train)
    X_method_test_scaled = scaler.transform(X_method_test)
    
    X_round_train_scaled = scaler.fit_transform(X_round_train)
    X_round_test_scaled = scaler.transform(X_round_test)
    
    # Train winner prediction model
    winner_model = RandomForestClassifier(n_estimators=100, random_state=42)
    winner_model.fit(X_winner_train_scaled, y_winner_train)
    
    # Train method prediction model
    method_model = RandomForestClassifier(n_estimators=100, random_state=42)
    method_model.fit(X_method_train_scaled, y_method_train)
    
    # Train round prediction model
    round_model = RandomForestClassifier(n_estimators=100, random_state=42)
    round_model.fit(X_round_train_scaled, y_round_train)
    
    # Evaluate models
    winner_pred = winner_model.predict(X_winner_test_scaled)
    method_pred = method_model.predict(X_method_test_scaled)
    round_pred = round_model.predict(X_round_test_scaled)
    
    print("\nWinner Prediction Model:")
    print(f"Accuracy: {accuracy_score(y_winner_test, winner_pred):.4f}")
    print(classification_report(y_winner_test, winner_pred, zero_division=0))
    
    print("\nMethod Prediction Model:")
    print(f"Accuracy: {accuracy_score(y_method_test, method_pred):.4f}")
    print(classification_report(y_method_test, method_pred, zero_division=0))
    
    print("\nRound Prediction Model:")
    print(f"Accuracy: {accuracy_score(y_round_test, round_pred):.4f}")
    print(classification_report(y_round_test, round_pred, zero_division=0))
    
    return winner_model, method_model, round_model, scaler

if __name__ == "__main__":
    print("Starting model training...")
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Load fight pairs data
    fight_pairs_df = load_fight_pairs()
    
    # Prepare features and targets
    winner_features, winner_target, method_features, method_target, round_features, round_target = prepare_features_and_targets(fight_pairs_df)
    
    # Train models
    winner_model, method_model, round_model, scaler = train_models(
        winner_features, winner_target, method_features, method_target, round_features, round_target)
    
    # Save models
    joblib.dump(winner_model, "models/winner_model.pkl")
    joblib.dump(method_model, "models/method_model.pkl")
    joblib.dump(round_model, "models/round_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    
    print("\nModel training complete! Saved models to the 'models' directory.")
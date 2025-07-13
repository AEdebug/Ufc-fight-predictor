import pandas as pd
import numpy as np
import joblib
import os
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class UFCFightPredictor:
    def __init__(self):
        """Initialize the UFC Fight Predictor"""
        try:
            # Get absolute paths for models
            base_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.join(base_dir, 'models')
            data_dir = os.path.join(base_dir, 'data')
            
            # Load models
            self.winner_model = joblib.load(os.path.join(models_dir, "winner_model.pkl"))
            self.method_model = joblib.load(os.path.join(models_dir, "method_model.pkl"))
            self.round_model = joblib.load(os.path.join(models_dir, "round_model.pkl"))
            self.scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
            
            # Load fighter data
            self.fighters_df = pd.read_csv(os.path.join(data_dir, "enhanced_fighters.csv"))
            self.weight_class_order = [
                "featherweight", "lightweight", "welterweight", "middleweight", "light_heavyweight", "heavyweight"
            ]
            self.weight_class_map = {wc: i for i, wc in enumerate(self.weight_class_order)}
            
            logger.info("Models and data loaded successfully")
        except FileNotFoundError as e:
            raise Exception(f"File not found: {str(e)}") from e
        except Exception as e:
            raise Exception(f"Error loading models or data: {str(e)}") from e
    
    def get_fighter_names(self):
        """Get list of all fighter names"""
        try:
            return self.fighters_df['name'].tolist()
        except Exception as e:
            print(f"Error getting fighter names: {str(e)}")
            return []
    
    def get_fighter_data(self, fighter_name):
        """Get fighter data by name"""
        try:
            logger.debug(f"Looking up fighter: {fighter_name}")
            fighter = self.fighters_df[self.fighters_df['name'] == fighter_name]
            
            if fighter.empty:
                logger.warning(f"Fighter not found: {fighter_name}")
                return None
                
            fighter_data = fighter.iloc[0].to_dict()
            print(f"DEBUG (fight_predictor): Found fighter data: {fighter_data}") # Changed from logger.debug
            
            # Calculate win percentage
            total_fights = fighter_data.get('total_fights', 0)
            wins = fighter_data.get('wins', 0)
            fighter_data['win_pct'] = (wins / total_fights) if total_fights > 0 else 0
            
            # Calculate rate features
            fighter_data['ko_rate'] = (fighter_data.get('ko_wins', 0) / total_fights) if total_fights > 0 else 0
            fighter_data['sub_rate'] = (fighter_data.get('submission_wins', 0) / total_fights) if total_fights > 0 else 0
            fighter_data['dec_rate'] = (fighter_data.get('decision_wins', 0) / total_fights) if total_fights > 0 else 0
            
            # Convert boolean values to integers for stance features
            fighter_data['stance_orthodox'] = int(fighter_data.get('stance_orthodox', 0))
            fighter_data['stance_southpaw'] = int(fighter_data.get('stance_southpaw', 0))

            # Convert weight class booleans to one-hot encoded format
            for wc in self.weight_class_order:
                fighter_data[f'weight_class_{wc}'] = int(fighter_data.get(f'weight_class_{wc}', 0))
            
            return fighter_data
            
        except Exception as e:
            logger.error(f"Error getting fighter data for {fighter_name}: {str(e)}", exc_info=True)
            return None
    
    def get_weight_class_index(self, fighter):
        # Find the weight class column that is 1 for this fighter
        for wc in self.weight_class_order:
            col = f"weight_class_{wc}"
            if col in fighter and fighter[col] == 1:
                return self.weight_class_map[wc]
        return None

    def get_fighter_names(self):
        """Get list of all fighter names"""
        return self.fighters_df['name'].tolist()

    def predict_fight(self, fighter1_name, fighter2_name):
        try:
            logger.debug(f"Predicting fight between {fighter1_name} and {fighter2_name}")
            
            # Get fighter data
            fighter1 = self.get_fighter_data(fighter1_name)
            fighter2 = self.get_fighter_data(fighter2_name)
            
            if fighter1 is None:
                logger.error(f"Fighter '{fighter1_name}' not found in database")
                return {'error': f"Fighter '{fighter1_name}' not found in database.",
                       'details': f"Available fighters: {self.get_fighter_names()}",
                       'received_data': {'fighter1': fighter1_name, 'fighter2': fighter2_name}}
            
            if fighter2 is None:
                logger.error(f"Fighter '{fighter2_name}' not found in database")
                return {'error': f"Fighter '{fighter2_name}' not found in database.",
                       'details': f"Available fighters: {self.get_fighter_names()}",
                       'received_data': {'fighter1': fighter1_name, 'fighter2': fighter2_name}}
            
            # Log fighter data
            logger.debug(f"Fighter 1 data: {fighter1}")
            logger.debug(f"Fighter 2 data: {fighter2}")
            
            # Create feature array with proper handling of missing values
            features = []
            try:
                # Log raw fighter data
                logger.debug(f"Raw fighter1 data: {fighter1}")
                logger.debug(f"Raw fighter2 data: {fighter2}")
                
                # Add physical difference features
                features.append(float(fighter1.get('height', 0)) - float(fighter2.get('height', 0)))  # height_diff
                features.append(float(fighter1.get('weight', 0)) - float(fighter2.get('weight', 0)))  # weight_diff
                features.append(float(fighter1.get('reach', 0)) - float(fighter2.get('reach', 0)))   # reach_diff
                features.append(float(fighter1.get('age', 0)) - float(fighter2.get('age', 0)))       # age_diff
                features.append(float(fighter1.get('win_pct', 0)) - float(fighter2.get('win_pct', 0)))  # win_pct_diff
                features.append(float(fighter1.get('ko_rate', 0)) - float(fighter2.get('ko_rate', 0)))  # ko_rate_diff
                features.append(float(fighter1.get('sub_rate', 0)) - float(fighter2.get('sub_rate', 0)))  # sub_rate_diff
                features.append(float(fighter1.get('dec_rate', 0)) - float(fighter2.get('dec_rate', 0)))  # dec_rate_diff
                features.append(float(fighter1.get('total_fights', 0)) - float(fighter2.get('total_fights', 0)))  # experience_diff
                
                # Add stance features for each fighter
                features.append(int(fighter1.get('stance_orthodox', 0)))  # fighter1_stance_orthodox
                features.append(int(fighter2.get('stance_orthodox', 0)))  # fighter2_stance_orthodox
                features.append(int(fighter1.get('stance_southpaw', 0)))  # fighter1_stance_southpaw
                features.append(int(fighter2.get('stance_southpaw', 0)))  # fighter2_stance_southpaw
                
                # Add weight class features for each fighter
                weight_classes = ['featherweight', 'lightweight', 'welterweight', 'middleweight', 'light_heavyweight', 'heavyweight']
                for wc in weight_classes:
                    features.append(int(fighter1.get(f'weight_class_{wc}', 0)))  # fighter1_weight_class_{wc}
                    features.append(int(fighter2.get(f'weight_class_{wc}', 0)))  # fighter2_weight_class_{wc}
                
                logger.debug(f"Final features array: {features}")
                logger.debug(f"Number of features: {len(features)}")
                
                # Convert to numpy array and scale
                features_array = np.array(features).reshape(1, -1)
                logger.debug(f"Features array shape before scaling: {features_array.shape}")
                
                try:
                    features_scaled = self.scaler.transform(features_array)[0]
                    logger.debug(f"Scaled features: {features_scaled}")
                    logger.debug(f"Scaled features shape: {features_scaled.shape}")
                except Exception as e:
                    logger.error(f"Error scaling features: {str(e)}", exc_info=True)
                    logger.error(f"Features array: {features_array}")
                    logger.error(f"Features array shape: {features_array.shape}")
                    raise

                
                # Make predictions
                winner_prob = self.winner_model.predict_proba([features_scaled])[0]
                fighter1_win_prob = winner_prob[1]
                fighter2_win_prob = winner_prob[0]
                
                if fighter1_win_prob > fighter2_win_prob:
                    winner = fighter1_name
                    confidence = fighter1_win_prob * 100
                else:
                    winner = fighter2_name
                    confidence = fighter2_win_prob * 100
                
                method_probs = self.method_model.predict_proba([features_scaled])[0]
                method_idx = np.argmax(method_probs)
                method = self.method_model.classes_[method_idx]
                
                round_probs = self.round_model.predict_proba([features_scaled])[0]
                round_idx = np.argmax(round_probs)
                round_num = int(self.round_model.classes_[round_idx])
                
                return {
                    'winner': winner,
                    'method': method,
                    'round': round_num,
                    'confidence': float(confidence)
                }
                
            except Exception as e:
                logger.error(f"Error processing features: {str(e)}", exc_info=True)
                return {'error': f"Error processing features: {str(e)}",
                        'details': f"Failed to create features from fighter data",
                        'fighter1_data': fighter1,
                        'fighter2_data': fighter2}
                
        except Exception as e:
            logger.error(f"Prediction failed with error: {str(e)}", exc_info=True)
            return {'error': f"An error occurred: {str(e)}",
                    'details': 'Internal prediction error',
                    'received_data': {'fighter1': fighter1_name, 'fighter2': fighter2_name}}
            feature_order = [
                'height_diff', 'weight_diff', 'reach_diff', 'age_diff',
                'win_pct_diff', 'ko_rate_diff', 'sub_rate_diff', 'dec_rate_diff', 'experience_diff',
                'fighter1_stance_orthodox', 'fighter2_stance_orthodox',
                'fighter1_stance_southpaw', 'fighter2_stance_southpaw',
                'fighter1_weight_class_featherweight', 'fighter1_weight_class_heavyweight',
                'fighter1_weight_class_light_heavyweight', 'fighter1_weight_class_lightweight',
                'fighter1_weight_class_middleweight', 'fighter1_weight_class_welterweight',
                'fighter2_weight_class_featherweight', 'fighter2_weight_class_heavyweight',
                'fighter2_weight_class_light_heavyweight', 'fighter2_weight_class_lightweight',
                'fighter2_weight_class_middleweight', 'fighter2_weight_class_welterweight'
            ]
        
        # Ensure features_df has all required columns
        for col in feature_order:
            if col not in features_df.columns:
                features_df[col] = 0  # Add missing columns with default value 0
        
        # Reorder features to match training data
        features_df = features_df[feature_order]
        
        # Convert to numpy array for compatibility with older scikit-learn versions
        scaled_features = self.scaler.transform(features_df.values)
        
        # Predict winner
        winner_prob = self.winner_model.predict_proba(scaled_features)[0]
        fighter1_win_prob = winner_prob[1] # Probability that fighter1_won is 1
        fighter2_win_prob = winner_prob[0] # Probability that fighter1_won is 0 (meaning fighter2 won)
        
        # Determine winner
        if fighter1_win_prob > fighter2_win_prob:
            winner = fighter1_name
            confidence = fighter1_win_prob * 100
        else:
            winner = fighter2_name
            confidence = fighter2_win_prob * 100
        
        # Predict method
        method_probs = self.method_model.predict_proba(scaled_features)[0]
        method_idx = np.argmax(method_probs)
        method = self.method_model.classes_[method_idx]
        
        # Predict round
        round_probs = self.round_model.predict_proba(scaled_features)[0]
        round_idx = np.argmax(round_probs)
        round_num = self.round_model.classes_[round_idx]
        
        # Return prediction
        prediction = {
            'fighter1': fighter1_name,
            'fighter2': fighter2_name,
            'winner': winner,
            'confidence': round(confidence, 2),
            'method': method,
            'round': round_num
        }
        
        return prediction

if __name__ == "__main__":
    # Example usage
    predictor = UFCFightPredictor()
    
    # Example prediction
    fighter1 = "Conor McGregor"
    fighter2 = "Dustin Poirier"
    
    prediction = predictor.predict_fight(fighter1, fighter2)
    
    print(f"\nPrediction for {fighter1} vs {fighter2}:")
    print(f"Winner: {prediction['winner']} (Confidence: {prediction['confidence']}%)")
    print(f"Method: {prediction['method']} in Round {prediction['round']}")
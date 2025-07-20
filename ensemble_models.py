"""
Advanced Multi-Model Ensemble for House Price Prediction
Combines multiple ML models for superior accuracy
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')

class HousePriceEnsemble:
    def __init__(self):
        """Initialize the ensemble with multiple models"""
        self.models = {}
        self.scalers = {}
        self.weights = {}
        self.is_trained = False
        self.feature_names = None
        
        # Initialize base models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize all base models with optimized parameters"""
        
        # Tree-based models (handle non-linear relationships well)
        self.models['random_forest'] = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.models['gradient_boosting'] = GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=8,
            min_samples_split=5,
            random_state=42
        )
        
        self.models['extra_trees'] = ExtraTreesRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        # Advanced gradient boosting
        self.models['xgboost'] = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=8,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        self.models['lightgbm'] = lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=8,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        # Linear models (good for interpretability and stable predictions)
        self.models['ridge'] = Ridge(alpha=1.0)
        self.models['lasso'] = Lasso(alpha=0.1)
        self.models['elastic_net'] = ElasticNet(alpha=0.1, l1_ratio=0.5)
        
        # Support Vector Machine (good for complex patterns)
        self.models['svr'] = SVR(kernel='rbf', C=100, gamma='scale')
        
        # Neural Network (captures complex non-linear relationships)
        self.models['neural_network'] = MLPRegressor(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=500,
            random_state=42
        )
        
        # Initialize scalers for each model
        for model_name in self.models.keys():
            if model_name in ['ridge', 'lasso', 'elastic_net', 'svr', 'neural_network']:
                self.scalers[model_name] = StandardScaler()
            else:
                self.scalers[model_name] = RobustScaler()
    
    def generate_enhanced_features(self, df):
        """Generate additional features for better predictions"""
        df_enhanced = df.copy()
        
        # Feature engineering
        df_enhanced['total_rooms'] = df_enhanced['bedrooms'] + df_enhanced['bathrooms']
        df_enhanced['sqft_per_room'] = df_enhanced['sqft_living'] / (df_enhanced['total_rooms'] + 1)
        df_enhanced['lot_to_living_ratio'] = df_enhanced['sqft_lot'] / df_enhanced['sqft_living']
        df_enhanced['age'] = 2024 - df_enhanced['yr_built']
        df_enhanced['age_squared'] = df_enhanced['age'] ** 2
        
        # Interaction features
        df_enhanced['grade_condition'] = df_enhanced['grade'] * df_enhanced['condition']
        df_enhanced['view_waterfront'] = df_enhanced['view'] * df_enhanced['waterfront']
        df_enhanced['sqft_grade'] = df_enhanced['sqft_living'] * df_enhanced['grade']
        
        # Binned features
        df_enhanced['size_category'] = pd.cut(df_enhanced['sqft_living'], 
                                            bins=[0, 1500, 2500, 4000, float('inf')], 
                                            labels=[1, 2, 3, 4])
        df_enhanced['age_category'] = pd.cut(df_enhanced['age'], 
                                           bins=[0, 10, 30, 50, float('inf')], 
                                           labels=[1, 2, 3, 4])
        
        return df_enhanced
    
    def train_ensemble(self, X, y, test_size=0.2):
        """Train all models in the ensemble"""
        print("🚀 Training Multi-Model Ensemble...")
        
        # Generate enhanced features
        X_enhanced = self.generate_enhanced_features(X)
        self.feature_names = X_enhanced.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_enhanced, y, test_size=test_size, random_state=42
        )
        
        model_scores = {}
        model_predictions = {}
        
        print(f"📊 Training {len(self.models)} models...")
        
        for model_name, model in self.models.items():
            print(f"   Training {model_name}...")
            
            # Scale features if needed
            if model_name in self.scalers:
                X_train_scaled = self.scalers[model_name].fit_transform(X_train)
                X_test_scaled = self.scalers[model_name].transform(X_test)
            else:
                X_train_scaled = X_train
                X_test_scaled = X_test
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            model_predictions[model_name] = y_pred
            
            # Calculate scores
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            model_scores[model_name] = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'accuracy': max(0, r2)  # Use R² as accuracy measure
            }
            
            print(f"      MAE: ${mae:,.0f} | RMSE: ${rmse:,.0f} | R²: {r2:.3f}")
        
        # Calculate ensemble weights based on performance
        self._calculate_weights(model_scores)
        
        # Test ensemble performance
        ensemble_pred = self._weighted_prediction(model_predictions)
        ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
        ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
        ensemble_r2 = r2_score(y_test, ensemble_pred)
        
        print(f"\n🎯 Ensemble Performance:")
        print(f"   MAE: ${ensemble_mae:,.0f}")
        print(f"   RMSE: ${ensemble_rmse:,.0f}")
        print(f"   R²: {ensemble_r2:.3f}")
        print(f"   Accuracy: {ensemble_r2*100:.1f}%")
        
        self.is_trained = True
        return model_scores, {
            'mae': ensemble_mae,
            'rmse': ensemble_rmse,
            'r2': ensemble_r2
        }
    
    def _calculate_weights(self, model_scores):
        """Calculate weights for ensemble based on model performance"""
        # Use inverse of RMSE as weight (better models get higher weights)
        total_inverse_rmse = sum(1/scores['rmse'] for scores in model_scores.values())
        
        for model_name, scores in model_scores.items():
            self.weights[model_name] = (1/scores['rmse']) / total_inverse_rmse
        
        print(f"\n⚖️  Model Weights:")
        for model_name, weight in sorted(self.weights.items(), key=lambda x: x[1], reverse=True):
            print(f"   {model_name}: {weight:.3f}")
    
    def _weighted_prediction(self, predictions_dict):
        """Combine predictions using weighted average"""
        weighted_pred = np.zeros(len(list(predictions_dict.values())[0]))
        
        for model_name, predictions in predictions_dict.items():
            weighted_pred += self.weights[model_name] * predictions
        
        return weighted_pred
    
    def predict(self, X):
        """Make ensemble predictions"""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")
        
        # Generate enhanced features
        X_enhanced = self.generate_enhanced_features(X)
        
        # Ensure same features as training
        for feature in self.feature_names:
            if feature not in X_enhanced.columns:
                X_enhanced[feature] = 0
        
        X_enhanced = X_enhanced[self.feature_names]
        
        predictions = {}
        
        # Get predictions from all models
        for model_name, model in self.models.items():
            if model_name in self.scalers:
                X_scaled = self.scalers[model_name].transform(X_enhanced)
            else:
                X_scaled = X_enhanced
            
            predictions[model_name] = model.predict(X_scaled)
        
        # Return weighted ensemble prediction
        return self._weighted_prediction(predictions)
    
    def get_model_contributions(self, X):
        """Get individual model predictions and their contributions"""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before getting contributions")
        
        X_enhanced = self.generate_enhanced_features(X)
        
        # Ensure same features as training
        for feature in self.feature_names:
            if feature not in X_enhanced.columns:
                X_enhanced[feature] = 0
        
        X_enhanced = X_enhanced[self.feature_names]
        
        contributions = {}
        
        for model_name, model in self.models.items():
            if model_name in self.scalers:
                X_scaled = self.scalers[model_name].transform(X_enhanced)
            else:
                X_scaled = X_enhanced
            
            pred = model.predict(X_scaled)[0]
            weight = self.weights[model_name]
            contribution = pred * weight
            
            contributions[model_name] = {
                'prediction': pred,
                'weight': weight,
                'contribution': contribution
            }
        
        return contributions
    
    def get_feature_importance(self):
        """Get feature importance from tree-based models"""
        if not self.is_trained:
            return None
        
        importance_dict = {}
        tree_models = ['random_forest', 'gradient_boosting', 'extra_trees', 'xgboost', 'lightgbm']
        
        for model_name in tree_models:
            if model_name in self.models:
                if hasattr(self.models[model_name], 'feature_importances_'):
                    importance_dict[model_name] = dict(zip(
                        self.feature_names, 
                        self.models[model_name].feature_importances_
                    ))
        
        # Average importance across tree models
        if importance_dict:
            avg_importance = {}
            for feature in self.feature_names:
                avg_importance[feature] = np.mean([
                    importance_dict[model][feature] 
                    for model in importance_dict.keys()
                ])
            
            return sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
        
        return None
    
    def save_ensemble(self, filepath):
        """Save the trained ensemble"""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before saving")
        
        ensemble_data = {
            'models': self.models,
            'scalers': self.scalers,
            'weights': self.weights,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        joblib.dump(ensemble_data, filepath)
        print(f"✅ Ensemble saved to {filepath}")
    
    def load_ensemble(self, filepath):
        """Load a trained ensemble"""
        ensemble_data = joblib.load(filepath)
        
        self.models = ensemble_data['models']
        self.scalers = ensemble_data['scalers']
        self.weights = ensemble_data['weights']
        self.feature_names = ensemble_data['feature_names']
        self.is_trained = ensemble_data['is_trained']
        
        print(f"✅ Ensemble loaded from {filepath}")

def create_sample_data(n_samples=2000):
    """Create enhanced sample data for training"""
    np.random.seed(42)
    
    data = {
        'bedrooms': np.random.randint(1, 6, n_samples),
        'bathrooms': np.random.choice([1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5], n_samples),
        'sqft_living': np.random.randint(800, 5000, n_samples),
        'sqft_lot': np.random.randint(3000, 20000, n_samples),
        'floors': np.random.choice([1, 1.5, 2, 2.5, 3], n_samples),
        'waterfront': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        'view': np.random.randint(0, 5, n_samples),
        'condition': np.random.randint(1, 6, n_samples),
        'grade': np.random.randint(3, 13, n_samples),
        'yr_built': np.random.randint(1900, 2024, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create more realistic price based on multiple factors
    df['price'] = (
        df['bedrooms'] * 45000 +
        df['bathrooms'] * 35000 +
        df['sqft_living'] * 180 +
        df['sqft_lot'] * 8 +
        df['floors'] * 25000 +
        df['waterfront'] * 250000 +
        df['view'] * 30000 +
        df['condition'] * 20000 +
        df['grade'] * 35000 +
        (2024 - df['yr_built']) * -800 +
        np.random.normal(0, 40000, n_samples)  # Add realistic noise
    )
    
    # Ensure positive prices
    df['price'] = np.maximum(df['price'], 150000)
    
    return df

if __name__ == "__main__":
    # Demo the ensemble
    print("🏠 House Price Ensemble Demo")
    print("=" * 50)
    
    # Create sample data
    df = create_sample_data(2000)
    X = df.drop('price', axis=1)
    y = df['price']
    
    # Initialize and train ensemble
    ensemble = HousePriceEnsemble()
    model_scores, ensemble_score = ensemble.train_ensemble(X, y)
    
    # Test prediction
    test_house = pd.DataFrame({
        'bedrooms': [3],
        'bathrooms': [2.5],
        'sqft_living': [2200],
        'sqft_lot': [8000],
        'floors': [2],
        'waterfront': [0],
        'view': [2],
        'condition': [4],
        'grade': [8],
        'yr_built': [2010]
    })
    
    prediction = ensemble.predict(test_house)
    print(f"\n🏡 Test Prediction: ${prediction[0]:,.0f}")
    
    # Show model contributions
    contributions = ensemble.get_model_contributions(test_house)
    print(f"\n📊 Model Contributions:")
    for model_name, contrib in contributions.items():
        print(f"   {model_name}: ${contrib['prediction']:,.0f} (weight: {contrib['weight']:.3f})")
    
    # Show feature importance
    importance = ensemble.get_feature_importance()
    if importance:
        print(f"\n🎯 Top Feature Importances:")
        for feature, imp in importance[:5]:
            print(f"   {feature}: {imp:.3f}")
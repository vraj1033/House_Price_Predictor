"""
Computer Vision Module for House Price Prediction
Analyzes property photos to extract features and assess quality
"""

import cv2
import numpy as np
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    print("⚠️  TensorFlow not available. Using basic computer vision features only.")
    TF_AVAILABLE = False
from PIL import Image, ImageEnhance
import requests
import base64
import json
import os
from typing import List, Dict, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PropertyVisionAnalyzer:
    def __init__(self):
        """Initialize the Computer Vision analyzer"""
        self.models = {}
        self.feature_extractors = {}
        self.quality_thresholds = {
            'excellent': 0.8,
            'good': 0.6,
            'fair': 0.4,
            'poor': 0.2
        }
        
        # Initialize models
        self._load_models()
        
    def _load_models(self):
        """Load pre-trained models for different tasks"""
        if not TF_AVAILABLE:
            logger.info("⚠️  TensorFlow not available. Using basic computer vision features only.")
            self.models = {}
            return
            
        try:
            # Load MobileNetV2 for general feature extraction
            self.models['feature_extractor'] = tf.keras.applications.MobileNetV2(
                weights='imagenet',
                include_top=False,
                pooling='avg'
            )
            
            # Load ResNet50 for detailed analysis
            self.models['detail_analyzer'] = tf.keras.applications.ResNet50(
                weights='imagenet',
                include_top=False,
                pooling='avg'
            )
            
            logger.info("✅ Computer Vision models loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Error loading CV models: {e}")
            self.models = {}
    
    def analyze_property_image(self, image_path: str) -> Dict:
        """
        Comprehensive analysis of a property image
        
        Args:
            image_path: Path to the property image
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Load and preprocess image
            image = self._load_and_preprocess_image(image_path)
            if image is None:
                return {'error': 'Failed to load image'}
            
            # Perform multiple analyses
            results = {
                'room_detection': self._detect_rooms(image),
                'amenity_detection': self._detect_amenities(image),
                'quality_assessment': self._assess_quality(image),
                'condition_analysis': self._analyze_condition(image),
                'style_classification': self._classify_style(image),
                'feature_extraction': self._extract_features(image),
                'price_impact_score': 0.0
            }
            
            # Calculate overall price impact
            results['price_impact_score'] = self._calculate_price_impact(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return {'error': str(e)}
    
    def _load_and_preprocess_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load and preprocess image for analysis"""
        try:
            if image_path.startswith('http'):
                # Download image from URL
                response = requests.get(image_path)
                image = Image.open(BytesIO(response.content))
            else:
                # Load local image
                image = Image.open(image_path)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize for processing
            image = image.resize((224, 224))
            
            # Convert to numpy array
            image_array = np.array(image) / 255.0
            
            return image_array
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return None
    
    def _detect_rooms(self, image: np.ndarray) -> Dict:
        """Detect and classify rooms in the image"""
        try:
            # Expand dimensions for model input
            image_batch = np.expand_dims(image, axis=0)
            
            # Extract features
            if 'feature_extractor' in self.models:
                features = self.models['feature_extractor'].predict(image_batch, verbose=0)
                
                # Simple room classification based on features
                # In production, you'd use a trained room classifier
                room_scores = {
                    'kitchen': self._calculate_room_probability(features, 'kitchen'),
                    'bathroom': self._calculate_room_probability(features, 'bathroom'),
                    'bedroom': self._calculate_room_probability(features, 'bedroom'),
                    'living_room': self._calculate_room_probability(features, 'living_room'),
                    'dining_room': self._calculate_room_probability(features, 'dining_room'),
                    'exterior': self._calculate_room_probability(features, 'exterior')
                }
                
                # Determine most likely room type
                detected_room = max(room_scores, key=room_scores.get)
                confidence = room_scores[detected_room]
                
                return {
                    'detected_room': detected_room,
                    'confidence': float(confidence),
                    'all_scores': room_scores
                }
            
            return {'detected_room': 'unknown', 'confidence': 0.0}
            
        except Exception as e:
            logger.error(f"Error in room detection: {e}")
            return {'detected_room': 'unknown', 'confidence': 0.0}
    
    def _detect_amenities(self, image: np.ndarray) -> Dict:
        """Detect amenities and features in the image"""
        try:
            # Convert to OpenCV format
            cv_image = (image * 255).astype(np.uint8)
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            
            amenities = {
                'fireplace': self._detect_fireplace(cv_image),
                'pool': self._detect_pool(cv_image),
                'hardwood_floors': self._detect_hardwood_floors(cv_image),
                'granite_counters': self._detect_granite_counters(cv_image),
                'stainless_appliances': self._detect_stainless_appliances(cv_image),
                'updated_kitchen': self._detect_updated_kitchen(cv_image),
                'crown_molding': self._detect_crown_molding(cv_image),
                'high_ceilings': self._detect_high_ceilings(cv_image)
            }
            
            # Count detected amenities
            detected_count = sum(1 for score in amenities.values() if score > 0.5)
            
            return {
                'amenities': amenities,
                'total_detected': detected_count,
                'premium_features': [k for k, v in amenities.items() if v > 0.7]
            }
            
        except Exception as e:
            logger.error(f"Error in amenity detection: {e}")
            return {'amenities': {}, 'total_detected': 0}
    
    def _assess_quality(self, image: np.ndarray) -> Dict:
        """Assess overall quality and condition of the property"""
        try:
            # Convert to PIL for quality analysis
            pil_image = Image.fromarray((image * 255).astype(np.uint8))
            
            # Analyze various quality factors
            quality_factors = {
                'brightness': self._analyze_brightness(pil_image),
                'contrast': self._analyze_contrast(pil_image),
                'sharpness': self._analyze_sharpness(pil_image),
                'color_balance': self._analyze_color_balance(pil_image),
                'cleanliness': self._analyze_cleanliness(image),
                'maintenance': self._analyze_maintenance(image),
                'modernization': self._analyze_modernization(image)
            }
            
            # Calculate overall quality score
            overall_score = np.mean(list(quality_factors.values()))
            
            # Determine quality category
            if overall_score >= self.quality_thresholds['excellent']:
                quality_category = 'excellent'
            elif overall_score >= self.quality_thresholds['good']:
                quality_category = 'good'
            elif overall_score >= self.quality_thresholds['fair']:
                quality_category = 'fair'
            else:
                quality_category = 'poor'
            
            return {
                'overall_score': float(overall_score),
                'quality_category': quality_category,
                'factors': quality_factors,
                'recommendations': self._generate_quality_recommendations(quality_factors)
            }
            
        except Exception as e:
            logger.error(f"Error in quality assessment: {e}")
            return {'overall_score': 0.5, 'quality_category': 'fair'}
    
    def _analyze_condition(self, image: np.ndarray) -> Dict:
        """Analyze property condition and identify issues"""
        try:
            cv_image = (image * 255).astype(np.uint8)
            
            condition_issues = {
                'water_damage': self._detect_water_damage(cv_image),
                'cracks': self._detect_cracks(cv_image),
                'peeling_paint': self._detect_peeling_paint(cv_image),
                'outdated_fixtures': self._detect_outdated_fixtures(cv_image),
                'wear_and_tear': self._detect_wear_and_tear(cv_image)
            }
            
            # Calculate condition score (inverse of issues)
            issue_score = np.mean(list(condition_issues.values()))
            condition_score = 1.0 - issue_score
            
            return {
                'condition_score': float(condition_score),
                'issues_detected': condition_issues,
                'major_issues': [k for k, v in condition_issues.items() if v > 0.6],
                'repair_priority': self._prioritize_repairs(condition_issues)
            }
            
        except Exception as e:
            logger.error(f"Error in condition analysis: {e}")
            return {'condition_score': 0.7, 'issues_detected': {}}
    
    def _classify_style(self, image: np.ndarray) -> Dict:
        """Classify architectural and interior style"""
        try:
            # Extract style features
            if 'detail_analyzer' in self.models:
                image_batch = np.expand_dims(image, axis=0)
                features = self.models['detail_analyzer'].predict(image_batch, verbose=0)
                
                # Style classification (simplified)
                style_scores = {
                    'modern': self._calculate_style_probability(features, 'modern'),
                    'traditional': self._calculate_style_probability(features, 'traditional'),
                    'contemporary': self._calculate_style_probability(features, 'contemporary'),
                    'rustic': self._calculate_style_probability(features, 'rustic'),
                    'luxury': self._calculate_style_probability(features, 'luxury'),
                    'minimalist': self._calculate_style_probability(features, 'minimalist')
                }
                
                detected_style = max(style_scores, key=style_scores.get)
                
                return {
                    'primary_style': detected_style,
                    'confidence': float(style_scores[detected_style]),
                    'style_scores': style_scores,
                    'style_appeal': self._calculate_style_appeal(detected_style)
                }
            
            return {'primary_style': 'unknown', 'confidence': 0.0}
            
        except Exception as e:
            logger.error(f"Error in style classification: {e}")
            return {'primary_style': 'unknown', 'confidence': 0.0}
    
    def _extract_features(self, image: np.ndarray) -> Dict:
        """Extract numerical features for price prediction"""
        try:
            features = {}
            
            # Color analysis
            mean_colors = np.mean(image, axis=(0, 1))
            features['avg_red'] = float(mean_colors[0])
            features['avg_green'] = float(mean_colors[1])
            features['avg_blue'] = float(mean_colors[2])
            
            # Texture analysis
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            features['texture_variance'] = float(np.var(gray))
            features['edge_density'] = self._calculate_edge_density(gray)
            
            # Spatial features
            features['brightness_std'] = float(np.std(gray))
            features['contrast_ratio'] = self._calculate_contrast_ratio(gray)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return {}
    
    def _calculate_price_impact(self, analysis_results: Dict) -> float:
        """Calculate overall price impact score from analysis results"""
        try:
            impact_score = 0.0
            
            # Quality impact (30% weight)
            if 'quality_assessment' in analysis_results:
                quality_score = analysis_results['quality_assessment'].get('overall_score', 0.5)
                impact_score += quality_score * 0.3
            
            # Condition impact (25% weight)
            if 'condition_analysis' in analysis_results:
                condition_score = analysis_results['condition_analysis'].get('condition_score', 0.7)
                impact_score += condition_score * 0.25
            
            # Amenities impact (25% weight)
            if 'amenity_detection' in analysis_results:
                amenity_count = analysis_results['amenity_detection'].get('total_detected', 0)
                amenity_score = min(1.0, amenity_count / 5.0)  # Normalize to max 5 amenities
                impact_score += amenity_score * 0.25
            
            # Style impact (20% weight)
            if 'style_classification' in analysis_results:
                style_appeal = analysis_results['style_classification'].get('style_appeal', 0.5)
                impact_score += style_appeal * 0.2
            
            return float(impact_score)
            
        except Exception as e:
            logger.error(f"Error calculating price impact: {e}")
            return 0.5
    
    # Helper methods for specific detections
    def _calculate_room_probability(self, features: np.ndarray, room_type: str) -> float:
        """Calculate probability of room type based on features"""
        # Simplified room detection - in production, use trained classifiers
        feature_sum = np.sum(features)
        
        room_signatures = {
            'kitchen': 0.3,
            'bathroom': 0.25,
            'bedroom': 0.35,
            'living_room': 0.4,
            'dining_room': 0.3,
            'exterior': 0.2
        }
        
        base_prob = room_signatures.get(room_type, 0.2)
        noise = np.random.normal(0, 0.1)  # Add some randomness
        
        return max(0.0, min(1.0, base_prob + noise))
    
    def _detect_fireplace(self, cv_image: np.ndarray) -> float:
        """Detect fireplace in image"""
        # Simplified detection - look for rectangular shapes and warm colors
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        # Look for warm colors (reds, oranges, browns)
        warm_mask = cv2.inRange(hsv, (0, 50, 50), (30, 255, 255))
        warm_ratio = np.sum(warm_mask > 0) / warm_mask.size
        
        return min(1.0, warm_ratio * 3.0)
    
    def _detect_pool(self, cv_image: np.ndarray) -> float:
        """Detect swimming pool in image"""
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        # Look for blue water colors
        blue_mask = cv2.inRange(hsv, (100, 50, 50), (130, 255, 255))
        blue_ratio = np.sum(blue_mask > 0) / blue_mask.size
        
        return min(1.0, blue_ratio * 5.0)
    
    def _detect_hardwood_floors(self, cv_image: np.ndarray) -> float:
        """Detect hardwood flooring"""
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        # Look for wood-like colors (browns, tans)
        wood_mask = cv2.inRange(hsv, (10, 50, 50), (25, 255, 200))
        wood_ratio = np.sum(wood_mask > 0) / wood_mask.size
        
        return min(1.0, wood_ratio * 2.0)
    
    def _detect_granite_counters(self, cv_image: np.ndarray) -> float:
        """Detect granite countertops"""
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Look for speckled texture typical of granite
        texture_variance = np.var(gray)
        normalized_variance = texture_variance / 10000.0
        
        return min(1.0, normalized_variance)
    
    def _detect_stainless_appliances(self, cv_image: np.ndarray) -> float:
        """Detect stainless steel appliances"""
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Look for metallic reflective surfaces
        bright_pixels = np.sum(gray > 200)
        bright_ratio = bright_pixels / gray.size
        
        return min(1.0, bright_ratio * 3.0)
    
    def _detect_updated_kitchen(self, cv_image: np.ndarray) -> float:
        """Detect updated/modern kitchen"""
        # Combine multiple factors
        granite_score = self._detect_granite_counters(cv_image)
        stainless_score = self._detect_stainless_appliances(cv_image)
        
        return (granite_score + stainless_score) / 2.0
    
    def _detect_crown_molding(self, cv_image: np.ndarray) -> float:
        """Detect crown molding"""
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Look for horizontal lines near ceiling
        edges = cv2.Canny(gray, 50, 150)
        horizontal_lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                                         minLineLength=50, maxLineGap=10)
        
        if horizontal_lines is not None:
            return min(1.0, len(horizontal_lines) / 10.0)
        
        return 0.0
    
    def _detect_high_ceilings(self, cv_image: np.ndarray) -> float:
        """Detect high ceilings"""
        height, width = cv_image.shape[:2]
        aspect_ratio = height / width
        
        # Higher aspect ratio might indicate high ceilings
        if aspect_ratio > 1.2:
            return min(1.0, (aspect_ratio - 1.0) * 2.0)
        
        return 0.0
    
    def _analyze_brightness(self, pil_image: Image.Image) -> float:
        """Analyze image brightness"""
        enhancer = ImageEnhance.Brightness(pil_image)
        # Convert to grayscale and calculate mean brightness
        gray = pil_image.convert('L')
        brightness = np.mean(np.array(gray)) / 255.0
        return float(brightness)
    
    def _analyze_contrast(self, pil_image: Image.Image) -> float:
        """Analyze image contrast"""
        gray = np.array(pil_image.convert('L'))
        contrast = np.std(gray) / 255.0
        return float(contrast)
    
    def _analyze_sharpness(self, pil_image: Image.Image) -> float:
        """Analyze image sharpness"""
        gray = np.array(pil_image.convert('L'))
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Normalize sharpness score
        sharpness = min(1.0, laplacian_var / 1000.0)
        return float(sharpness)
    
    def _analyze_color_balance(self, pil_image: Image.Image) -> float:
        """Analyze color balance"""
        rgb_array = np.array(pil_image)
        r_mean, g_mean, b_mean = np.mean(rgb_array, axis=(0, 1))
        
        # Calculate color balance (closer to equal RGB = better balance)
        color_std = np.std([r_mean, g_mean, b_mean])
        balance_score = 1.0 - min(1.0, color_std / 100.0)
        
        return float(balance_score)
    
    def _analyze_cleanliness(self, image: np.ndarray) -> float:
        """Analyze cleanliness of the space"""
        # Look for clutter, mess, or disorganization
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # High edge density might indicate clutter
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Lower edge density = cleaner appearance
        cleanliness = 1.0 - min(1.0, edge_density * 2.0)
        
        return float(cleanliness)
    
    def _analyze_maintenance(self, image: np.ndarray) -> float:
        """Analyze maintenance quality"""
        # Simplified maintenance analysis
        quality_factors = [
            self._detect_water_damage((image * 255).astype(np.uint8)),
            self._detect_cracks((image * 255).astype(np.uint8)),
            self._detect_peeling_paint((image * 255).astype(np.uint8))
        ]
        
        # Good maintenance = low issues
        maintenance_score = 1.0 - np.mean(quality_factors)
        
        return float(maintenance_score)
    
    def _analyze_modernization(self, image: np.ndarray) -> float:
        """Analyze modernization level"""
        cv_image = (image * 255).astype(np.uint8)
        
        modern_features = [
            self._detect_stainless_appliances(cv_image),
            self._detect_granite_counters(cv_image),
            self._detect_updated_kitchen(cv_image)
        ]
        
        modernization_score = np.mean(modern_features)
        
        return float(modernization_score)
    
    def _detect_water_damage(self, cv_image: np.ndarray) -> float:
        """Detect water damage signs"""
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        # Look for discoloration (yellows, browns)
        damage_mask = cv2.inRange(hsv, (15, 50, 50), (35, 255, 200))
        damage_ratio = np.sum(damage_mask > 0) / damage_mask.size
        
        return min(1.0, damage_ratio * 5.0)
    
    def _detect_cracks(self, cv_image: np.ndarray) -> float:
        """Detect cracks in walls/surfaces"""
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Look for thin lines (potential cracks)
        edges = cv2.Canny(gray, 30, 100)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, 
                               minLineLength=20, maxLineGap=5)
        
        if lines is not None:
            return min(1.0, len(lines) / 20.0)
        
        return 0.0
    
    def _detect_peeling_paint(self, cv_image: np.ndarray) -> float:
        """Detect peeling paint"""
        # Look for irregular textures and color variations
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        texture_variance = np.var(gray)
        
        # High variance might indicate peeling/flaking
        peeling_score = min(1.0, texture_variance / 5000.0)
        
        return float(peeling_score)
    
    def _detect_outdated_fixtures(self, cv_image: np.ndarray) -> float:
        """Detect outdated fixtures"""
        # Look for older color schemes and styles
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        # Look for dated colors (avocado green, harvest gold, etc.)
        dated_mask = cv2.inRange(hsv, (40, 50, 50), (80, 255, 200))
        dated_ratio = np.sum(dated_mask > 0) / dated_mask.size
        
        return min(1.0, dated_ratio * 3.0)
    
    def _detect_wear_and_tear(self, cv_image: np.ndarray) -> float:
        """Detect general wear and tear"""
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Look for scratches, scuffs, and wear patterns
        # High frequency noise might indicate wear
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = cv2.absdiff(gray, blur)
        wear_score = np.mean(noise) / 255.0
        
        return float(wear_score)
    
    def _calculate_style_probability(self, features: np.ndarray, style: str) -> float:
        """Calculate style probability based on features"""
        # Simplified style classification
        style_signatures = {
            'modern': 0.4,
            'traditional': 0.3,
            'contemporary': 0.35,
            'rustic': 0.25,
            'luxury': 0.45,
            'minimalist': 0.3
        }
        
        base_prob = style_signatures.get(style, 0.2)
        noise = np.random.normal(0, 0.1)
        
        return max(0.0, min(1.0, base_prob + noise))
    
    def _calculate_style_appeal(self, style: str) -> float:
        """Calculate market appeal of detected style"""
        appeal_scores = {
            'modern': 0.8,
            'contemporary': 0.75,
            'luxury': 0.9,
            'traditional': 0.7,
            'minimalist': 0.65,
            'rustic': 0.6
        }
        
        return appeal_scores.get(style, 0.5)
    
    def _calculate_edge_density(self, gray_image: np.ndarray) -> float:
        """Calculate edge density in image"""
        edges = cv2.Canny(gray_image, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        return float(edge_density)
    
    def _calculate_contrast_ratio(self, gray_image: np.ndarray) -> float:
        """Calculate contrast ratio"""
        min_val = np.min(gray_image)
        max_val = np.max(gray_image)
        
        if min_val == 0:
            return float(max_val / 1.0)
        
        return float(max_val / min_val)
    
    def _generate_quality_recommendations(self, quality_factors: Dict) -> List[str]:
        """Generate recommendations based on quality analysis"""
        recommendations = []
        
        if quality_factors.get('brightness', 0.5) < 0.4:
            recommendations.append("Improve lighting - add more light sources or increase natural light")
        
        if quality_factors.get('contrast', 0.5) < 0.3:
            recommendations.append("Enhance contrast - consider repainting with contrasting colors")
        
        if quality_factors.get('cleanliness', 0.5) < 0.6:
            recommendations.append("Declutter and deep clean the space")
        
        if quality_factors.get('maintenance', 0.5) < 0.5:
            recommendations.append("Address maintenance issues - repair damage and refresh surfaces")
        
        if quality_factors.get('modernization', 0.5) < 0.4:
            recommendations.append("Consider updating fixtures and finishes for modern appeal")
        
        return recommendations
    
    def _prioritize_repairs(self, condition_issues: Dict) -> List[str]:
        """Prioritize repairs based on severity"""
        priority_repairs = []
        
        # Sort issues by severity
        sorted_issues = sorted(condition_issues.items(), key=lambda x: x[1], reverse=True)
        
        for issue, severity in sorted_issues:
            if severity > 0.6:
                priority_repairs.append(f"High Priority: Address {issue.replace('_', ' ')}")
            elif severity > 0.4:
                priority_repairs.append(f"Medium Priority: Fix {issue.replace('_', ' ')}")
            elif severity > 0.2:
                priority_repairs.append(f"Low Priority: Monitor {issue.replace('_', ' ')}")
        
        return priority_repairs

# Utility functions for integration
def analyze_multiple_images(image_paths: List[str]) -> Dict:
    """Analyze multiple property images and combine results"""
    analyzer = PropertyVisionAnalyzer()
    
    all_results = []
    for image_path in image_paths:
        result = analyzer.analyze_property_image(image_path)
        if 'error' not in result:
            all_results.append(result)
    
    if not all_results:
        return {'error': 'No valid images could be analyzed'}
    
    # Combine results
    combined_results = {
        'total_images': len(all_results),
        'average_quality': np.mean([r['quality_assessment']['overall_score'] for r in all_results]),
        'average_condition': np.mean([r['condition_analysis']['condition_score'] for r in all_results]),
        'detected_amenities': set(),
        'detected_rooms': [],
        'overall_price_impact': 0.0,
        'recommendations': []
    }
    
    # Aggregate amenities and rooms
    for result in all_results:
        # Collect amenities
        amenities = result.get('amenity_detection', {}).get('premium_features', [])
        combined_results['detected_amenities'].update(amenities)
        
        # Collect rooms
        room = result.get('room_detection', {}).get('detected_room', 'unknown')
        if room != 'unknown':
            combined_results['detected_rooms'].append(room)
        
        # Collect recommendations
        recs = result.get('quality_assessment', {}).get('recommendations', [])
        combined_results['recommendations'].extend(recs)
    
    # Calculate overall price impact
    combined_results['overall_price_impact'] = np.mean([r['price_impact_score'] for r in all_results])
    
    # Remove duplicates
    combined_results['detected_amenities'] = list(combined_results['detected_amenities'])
    combined_results['recommendations'] = list(set(combined_results['recommendations']))
    
    return combined_results

if __name__ == "__main__":
    # Demo the computer vision system
    print("🏠 Property Computer Vision Demo")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = PropertyVisionAnalyzer()
    
    # Note: In a real scenario, you would provide actual image paths
    print("Computer Vision system initialized successfully!")
    print("Ready to analyze property images for:")
    print("- Room detection and classification")
    print("- Amenity identification")
    print("- Quality assessment")
    print("- Condition analysis")
    print("- Style classification")
    print("- Price impact calculation")
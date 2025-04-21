import google.generativeai as genai
import json
import os
from typing import Dict, Any

class AVDecisionExplainer:
    def __init__(self, api_key: str):
        """Initialize with Gemini API key"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
    
    def explain_decision(self, decision_data: Dict[str, Any]) -> str:
        """Generate a simple 1-2 line explanation for a car decision"""
        # Extract only the essential information
        action = decision_data.get('action', 'unknown')
        primary_detection = decision_data.get('primary_detection', 'none')
        speed = decision_data.get('speed', 'unknown')
        lane_position = decision_data.get('lane_position', 'unknown')
        weather = decision_data.get('weather', 'unknown')
        road_condition = decision_data.get('road_condition', 'unknown')
        
        # Additional context if available
        object_details = decision_data.get('object_details', {})
        object_size = object_details.get('size', '')
        object_distance = object_details.get('distance', '')
        object_type = object_details.get('type', '')
        
        nearby_vehicles = decision_data.get('nearby_vehicles', {})
        left_lane = nearby_vehicles.get('left_lane', 'unknown')
        right_lane = nearby_vehicles.get('right_lane', 'unknown')
        
        prompt = f"""
        As an autonomous vehicle decision explainer, provide a VERY SHORT explanation (1-2 lines only) for why the car made this decision:
        
        Action: {action}
        Primary Detection: {primary_detection}
        Speed: {speed} mph
        Lane Position: {lane_position}
        Weather: {weather}
        Road Condition: {road_condition}
        
        Additional Context:
        - Object Size: {object_size}
        - Object Distance: {object_distance}
        - Object Type: {object_type}
        - Left Lane: {left_lane}
        - Right Lane: {right_lane}
        
        Explain in 1-2 lines ONLY why the car made this decision. Be concise and focus on the most important reason.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error generating explanation: {str(e)}"

# Example usage for critical situations
def main():
    # Initialize with your Gemini API key
    api_key = "AIzaSyBVf9iH10FG7Gj0HjuuQGXfn_pFOymlqjA"  # Replace with your actual API key
    explainer = AVDecisionExplainer(api_key=api_key)
    
    # Critical Situation 1: Emergency Stop for Pedestrian
    critical_decision1 = {
        'action': 'emergency_stop',
        'primary_detection': 'pedestrian_running_into_road',
        'speed': 30,
        'lane_position': 'center',
        'weather': 'clear',
        'road_condition': 'dry',
        'object_details': {
            'type': 'child',
            'distance': '10m',
            'movement': 'running_into_road'
        },
        'nearby_vehicles': {
            'behind': '15m'
        }
    }
    
    # Critical Situation 2: Running Red Light to Avoid Rear-End Collision
    critical_decision2 = {
        'action': 'proceed_through_red_light',
        'primary_detection': 'imminent_rear_collision',
        'speed': 20,
        'lane_position': 'center',
        'weather': 'clear',
        'road_condition': 'dry',
        'object_details': {
            'type': 'heavy_truck',
            'distance': '3m',
            'speed': '40mph'
        },
        'nearby_vehicles': {
            'intersection': 'clear',
            'behind': '3m'
        }
    }
    
    # Critical Situation 3: Sudden Steering Input for Falling Object
    critical_decision3 = {
        'action': 'sharp_left_turn',
        'primary_detection': 'falling_object_from_truck',
        'speed': 60,
        'lane_position': 'center',
        'weather': 'windy',
        'road_condition': 'dry',
        'object_details': {
            'type': 'ladder',
            'distance': '20m',
            'trajectory': 'falling_into_lane'
        },
        'nearby_vehicles': {
            'left_lane': 'clear',
            'right_lane': 'occupied',
            'behind': '30m'
        }
    }
    
    # Critical Situation 4: Cyclist Sudden Appearance
    critical_decision4 = {
        'action': 'immediate_deceleration_and_swerve',
        'primary_detection': 'cyclist_entering_from_blind_spot',
        'speed': 25,
        'lane_position': 'right',
        'weather': 'rain',
        'road_condition': 'wet',
        'object_details': {
            'type': 'cyclist',
            'distance': '5m',
            'movement': 'entering_lane'
        },
        'nearby_vehicles': {
            'left_lane': 'occupied',
            'oncoming': 'clear'
        }
    }
    
    # Critical Situation 5: Airbag Deployment Decision
    critical_decision5 = {
        'action': 'deploy_airbags',
        'primary_detection': 'unavoidable_frontal_collision',
        'speed': 35,
        'lane_position': 'center',
        'weather': 'clear',
        'road_condition': 'dry',
        'object_details': {
            'type': 'stopped_vehicle',
            'distance': '2m',
            'impact_probability': '95%'
        },
        'nearby_vehicles': {
            'all_sides': 'blocked'
        }
    }
    
    # Generate and print explanations for all critical situations
    critical_scenarios = [
        critical_decision1,
        critical_decision2,
        critical_decision3,
        critical_decision4,
        critical_decision5
    ]
    
    for i, scenario in enumerate(critical_scenarios, 1):
        explanation = explainer.explain_decision(scenario)
        print(f"\nCritical Situation {i}: {scenario['action']}")
        print(f"Explanation: {explanation}")

if __name__ == "__main__":
    main()
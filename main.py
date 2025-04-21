import google.generativeai as genai
import json
import time
from typing import Dict, List, Any, Tuple
import numpy as np
from datetime import datetime

class NextGenMobilityExplainer:
    def __init__(self, api_key: str):
        """Initialize with Gemini API key and advanced features"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        self.passenger_profiles = {}
        self.decision_history = []
        self.passenger_state = {}
        
    def detect_passenger_state(self, passenger_id: str, biometric_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect passenger emotional and physical state from biometric sensors"""
        # Simulated biometric analysis
        heart_rate = biometric_data.get('heart_rate', 70)
        skin_conductance = biometric_data.get('skin_conductance', 0.5)
        facial_expression = biometric_data.get('facial_expression', 'neutral')
        
        # Simple state detection logic (would be ML model in production)
        if heart_rate > 100 and skin_conductance > 0.8:
            emotional_state = 'anxious'
        elif heart_rate < 60:
            emotional_state = 'relaxed'
        elif facial_expression == 'smile':
            emotional_state = 'content'
        else:
            emotional_state = 'neutral'
        
        self.passenger_state[passenger_id] = {
            'emotional_state': emotional_state,
            'heart_rate': heart_rate,
            'timestamp': datetime.now().isoformat()
        }
        
        return self.passenger_state[passenger_id]
    
    def predict_future_actions(self, current_scenario: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Predict and explain probable upcoming actions"""
        prompt = f"""
        Based on the current driving scenario:
        {json.dumps(current_scenario, indent=2)}
        
        Predict the NEXT 3 MOST LIKELY actions the vehicle might take in the next 30 seconds.
        For each prediction, provide:
        1. Action
        2. Probability (%)
        3. Reason
        4. Potential passenger reaction
        
        Format as JSON array.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return json.loads(response.text)
        except:
            return []
    
    def adaptive_explanation(self, decision_data: Dict[str, Any], passenger_id: str) -> str:
        """Generate explanations adapted to passenger's current state and preferences"""
        passenger_profile = self.passenger_profiles.get(passenger_id, {})
        passenger_state = self.passenger_state.get(passenger_id, {})
        
        # Determine communication style based on profile and state
        technical_level = passenger_profile.get('technical_level', 'basic')
        preferred_language = passenger_profile.get('preferred_language', 'english')
        emotional_state = passenger_state.get('emotional_state', 'neutral')
        
        prompt = f"""
        Generate an explanation for this autonomous vehicle decision, adapted to the passenger:
        
        Decision: {decision_data['action']}
        Context: {json.dumps(decision_data, indent=2)}
        
        Passenger State:
        - Technical Understanding: {technical_level}
        - Emotional State: {emotional_state}
        - Language: {preferred_language}
        
        Rules:
        1. For anxious passengers: Use calm, reassuring language
        2. For technical passengers: Include more technical details
        3. For basic level: Use simple analogies
        4. Always prioritize safety reassurance
        
        Keep it to 1-2 sentences.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"I made a decision to ensure your safety. {decision_data['action']}"
    
    def generate_safety_confidence_score(self, decision_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a safety confidence analysis for the decision"""
        prompt = f"""
        Analyze the safety of this autonomous vehicle decision:
        {json.dumps(decision_data, indent=2)}
        
        Provide:
        1. Safety confidence score (0-100)
        2. Primary safety factors
        3. Risk mitigation measures taken
        4. Alternative actions considered and why they were rejected
        
        Format as JSON.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return json.loads(response.text)
        except:
            return {"safety_score": 85, "primary_factors": ["Default safety protocol"]}
    
    def situational_learning_feedback(self, decision_data: Dict[str, Any], outcome: Dict[str, Any]) -> Dict[str, Any]:
        """Generate learning points from decisions for continuous improvement"""
        prompt = f"""
        Analyze this autonomous vehicle decision and its outcome:
        
        Decision: {json.dumps(decision_data, indent=2)}
        Outcome: {json.dumps(outcome, indent=2)}
        
        Provide:
        1. What went well
        2. What could be improved
        3. Learning points for similar situations
        4. Suggested algorithm adjustments
        
        Format as JSON.
        """
        
        try:
            response = self.model.generate_content(prompt)
            learning_points = json.loads(response.text)
            self.decision_history.append({
                'decision': decision_data,
                'outcome': outcome,
                'learning': learning_points,
                'timestamp': datetime.now().isoformat()
            })
            return learning_points
        except:
            return {"learning_points": ["Decision logged for future analysis"]}
    
    def create_passenger_profile(self, passenger_id: str, survey_data: Dict[str, Any]) -> None:
        """Create personalized passenger profile for adaptive communication"""
        self.passenger_profiles[passenger_id] = {
            'technical_level': survey_data.get('technical_level', 'basic'),
            'preferred_language': survey_data.get('preferred_language', 'english'),
            'communication_style': survey_data.get('communication_style', 'concise'),
            'safety_priority': survey_data.get('safety_priority', 'high'),
            'created_at': datetime.now().isoformat()
        }
    
    def emergency_protocol_explainer(self, emergency_type: str, actions_taken: List[str]) -> Dict[str, Any]:
        """Specialized explainer for emergency situations"""
        prompt = f"""
        Create a clear emergency protocol explanation:
        
        Emergency Type: {emergency_type}
        Actions Taken: {json.dumps(actions_taken, indent=2)}
        
        Provide:
        1. Why this is considered an emergency
        2. Why these specific actions were taken
        3. Expected outcome
        4. Instructions for passengers
        5. Post-emergency status report
        
        Format as JSON with clear, calm language.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return json.loads(response.text)
        except:
            return {
                "explanation": f"Emergency protocol activated for {emergency_type}",
                "passenger_instructions": "Please remain calm and follow vehicle guidance"
            }

# Enhanced Example Usage
def advanced_demo():
    explainer = NextGenMobilityExplainer(api_key="YOUR_API_KEY")
    
    # Create a passenger profile
    explainer.create_passenger_profile("passenger_001", {
        'technical_level': 'intermediate',
        'preferred_language': 'english',
        'communication_style': 'detailed',
        'safety_priority': 'high'
    })
    
    # Simulate biometric data
    biometric_data = {
        'heart_rate': 95,
        'skin_conductance': 0.7,
        'facial_expression': 'tense'
    }
    
    # Detect passenger state
    passenger_state = explainer.detect_passenger_state("passenger_001", biometric_data)
    print(f"Passenger State: {passenger_state}")
    
    # Complex emergency scenario
    emergency_decision = {
        'action': 'emergency_brake_and_pull_over',
        'primary_detection': 'child_running_onto_road',
        'speed': 35,
        'lane_position': 'center',
        'weather': 'clear',
        'road_condition': 'dry',
        'object_details': {
            'type': 'child',
            'distance': '15m',
            'trajectory': 'entering_road'
        }
    }
    
    # Generate adaptive explanation
    explanation = explainer.adaptive_explanation(emergency_decision, "passenger_001")
    print(f"\nAdaptive Explanation: {explanation}")
    
    # Generate safety analysis
    safety_analysis = explainer.generate_safety_confidence_score(emergency_decision)
    print(f"\nSafety Analysis: {json.dumps(safety_analysis, indent=2)}")
    
    # Predict future actions
    future_actions = explainer.predict_future_actions(emergency_decision)
    print(f"\nPredicted Actions: {json.dumps(future_actions, indent=2)}")
    
    # Emergency protocol explanation
    emergency_explanation = explainer.emergency_protocol_explainer(
        "pedestrian_hazard",
        ["emergency_brake", "hazard_lights_on", "alert_authorities"]
    )
    print(f"\nEmergency Protocol: {json.dumps(emergency_explanation, indent=2)}")
    
    # Simulate outcome and generate learning points
    outcome = {
        'collision_avoided': True,
        'time_to_stop': '2.3s',
        'passenger_feedback': 'felt_safe',
        'external_impact': 'none'
    }
    
    learning_points = explainer.situational_learning_feedback(emergency_decision, outcome)
    print(f"\nLearning Points: {json.dumps(learning_points, indent=2)}")

if __name__ == "__main__":
    advanced_demo()
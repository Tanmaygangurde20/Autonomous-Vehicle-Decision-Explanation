# Autonomous Vehicle Decision Explainer

This application uses Google's Gemini LLM to provide simple 1-2 line explanations for autonomous vehicle decisions based on sensor data.

## Features

- **Simple Explanations**: Generates concise 1-2 line explanations for why the car made a specific decision
- **Sensor Data Analysis**: Processes complex sensor data to extract the most relevant information
- **Natural Language**: Uses Gemini LLM to create human-readable explanations

## How It Works

The application takes sensor data in JSON format and uses Gemini LLM to generate a simple explanation of why the car made a specific decision. For example:

```
Action: emergency_swerve_right
Explanation: Car swerved right to avoid a large metal debris detected 15m ahead in the center lane, as the right lane was clear and safer than the occupied left lane.
```

## Requirements

- Python 3.7+
- Google Generative AI Python SDK
- Required Python packages:
  - google-generativeai
  - json
  - os

## Installation

1. Clone this repository
2. Install required packages:
   ```
   pip install google-generativeai
   ```
3. Set up your Gemini API key:
   - Get an API key from Google AI Studio
   - Replace the API key in the `main()` function of `app.py`

## Usage

Run the application with:
```
python app.py
```

The application will process example decision data and generate simple explanations for each decision.

## Example Input/Output

### Input:
```json
{
    'action': 'emergency_swerve_right',
    'sensors': {
        'primary_detection': 'fallen_object_in_lane',
        'other_sensors': {
            'lidar': {'object_size': '1.5m', 'distance': '15m'},
            'radar': {'closing_speed': '30mph'},
            'camera': {'object_classification': 'metal_debris'}
        }
    },
    'vehicle_state': {
        'speed': 65,
        'lane_position': 'center',
        'nearby_vehicles': {
            'left_lane': 'occupied',
            'right_lane': 'clear',
            'behind': '20m'
        }
    },
    'environment': {
        'weather': 'clear',
        'visibility': 'good',
        'road_condition': 'dry'
    }
}
```

### Output:
```
Action: emergency_swerve_right
Explanation: Car swerved right to avoid a large metal debris detected 15m ahead in the center lane, as the right lane was clear and safer than the occupied left lane.
```

## Customization

You can modify the prompt template in the `explain_decision()` method to adjust the style or focus of the explanations.

## License

MIT 
import os
import json
import torch
from datasets import Dataset
from transformers import TrainingArguments, DataCollatorForLanguageModeling
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported

# Set your Hugging Face token if needed
os.environ["HF_TOKEN"] = ""  # Replace with your actual token

# 1. LOAD THE DATASET
def load_dataset_from_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    return Dataset.from_dict({
        'instruction': [item['instruction'] for item in data],
        'input': [item['input'] for item in data],
        'output': [item['output'] for item in data]
    })

# Load from file
dataset = load_dataset_from_file("/content/autonomous_vehicle_dataset_3000.json")

# 2. FORMAT THE DATASET FOR TRAINING
# Template for AV decision explanation training data
av_prompt_template = """
<s>[INST] Generate an explanation for why the autonomous vehicle took the specified action based on the sensor data provided.

Timestamp: {timestamp}

Sensor Data:
- LiDAR: {lidar_data}
- Radar: {radar_data}
- Camera: {camera_data}
- GPS: {gps_data}
- IMU: {imu_data}
- Ultrasonic: {ultrasonic_data}

Vehicle State:
{vehicle_state}

Environment:
{environment}

Action Taken: {action} [/INST]

{explanation} </s>
"""

# Function to format the complex AV dataset
def formatting_av_prompts_func(examples):
    formatted_prompts = []
    
    for i in range(len(examples["instruction"])):
        # Extract nested sensor data
        input_data = examples["input"][i]
        
        # Format the nested sensor data for better readability
        lidar_data = f"Detected objects: {input_data['sensors']['LiDAR']['detected_objects']}, " \
                     f"Closest object distance: {input_data['sensors']['LiDAR']['closest_object_distance']} feet, " \
                     f"Object shape: {input_data['sensors']['LiDAR']['object_shape']}, " \
                     f"Dimensions: {input_data['sensors']['LiDAR']['object_height']}x{input_data['sensors']['LiDAR']['object_width']}x{input_data['sensors']['LiDAR']['object_length']} meters"
        
        radar_data = f"Object velocity: {input_data['sensors']['Radar']['object_velocity_towards_vehicle']} mph toward vehicle, " \
                    f"Position: {input_data['sensors']['Radar']['relative_position']}, " \
                    f"Cross-section: {input_data['sensors']['Radar']['object_cross_section']} sq ft"
        
        camera_data = f"Object class: {input_data['sensors']['Camera']['object_class']}, " \
                     f"Color: {input_data['sensors']['Camera']['object_color']}, " \
                     f"Visibility: {input_data['sensors']['Camera']['object_visibility']}"
        
        gps_data = f"Coordinates: {input_data['sensors']['GPS']['latitude']}, {input_data['sensors']['GPS']['longitude']}, " \
                  f"Speed: {input_data['sensors']['GPS']['speed_mph']} mph, " \
                  f"Heading: {input_data['sensors']['GPS']['heading']}, " \
                  f"Road type: {input_data['sensors']['GPS']['road_type']}"
        
        imu_data = f"Acceleration: {input_data['sensors']['IMU']['acceleration']} m/s², " \
                  f"Yaw rate: {input_data['sensors']['IMU']['yaw_rate']}, " \
                  f"Roll: {input_data['sensors']['IMU']['roll']}, " \
                  f"Pitch: {input_data['sensors']['IMU']['pitch']}"
        
        ultrasonic_data = f"Proximity (feet) - Left: {input_data['sensors']['Ultrasonic']['proximity_left']}, " \
                         f"Right: {input_data['sensors']['Ultrasonic']['proximity_right']}, " \
                         f"Front: {input_data['sensors']['Ultrasonic']['proximity_front']}, " \
                         f"Rear: {input_data['sensors']['Ultrasonic']['proximity_rear']}"
        
        vehicle_state = f"Lane position: {input_data['vehicle_state']['lane']}\n" \
                       f"Right lane status: {input_data['vehicle_state']['right_lane_status']}\n" \
                       f"Left lane status: {input_data['vehicle_state']['left_lane_status']}\n" \
                       f"Distances - Rear vehicle: {input_data['vehicle_state']['rear_vehicle_distance']} feet, " \
                       f"Front vehicle: {input_data['vehicle_state']['front_vehicle_distance']} feet\n" \
                       f"Vehicle type: {input_data['vehicle_state']['vehicle_type']}"
        
        environment = f"Weather: {input_data['environment']['weather']}\n" \
                     f"Road condition: {input_data['environment']['road_condition']}\n" \
                     f"Visibility: {input_data['environment']['visibility']}\n" \
                     f"Time of day: {input_data['environment']['time_of_day']}\n" \
                     f"Traffic density: {input_data['environment']['traffic_density']}\n" \
                     f"Light conditions: {input_data['environment']['light_conditions']}"
        
        # Format the prompt using the template
        prompt = av_prompt_template.format(
            timestamp=input_data["timestamp"],
            lidar_data=lidar_data,
            radar_data=radar_data,
            camera_data=camera_data,
            gps_data=gps_data,
            imu_data=imu_data,
            ultrasonic_data=ultrasonic_data,
            vehicle_state=vehicle_state,
            environment=environment,
            action=input_data["action"],
            explanation=examples["output"][i]
        )
        
        formatted_prompts.append(prompt)
    
    return {"text": formatted_prompts}

# Process the dataset with the formatting function
processed_dataset = dataset.map(formatting_av_prompts_func, batched=True)

# Print a sample to verify formatting
print("Sample formatted data:")
print(processed_dataset["text"][0])

# 3. LOAD THE MODEL
print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Llama-3.2-3B",
    max_seq_length=4096,
    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    load_in_4bit=True if torch.cuda.is_available() else False,
)

# Fix for pickling error: Set padding token
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 4. PREPARE FOR TRAINING WITH UNSLOTH'S DIRECT APPROACH
print("Configuring LoRA parameters...")
# Define LoRA config
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                   "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing=True,
    random_state=42,
    use_rslora=False,
)

# 5. SETUP TRAINING ARGS MANUALLY
print("Setting up training arguments...")
output_dir = "av_explanation_model"

# Create training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    learning_rate=2e-4,
    weight_decay=0.01,
    bf16=is_bfloat16_supported() and torch.cuda.is_available(),
    fp16=not is_bfloat16_supported() and torch.cuda.is_available(),
    logging_steps=10,
    report_to="none",
    save_strategy="epoch",
    save_total_limit=3,
    optim="adamw_torch",  # Use standard AdamW instead of 8-bit
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
)

# 6. USE SFTTrainer WITH THE CORRECT SETUP
print("Initializing SFTTrainer...")

# IMPORTANT: Let SFTTrainer handle the tokenization internally
# Don't pre-tokenize the dataset manually

try:
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,  # Use the processed but not tokenized dataset
        tokenizer=tokenizer,
        dataset_text_field="text",  # Specify which field contains the text
        max_seq_length=4096,
    )
    
    # 7. RUN TRAINING
    print("Starting training...")
    train_result = trainer.train()
    print(f"Training completed. Stats: {train_result}")
    
    # 8. SAVE THE MODEL
    print("Saving model...")
    trainer.save_model(output_dir)
    print(f"Model saved to {output_dir}")
    
except Exception as e:
    print(f"Error during training: {e}")
    
    # Fallback to manual approach if SFTTrainer fails
    print("Falling back to alternative approach...")
    
    # Properly tokenize the dataset
    def tokenize_function(examples):
        # Using the tokenizer properly to get tokenized outputs
        outputs = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=4096,
            return_tensors=None,  # Return Python lists
        )
        return outputs
    
    # Tokenize dataset properly
    print("Tokenizing dataset manually...")
    tokenized_dataset = processed_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["instruction", "input", "output"],
    )
    
    # DataCollator that will handle padding correctly
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False,
    )
    
    # Use base Trainer
    from transformers import Trainer
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Run training with fallback method
    print("Starting training with fallback method...")
    train_result = trainer.train()
    print(f"Training completed with fallback method. Stats: {train_result}")
    
    # Save the model
    trainer.save_model(output_dir)
    print(f"Model saved to {output_dir}")

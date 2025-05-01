

---

# 🚗 Autonomous Vehicle Decision Explainer

**Bridging trust and transparency in autonomous driving**  
_A project integrating fine-tuned Large Language Models (LLMs) and upcoming Feature Extraction Networks (FEN) to make autonomous vehicle decisions interpretable and explainable._

---

## 📖 Overview

Autonomous vehicles (AVs) are rapidly transforming the transportation landscape with their ability to navigate roads without human intervention. However, these systems make decisions using complex sensor fusion, deep learning models, and rules that are not easily interpretable by humans.  

This lack of transparency raises questions regarding **safety**, **trust**, and **accountability**.  
To address this, our project integrates **Feature Extraction Networks (FEN)** (coming soon) with **fine-tuned Large Language Models (LLMs)** to generate clear, understandable explanations for each vehicle decision — ultimately improving user confidence and regulatory acceptance.

---

## ✨ Key Features

- **Fine-tuned Meta LLaMA 3.2B Model**  
  Domain-adapted for autonomous vehicle decision explanations using a custom 3000-sample dataset.

- **One-Shot Prompt Fine-tuning**  
  Combined fine-tuning via `app.py` and `main.py` using structured prompts with domain-specific instructions.

- **Sensor Fusion Context Handling**  
  Accepts multi-modal data inputs: LiDAR, Radar, Camera, GPS, IMU, Ultrasonic, vehicle state, and environmental conditions.

- **Clear Action Explanations**  
  Produces simple, precise 1–2 line rationales for vehicle actions (e.g., lane change, braking, speed adjustment).

- **[Coming Soon] Feature Extraction Networks (FEN)**  
  Planned integration of FENs to improve sensor feature interpretation before explanation generation — boosting both performance and explainability.

---

## 🗂️ Dataset Structure (3000 Samples)

Each data sample contains an **Instruction–Input–Output (IIO)** format optimized for LLM fine-tuning.

```json
{
  "instruction": "Generate an explanation for why the autonomous vehicle took the specified action based on the sensor data provided.",
  "input": {
    "timestamp": "2024-04-03T17:55:41Z",
    "sensors": {
      "LiDAR": { "detected_objects": 4, "closest_object_distance": 78.4, ... },
      "Radar": { "object_velocity_towards_vehicle": -22, ... },
      "Camera": { "object_class": "motorcycle", ... },
      "GPS": { "latitude": 29.7604, "longitude": -95.3698, ... },
      "IMU": { "acceleration": -0.7, ... },
      "Ultrasonic": { "proximity_left": 3.5, ... }
    },
    "vehicle_state": { "lane": "left", "right_lane_status": "motorcycle_lane_splitting", ... },
    "environment": { "weather": "clear", "road_condition": "dry", ... },
    "action": "maintain_speed_slight_left_offset"
  },
  "output": "The vehicle maintained speed but shifted slightly left..."
}
```

---

## 🛠️ Fine-tuning Process

We fine-tuned **Meta LLaMA 3.2B** using [Unsloth](https://github.com/unslothai/unsloth) for optimized, memory-efficient training.

### 🔹 Fine-tuning scripts

- `app.py`: Contains simple fine-tuning loop using one-shot learning with structured prompts.
- `main.py`: Handles full model training using the 3000-sample dataset in IIO format.

### ⚙️ Fine-tuning Configuration

| Parameter                  | Value                        |
|----------------------------|------------------------------|
| Model                      | Meta LLaMA 3.2B             |
| Dataset size               | 3000 samples                |
| Max sequence length        | 4096 tokens                 |
| LoRA (Low Rank Adaptation) | Enabled (r=16, alpha=16)    |
| Training epochs            | 3                           |
| Optimizer                  | AdamW                       |
| Learning rate              | 2e-4                        |
| Batch size                 | 1 (gradient accumulation=8) |
| Gradient checkpointing     | Enabled                     |

---

## 📊 Example Prompt & Explanation

**Input Prompt (Auto-generated)**

```
[INST] Generate an explanation for why the autonomous vehicle took the specified action based on the sensor data provided.

Timestamp: 2024-04-03T17:55:41Z

Sensor Data:
- LiDAR: Detected objects: 4, Closest object distance: 78.4 ft
- Radar: Relative speed -22 mph (approaching object)
- Camera: Red motorcycle detected (medium visibility)
- Vehicle state: Left lane, motorcycle lane splitting on right
- Environment: Dusk, moderate traffic

Action Taken: maintain_speed_slight_left_offset [/INST]
```

**Generated Explanation (Output)**

> _The vehicle maintained speed but shifted slightly left within its lane when detecting a red motorcycle lane splitting 78.4 ft ahead in the adjacent right lane. This defensive positioning created maximum space for the motorcycle (-22 mph relative speed) while maintaining safe following distance with the vehicle ahead, especially important at dusk when motorcycle visibility is reduced._

---

## 🚀 How to Fine-tune (Quick Start)

1️⃣ Install dependencies:
```bash
pip install torch transformers datasets trl unsloth
```

2️⃣ Set Hugging Face token (if required):
```bash
export HF_TOKEN=your_hf_token_here
```

3️⃣ Run training script:
```bash
python app.py  # for simple prompt fine-tuning (one-shot)
# OR
python main.py # for full dataset fine-tuning
```

4️⃣ Fine-tuned model saved at:
```
./av_explanation_model/
```

---

## 🧩 Inference with Fine-tuned Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_dir = "av_explanation_model"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)

prompt = "<your_formatted_prompt_here>"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 🏆 Why Fine-tuning Works

| **Base LLaMA** | **Fine-tuned LLaMA (Ours)** |
|----------------|-----------------------------|
| Generic text generation | Domain-specific explanations |
| Poor understanding of AV sensor data | Strong contextual reasoning with sensor fusion |
| Weak action-reason linkages | Clear, concise decision justifications |

---

## 🌟 Planned Feature (Coming Soon)

### 🔹 Feature Extraction Network (FEN)

We will integrate **Feature Extraction Networks** to:

- Extract **salient features** from raw sensor data (LiDAR, Radar, Camera)
- Improve model’s **multi-modal understanding**
- Enhance **explanation accuracy** and **interpretability**

This hybrid pipeline (FEN + LLM) aims to further bridge the gap between AV decision-making and human understanding — boosting trust, regulatory acceptance, and user confidence.

---

## 🤝 Acknowledgements

- [Meta LLaMA](https://ai.meta.com/llama/)
- [Unsloth](https://github.com/unslothai/unsloth)
- [Hugging Face Transformers](https://huggingface.co/transformers)

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 📢 Stay tuned!

> **Upcoming Release**  
> 🛠️ Feature Extraction Network (FEN) integration  
> 🖥️ REST API interface for real-time inference  
> 🧪 Expanded dataset with 10K+ autonomous driving scenarios

---

If you'd like — I can **also generate README badges** (Python version, license, model size, etc.) to make it even cleaner. Want that? 🚀

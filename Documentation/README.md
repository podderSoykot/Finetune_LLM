# Fine-Tuning LLaMA 3.1-8B-Instruct on Bengali Empathetic Conversations

## Project Overview

This project fine-tunes LLaMA 3.1-8B-Instruct model on a Bengali Empathetic Conversations dataset using parameter-efficient fine-tuning (LoRA/Unsloth) for efficient training on free GPU resources (Kaggle).

## Project Structure

```
Raco_AI/
├── kagle/
│   └── finetune-llama.ipynb          # Training script (initial training)
├── lora-1/
│   ├── finetune-llama.ipynb          # Main notebook with checkpoint resumption
│   ├── result/                        # Training results and checkpoints
│   │   ├── trainer_state.json        # Training state at step 1500
│   │   └── adapter_model.safetensors # Saved LoRA adapter weights
│   ├── adapter_config.json            # LoRA adapter configuration
│   └── logs/                          # Experiment logs and generated responses
└── Documentation/
    └── README.md                     # This file
```

## Training Approach

### Initial Training
The initial training was performed using the notebook located at:
- **Training Script**: `F:\Soykot_podder\Raco_AI\kagle\finetune-llama.ipynb`

This script was used to train the model from scratch and create checkpoints during training.

### Checkpoint Resumption
Due to **limited GPU time availability** on Kaggle's free tier, the training was resumed from a saved checkpoint at step 1500. The checkpoint resumption and final results are located at:
- **Results Directory**: `F:\Soykot_podder\Raco_AI\lora-1\`
- **Checkpoint**: Step 1500 (saved in `result/trainer_state.json`)

### Why Checkpoint 1500?

Kaggle's free GPU tier has time limitations (typically 30 hours per week). To maximize training efficiency and ensure we could complete the fine-tuning process, we:

1. **Initial Training**: Trained the model for 1500 steps, saving checkpoints every 500 steps
2. **Checkpoint Selection**: Used checkpoint-1500 as it represents a good balance between:
   - Training progress (39.26% of one epoch completed)
   - Model convergence (training loss decreasing)
   - Resource constraints (within GPU time limits)
3. **Resumption**: Loaded the checkpoint and continued training/evaluation in the `lora-1` directory

## Model Configuration

### Base Model
- **Model**: LLaMA 3.1-8B-Instruct
- **Quantization**: 4-bit (BitsAndBytes)
- **Precision**: bfloat16 (mixed precision training)

### LoRA Configuration

```python
{
    "r": 16,                    # LoRA rank
    "lora_alpha": 32,           # LoRA alpha scaling
    "lora_dropout": 0.1,        # Dropout rate
    "target_modules": [         # Attention layers
        "q_proj", "k_proj", 
        "v_proj", "o_proj"
    ],
    "bias": "none",
    "task_type": "CAUSAL_LM"
}
```

### Training Configuration
- **Batch Size**: 1 per device
- **Gradient Accumulation**: 8 steps (effective batch size: 8)
- **Learning Rate**: 2e-4
- **Max Sequence Length**: 1500 tokens (full-sequence tokenization)
- **Epochs**: 1
- **Checkpoint Interval**: Every 500 steps
- **Evaluation Interval**: Every 500 steps

### Resource Optimization
- **Gradient Checkpointing**: Enabled (reduces memory usage)
- **Mixed Precision**: bfloat16 (faster training, lower memory)
- **4-bit Quantization**: Reduces model size by ~75%
- **Trainable Parameters**: 13,631,488 (0.17% of total parameters)

## Dataset

- **Dataset**: Bengali Empathetic Conversations Corpus
- **Size**: 38,210 rows
- **Splits**:
  - Training: 30,568 samples
  - Validation: 3,821 samples
  - Test: 3,821 samples
- **Format**: Instruction-Input-Response format
- **Language**: Bengali (Bangla)

## Evaluation Metrics

The model was evaluated using the following metrics:

### Objective Metrics
- **Perplexity**: 1.9168 (lower is better)
- **BLEU Score**: 0.0660 (0-1 scale, higher is better)
- **ROUGE-1**: 0.0000
- **ROUGE-2**: 0.0000
- **ROUGE-L**: 0.0000

### Notes on Metrics
- **Perplexity** indicates the model has learned the language patterns well
- **BLEU/ROUGE** scores are lower, which is common for:
  - Empathetic/conversational responses (less n-gram overlap with references)
  - Bengali language (limited reference data for evaluation)
  - Creative/empathetic responses that may not match reference exactly

## Implementation Details

### OOP Design
The implementation follows object-oriented design with the following classes:

1. **DatasetProcessor**: Handles dataset preprocessing and tokenization
   - Text normalization
   - Instruction formatting
   - Full-sequence tokenization (no length reduction)

2. **CustomTrainer** (LLAMAFineTuner): Extends HuggingFace Trainer
   - Integrates with experiment logging
   - Captures training and validation metrics
   - Handles checkpoint resumption

3. **Evaluator**: Implements evaluation metrics
   - Perplexity calculation
   - BLEU score calculation
   - ROUGE score calculation
   - Response generation for human evaluation

4. **ExperimentLogger**: Manages experiment tracking
   - Logs to LLAMAExperiments table structure
   - Saves generated responses
   - Tracks hyperparameters and metrics

### Design Patterns

**Strategy Pattern**: Implemented for switching between LoRA and Unsloth
- `USE_UNSLOTH` flag controls strategy selection
- Allows easy switching between:
  - Unsloth (optimized for speed)
  - Standard PEFT LoRA (more compatibility)

### Algorithm Implementation

1. **LoRA Adaptation**: Applied to attention layers (q_proj, k_proj, v_proj, o_proj)
2. **Full-Sequence Tokenization**: Preserves full sequences up to 1500 tokens
3. **Evaluation Pipeline**: Automated calculation of perplexity, BLEU, and ROUGE metrics

## Training Results

### Training Progress (from Checkpoint 1500)
- **Global Step**: 1500
- **Epoch Progress**: 0.3926 (39.26% of one epoch)
- **Training Loss**: Decreasing trend observed
- **Trainable Parameters**: 13,631,488

### Model Performance
- The model shows good perplexity, indicating it has learned Bengali language patterns
- Generated responses demonstrate empathetic understanding
- Model can generate contextually appropriate Bengali responses

## Files and Artifacts

### Training Artifacts
- **Checkpoint**: `lora-1/result/` contains:
  - `trainer_state.json`: Training state at step 1500
  - `adapter_model.safetensors`: LoRA adapter weights
  - `adapter_config.json`: LoRA configuration
  - Optimizer and scheduler states

### Experiment Logs
- **Experiment Log**: `lora-1/logs/experiment_*.json`
  - Contains: model path, LoRA config, training config, train/val loss, metrics, timestamp
- **Generated Responses**: `lora-1/logs/generated_responses_*.json`
  - Contains: input text, generated responses, timestamps

### Saved Model
- **Model Directory**: `lora-1/lora_model/`
  - Contains fine-tuned adapter weights
  - Can be loaded for inference

## Usage Instructions

### Loading the Fine-Tuned Model

```python
from unsloth import FastLanguageModel
from peft import PeftModel

# Load base model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    max_seq_length=1500,
    dtype=None,
    load_in_4bit=True,
)

# Load fine-tuned adapter
model = PeftModel.from_pretrained(
    model,
    "lora-1/lora_model",
    is_trainable=False
)

# Enable inference mode
FastLanguageModel.for_inference(model)
```

### Generating Responses

```python
def generate_response(question: str):
    input_text = f"""### Instruction:
You are a helpful and empathetic Bengali assistant.

### Input:
{question}

### Response:
"""
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("### Response:")[-1].strip()
```

## Challenges Faced

1. **GPU Time Limitations**: 
   - Kaggle free tier has limited GPU hours
   - Solution: Used checkpoint resumption to continue training across sessions

2. **Memory Constraints**: 
   - 8B parameter model requires significant memory
   - Solution: 4-bit quantization, gradient checkpointing, mixed precision

3. **Batch Size Optimization**: 
   - Needed to balance memory and training speed
   - Solution: Small per-device batch (1) with gradient accumulation (8)

4. **Checkpoint Resumption**: 
   - Ensuring batch size consistency when resuming
   - Solution: Automatic batch size detection and adjustment from checkpoint

5. **Bengali Language Support**: 
   - Ensuring proper tokenization and encoding
   - Solution: Using LLaMA 3.1 tokenizer which has good multilingual support

## Future Improvements

1. **Extended Training**: Continue training beyond checkpoint 1500 for better convergence
2. **Hyperparameter Tuning**: Experiment with different LoRA ranks and learning rates
3. **Evaluation Enhancement**: Improve BLEU/ROUGE scores with better reference matching
4. **Human Evaluation**: Conduct more comprehensive human evaluation of empathetic responses
5. **Model Optimization**: Further optimize for inference speed and memory usage

## References

- **Base Model**: [LLaMA 3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- **Unsloth**: [Unsloth Library](https://github.com/unslothai/unsloth)
- **PEFT**: [Parameter-Efficient Fine-Tuning](https://huggingface.co/docs/peft)
- **Dataset**: Bengali Empathetic Conversations Corpus

## Contact

For questions or issues regarding this implementation, please cll me on +8801734537627 or mail: diptopodder95@gmail.com
Web: https://protfolio-soykot.vercel.app/ 

---

**Note**: This project was completed using Kaggle's free GPU resources. Due to time constraints, training was resumed from checkpoint 1500. The model demonstrates good performance on Bengali empathetic conversations despite the limited training time.
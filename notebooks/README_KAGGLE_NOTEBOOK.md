# Kaggle Training Notebook Guide

## File: `03_fine_tuning_kaggle.ipynb`

This notebook is designed to run on Kaggle's free GPU environment for fine-tuning LLaMA 3.1-8B-Instruct on Bengali Empathetic Conversations.

## Features

✅ **Unsloth + LoRA** for efficient fine-tuning  
✅ **Full-sequence tokenization** (no sequence length reduction)  
✅ **Experiment logging** (LLAMAExperiments table structure)  
✅ **Evaluation metrics**: Perplexity, BLEU, ROUGE  
✅ **Generated responses logging** (GeneratedResponses table structure)  
✅ **Memory optimization**: 4-bit quantization, gradient checkpointing, mixed precision  

## How to Use on Kaggle

### Step 1: Upload Dataset
1. Go to Kaggle and create a new notebook
2. Add the Bengali Empathetic Conversations dataset:
   - Search for: "Bengali Empathetic Conversations Corpus"
   - Or upload your own dataset
   - Add it as a data source to your notebook

### Step 2: Configure Notebook
1. Enable GPU: Settings → Accelerator → GPU T4 (or P100)
2. Set Internet: ON (for downloading models)
3. Copy the notebook code to your Kaggle notebook

### Step 3: Update Dataset Path
In **Section 4: Load Dataset**, update the dataset loading code:

```python
# Option 1: From Kaggle dataset
dataset = load_dataset("csv", data_files="/kaggle/input/bengali-empathetic-conversations/train.csv")

# Option 2: From HuggingFace
dataset = load_dataset("raseluddin/bengali-empathetic-conversations")
```

### Step 4: Run the Notebook
1. Run all cells sequentially
2. Monitor training progress
3. Check logs in `./logs/` directory
4. Model will be saved to `lora_model/`

## Configuration Options

### Switch Between LoRA Methods
```python
USE_UNSLOTH = True   # Use Unsloth (recommended for Kaggle)
USE_UNSLOTH = False  # Use standard PEFT LoRA
```

### Adjust Training Parameters
Edit the `TRAINING_CONFIG` dictionary:
- `num_train_epochs`: Number of training epochs
- `per_device_train_batch_size`: Batch size (keep at 1 for memory)
- `gradient_accumulation_steps`: Effective batch size
- `learning_rate`: Learning rate (default: 2e-4)

### Adjust LoRA Parameters
Edit the `LORA_CONFIG` dictionary:
- `r`: LoRA rank (default: 16)
- `lora_alpha`: LoRA alpha (default: 32)
- `target_modules`: Which layers to apply LoRA

## Output Files

After training, you'll have:

1. **Model**: `lora_model/` - Fine-tuned model and tokenizer
2. **Experiment Log**: `./logs/experiment_YYYYMMDD_HHMMSS.json`
   - Contains: train_loss, val_loss, metrics, configs
3. **Generated Responses**: `./logs/generated_responses_YYYYMMDD_HHMMSS.json`
   - Contains: input_text, response_text, experiment_id

## Memory Management

If you encounter OOM (Out of Memory) errors:

1. **Reduce max_length**: Set `DATASET_CONFIG["max_length"]` to 1024 or 512
2. **Increase gradient accumulation**: Increase `gradient_accumulation_steps`
3. **Reduce LoRA rank**: Set `LORA_CONFIG["r"]` to 8 or 4
4. **Use smaller batch**: Keep `per_device_train_batch_size` at 1

## Evaluation Metrics

The notebook automatically calculates:
- **Perplexity**: Lower is better (measures model confidence)
- **BLEU Score**: Higher is better (n-gram overlap)
- **ROUGE-1, ROUGE-2, ROUGE-L**: Higher is better (recall-oriented)

## Important Notes

⚠️ **Sequence Length**: The notebook preserves full sequences. Only truncates if absolutely necessary (when sequence > max_length).

⚠️ **HuggingFace Token**: You may need to set your HuggingFace token:
```python
import os
os.environ["HF_TOKEN"] = "your_token_here"
```

⚠️ **Model Access**: LLaMA 3.1 requires HuggingFace access. Make sure you have:
- Accepted the model license on HuggingFace
- Set your HF token in Kaggle secrets or environment

## Troubleshooting

### Issue: Model download fails
**Solution**: Set HuggingFace token in Kaggle secrets

### Issue: Out of Memory
**Solution**: Reduce max_length, increase gradient accumulation, or use smaller LoRA rank

### Issue: Training is slow
**Solution**: This is normal for 8B models. Training can take several hours.

### Issue: Dataset format error
**Solution**: Check dataset structure. It should have 'context' and 'response' columns.

## Next Steps

After training:
1. Download the model from `lora_model/`
2. Review experiment logs
3. Analyze evaluation metrics
4. Generate more sample responses
5. Create documentation and video explanation

## Support

For issues or questions:
- Check the PROJECT_PLAN.md for detailed architecture
- Review HuggingFace Transformers documentation
- Check Unsloth GitHub: https://github.com/unslothai/unsloth





# Instructions: Resuming Training on Kaggle

## Your Setup

You've uploaded your checkpoint files to Kaggle at:
```
/kaggle/input/models/diptopodder/checkpoint-1500/pytorch/default/1
```

## Files Available

✅ All necessary files are present:
- `adapter_config.json` - LoRA configuration
- `adapter_model.safetensors` - Trained adapter weights (up to step ~1242)
- `optimizer.pt` - Optimizer state
- `scheduler.pt` - Learning rate scheduler state
- `training_args.bin` - Training arguments
- `rng_state.pth` - Random number generator state
- `tokenizer.json` & `tokenizer_config.json` - Tokenizer files

## How to Resume Training

### Step 1: Run the Notebook Cells in Order

1. **Cells 1-10**: Setup, configuration, dataset loading, model initialization
   - These should run normally

2. **Cell 11.5 (NEW)**: Load Saved Adapter
   - This cell will:
     - Load your saved adapter from `/kaggle/input/models/diptopodder/checkpoint-1500/pytorch/default/1`
     - Restore the LoRA weights
     - Detect optimizer/scheduler state files
   - **Make sure `RESUME_FROM_SAVED = True`**

3. **Cell 11.6 (NEW)**: Create Checkpoint Structure
   - This cell will:
     - Create a checkpoint directory at `./results/checkpoint-1000/`
     - Copy all necessary files from your saved checkpoint
     - Create `trainer_state.json` with step information
   - **Make sure `CREATE_CHECKPOINT_FROM_SAVED = True`**
   - **Set `CHECKPOINT_STEP = 1000`** (your last saved checkpoint)

4. **Cell 12**: Training
   - The training cell now automatically detects checkpoints
   - It will find `./results/checkpoint-1000/` and resume from there
   - Training will continue from step 1000 to step 3821

### Step 2: Verify Paths

Before running, verify these paths in the notebook:

```python
# In Cell 11.5
SAVED_ADAPTER_PATH = "/kaggle/input/models/diptopodder/checkpoint-1500/pytorch/default/1"
RESUME_FROM_SAVED = True

# In Cell 11.6
CREATE_CHECKPOINT_FROM_SAVED = True
CHECKPOINT_STEP = 1000
```

### Step 3: Expected Behavior

When you run the training cell:

1. **Checkpoint Detection**:
   ```
   ============================================================
   FOUND EXISTING CHECKPOINT: ./results/checkpoint-1000
   Resuming from step 1000
   ============================================================
   ```

2. **Training Progress**:
   - Will start from step 1000 (not step 0)
   - Will continue until step 3821 (completing the epoch)
   - Will save checkpoints every 500 steps (1500, 2000, 2500, etc.)

3. **Training State**:
   - Optimizer momentum will be restored
   - Learning rate schedule will continue from where it left off
   - Model weights will be from step 1000

## Important Notes

### Step Number
- You stopped training at **step 1242**
- Your last checkpoint was saved at **step 1000**
- Training will resume from **step 1000** (242 steps earlier, but this is safe)
- This means you'll re-train steps 1000-1242, but with the saved weights as starting point

### What Gets Restored
- ✅ Model weights (LoRA adapter)
- ✅ Optimizer state (Adam momentum)
- ✅ Scheduler state (learning rate schedule)
- ✅ Training step counter (will resume from 1000)

### What Doesn't Get Restored
- ⚠️ Training loss history (will start fresh in logs)
- ⚠️ Exact step 1242 state (will resume from 1000)

## Troubleshooting

### If adapter loading fails:
1. Check that the path is correct: `/kaggle/input/models/diptopodder/checkpoint-1500/pytorch/default/1`
2. Verify the dataset is added to your Kaggle notebook
3. Check that `adapter_model.safetensors` exists in that path

### If checkpoint creation fails:
1. Make sure Cell 11.5 ran successfully first
2. Check that `SAVED_ADAPTER_PATH` is accessible
3. Verify all files exist in the source directory

### If training starts from step 0:
1. Check that Cell 11.6 created the checkpoint structure
2. Verify `./results/checkpoint-1000/` exists
3. Check that `trainer_state.json` is in the checkpoint directory

## Expected Training Time

- **Remaining steps**: ~2,821 steps (from 1000 to 3821)
- **At ~0.07 it/s**: ~11 hours
- **With Unsloth optimizations**: May be faster

## Success Indicators

You'll know it's working when:
1. Cell 11.5 shows: "✓ Saved adapter loaded successfully!"
2. Cell 11.6 shows: "✓ Checkpoint structure created at: ./results/checkpoint-1000"
3. Training cell shows: "FOUND EXISTING CHECKPOINT" and "Resuming from step 1000"
4. Training progress bar starts from step 1000, not step 0

Good luck completing your training! 🚀



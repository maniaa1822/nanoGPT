# train a larger character-level model on the Seven-State Human dataset
# Scaled up for better state discrimination and JS analysis

out_dir = 'out-seven-state-human-char-large'
eval_interval = 500
eval_iters = 200
log_interval = 50

always_save_checkpoint = True

wandb_log = False
wandb_project = 'seven-state-human-char'
wandb_run_name = 'gpt-large'

dataset = 'seven_state_human'
gradient_accumulation_steps = 1
batch_size = 256  # Reduced batch size for larger model
block_size = 32  # Increased context window for better long-range dependencies

# Larger model architecture for better state representation
n_layer = 8      # Doubled layers: 4 -> 8
n_head = 8       # Doubled heads: 4 -> 8
n_embd = 64     # 4x embedding: 32 -> 64
dropout = 0.1

learning_rate = 3e-4  # Lower LR for larger model
max_iters = 10000     # More training iterations
lr_decay_iters = 10000
min_lr = 3e-5         # Lower minimum LR
beta2 = 0.95          # More aggressive momentum

warmup_iters = 500    # Longer warmup

# Enable compilation for performance
compile = True

# Optimization settings for larger model
weight_decay = 0.1
grad_clip = 1.0

# Use mixed precision if available
dtype = 'bfloat16'  # Enable mixed precision training
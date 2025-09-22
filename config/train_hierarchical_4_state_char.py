# train a miniature character-level model on the Hierarchical 4-State binary sequence

out_dir = 'out-hierarchical-4-state-char'
eval_interval = 250
eval_iters = 200
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = 'hierarchical-4-state-char'
wandb_run_name = 'mini-gpt'

dataset = 'hierarchical_4_state'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 32  # context length

# baby GPT model (keep comparable to seven state config)
n_layer = 2
n_head = 2
n_embd = 16
dropout = 0.2

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100

# Optional overrides for CPU-only runs
# device = 'cpu'
# compile = False

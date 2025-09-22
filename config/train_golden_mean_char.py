# train a miniature character-level model on the Golden Mean binary sequence

out_dir = 'out-golden-mean-char'
eval_interval = 250
eval_iters = 200
log_interval = 10

# only save when val improves (tiny dataset likely to overfit)
always_save_checkpoint = False

wandb_log = False
wandb_project = 'golden-mean-char'
wandb_run_name = 'mini-gpt'

dataset = 'golden_mean'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 32  # context of up to 256 previous bits

# baby GPT model
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




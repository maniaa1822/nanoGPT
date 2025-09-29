# longer run on the corrected Seven-State Human dataset (50k)

out_dir = 'out-seven-state-human-char_50k_long'
eval_interval = 500
eval_iters = 200
log_interval = 10

always_save_checkpoint = True

wandb_log = False
wandb_project = 'seven-state-human-char'
wandb_run_name = 'gpt-mini-50k-long'

dataset = 'seven_state_human'
gradient_accumulation_steps = 1
batch_size = 512
block_size = 32

n_layer = 4
n_head = 4
n_embd = 32
dropout = 0.2

learning_rate = 1e-3
max_iters = 20000
lr_decay_iters = 20000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 200

# device = 'cpu'
# compile = False




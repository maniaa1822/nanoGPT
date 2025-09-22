# train a small char-level model on complex-csm.dat

out_dir = 'out-complex-csm-char'
eval_interval = 250
eval_iters = 200
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = 'complex-csm-char'
wandb_run_name = 'mini-gpt'

dataset = 'complex_csm'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100

# device = 'cpu'
# compile = False




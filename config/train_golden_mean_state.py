# train a small char-level model on golden_mean with auxiliary state head

out_dir = 'out-golden-mean-state'
eval_interval = 250
eval_iters = 200
log_interval = 10

always_save_checkpoint = True

wandb_log = False
wandb_project = 'golden-mean-char'
wandb_run_name = 'gm-state-head'

dataset = 'golden_mean'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 32

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

# Auxiliary state head config
state_head_classes = 2
state_loss_weight = 0.5
state_labels_dat = '../experiments/datasets/golden_mean/golden_mean.states.dat'

# device can be overridden from CLI
# device = 'cpu'
# compile = False



from pathlib import Path
# -----------------------------
log_rotation = "512 KB"
log_path = Path('log/log.log')
# -----------------------------
data_name = 'mine'
model_name = 'resnet18'
optimizer_name = 'adam'
batch_size = 40
epoch = 100
learn_rate = 2e-4
train_set_path = Path('data/train-set')
test_set_path = Path('data/test-set')
# -----------------------------
model_save_path = Path('models/tmp')
best_model_save_path = Path('models/model_best.pth')

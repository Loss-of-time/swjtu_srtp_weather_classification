from pathlib import Path

model_name ='resnet18'
optimizer_name = 'adam'
batch_size = 40
epoch = 10
learn_rate = 5e-4
train_set_path = Path('data/train-set')
test_set_path = Path('data/test-set')
log_path = Path('log/log.log')
model_save_path = Path('models/tmp')
best_model_save_path = Path('models/model_best.pth')

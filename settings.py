from pathlib import Path

# -----------------------------
log_rotation = "128 KB"
log_path = Path("log/log.log")
# -----------------------------
data_name = "rscm"
model_name = "resnet18"
optimizer_name = "sgd"  # adam | sgd | rmsprop
batch_size = 100  # 6GB显存 & 224 x 224 px
epoch = 100
learn_rate = 1e-4
train_set_path = Path("data/train-set")
test_set_path = Path("data/test-set")
# -----------------------------
model_save_path = Path("models/tmp")
best_model_save_path = Path("models/model_best.pth")

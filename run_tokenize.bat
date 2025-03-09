@echo off
for %%f in (data/train/*.jpg) do (
    python -m homework.tokenize checkpoints/1.pth data/tokenized_train.pth "%%f"
)

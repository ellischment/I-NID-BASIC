import os

dirs = [
    'data/raw',
    'data/processed/clinc_oos_gamma3',
    'models/pretrained',
    'models/finetuned',
    'logs'
]

for dir_path in dirs:
    os.makedirs(dir_path, exist_ok=True)
    print(f"Created directory: {dir_path}")

print("\nDirectory structure ready!")
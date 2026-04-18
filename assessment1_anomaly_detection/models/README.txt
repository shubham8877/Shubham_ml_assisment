This folder is intentionally empty in the repository.

Model artifacts (.pkl, .pt) are generated here when you run:

    cd src && python train.py

They are excluded from git via .gitignore because:
  - PyTorch/sklearn binaries are large (20-100 MB each)
  - They should be regenerated from code, not version-controlled
  - Binary diffs are useless in git history

After running training you will see:
  models/
  ├── preprocessor.pkl      # Fitted sklearn ColumnTransformer
  ├── isolation_forest.pkl  # Trained IsolationForest model
  ├── autoencoder.pt        # Trained Autoencoder state dict
  └── threshold.pkl         # Tuned decision threshold + input dim

class Config:
    # Data params
    known_ratio = 0.75
    labeled_ratio = 0.1
    gamma_values = [3, 5, 10]

    # Model params
    batch_size = 512
    temperature = 0.07
    rho = 0.7
    tau_g = 0.9
    lambda1 = 0.05
    lambda2 = 2  # 7 for balanced datasets
    max_seq_length = 128

    # Training params
    num_epochs = 3
    learning_rate = 5e-5
    weight_decay = 0.01
    grad_clip = 1.0
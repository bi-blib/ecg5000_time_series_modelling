import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from helpers import *

def main():

    ### Load the data
    train, test = loadRawDataFromFile()
    X_train, y_train, X_val, y_val, X_test, y_test = cleanAndSplitRaw(train, test)

    y_train = np.array([1 if y > 0 else 0 for y in y_train])
    y_val = np.array([1 if y > 0 else 0 for y in y_val])
    y_test = np.array([1 if y > 0 else 0 for y in y_test])

    print("Train Signal Shape:", X_train.shape)
    print("Val Signal Shape:", X_val.shape)
    print("Test Signal Shape:", X_test.shape)
    print("Train Label Shape:", y_train.shape)
    print("Val Label Shape:", y_val.shape)
    print("Test Label Shape:", y_test.shape)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    train_loader, val_loader, test_loader = getLoaders(
        X_train_scaled=X_train_scaled,
        X_val_scaled=X_val_scaled,
        X_test_scaled=X_test_scaled,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test
    )

    x_batch, y_batch = next(iter(train_loader))

    print("Input shape:", x_batch.shape) #batch size, n of time steps
    print("Target shape:", y_batch.shape) #batch size, n of output labels

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    ### Initialize model and optimizer
    input_size = 1
    n_classes = len(np.unique(y_train))
    n_timesteps = X_train.shape[1] # to find the context_length

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Different models s/m/xLSTM
    ###Choose model
    opt=input("Choose the model you want to use(mLSTM/sLSTM/xLSTM):")
    model = chooseTypeLSTM(opt,input_size,n_classes,n_timesteps, device)

    ### Print number of trainable parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params} trainable parameters.")
    
    # Loss function
    criterion = getCriterion(y_train=y_train, device=device)

    ### Training loop
    train_losses, val_losses = performTrainingLoop(model, train_loader, val_loader, device, criterion)
    plotLossCurve(train_losses, val_losses, save_path="b_loss.png")

    ### Model Testing  (Best models m/s/xLSTM)
    best_model = chooseTypeLSTM(opt,input_size,n_classes,n_timesteps, device)
       
    best_model.load_state_dict(torch.load("best_model.pth")) #load best model (by val loss)

    # Scale X_test with the scaler fit on X_train (never re-fit on test
    # data), then reshape to (n_samples, seq_len, 1) same as the train set.
    X_test_tensor = torch.tensor(
        X_test_scaled.reshape(-1, X_test_scaled.shape[1], 1),
        device=device,
    )

    y_pred, y_true, all_logits = performTestingLoop(best_model, test_loader, device)

    ### Metrics
    class_labels = np.unique(y_train)
    performMetrics(y_true, y_pred, all_logits, class_labels, prefix="b", roc_suffix="auc")

if __name__ == "__main__":
    main()
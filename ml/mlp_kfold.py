def load_data(mimos, include_esp, data_loc):
    df_charge = {}
    df_dist = {}

    # Iterate through each mimo in the list
    for mimo in mimos:
        # Load charge data from CSV file and store in dictionary
        df_charge[mimo] = pd.read_csv(f"{data_loc}/{mimo}_charge_esp.csv")
        df_charge[mimo] = df_charge[mimo].drop(columns=["replicate"])
        
        # Option to include the ESP features
        include_esp = include_esp.strip().lower()
        if include_esp not in ['t', 'true', True]:
            df_charge[mimo].drop(columns=["upper", "lower"], inplace=True)

        # Load distance data from CSV file and store in dictionary
        df_dist[mimo] = pd.read_csv(f"{data_loc}/{mimo}_pairwise_distance.csv")
        df_dist[mimo] = df_dist[mimo].drop(columns=["replicate"])

    return df_charge, df_dist


def validate(model, dataloader, device):

    val_loss = []
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for batch in dataloader:
            X, y = batch
            X = X.to(device)
            y = y.to(device)

            # validate your model on each batch here
            y_pred = model(X)
            loss = torch.nn.functional.cross_entropy(y_pred.squeeze(), y)
            val_loss.append(loss.item())

    loss = np.array(val_loss).mean()

    return loss


def train(feature, layers, lr, n_epochs, l2, train_dataloader, val_dataloader, device):

    mlp_cls = {}
    train_loss_per_epoch = {}
    val_loss_per_epoch = {}

    # Train MLP classifiers for each feature
    print("> Training MLP for " + feature + " features:\n")
    print("+-------+------------+----------+")
    print("| Epoch | Train-loss | Val-loss |")
    print("+-------+------------+----------+")

    val_losses = []
    train_losses = []
    model = MimoMLP(layers[feature]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    for epoch in range(n_epochs):
        # Train model on training data
        epoch_loss = gradient_step(
            model, train_dataloader[feature], optimizer, device=device
        )

        # Validate model on validation data
        val_loss = validate(model, val_dataloader[feature], device=device)

        # Record train and loss performance
        train_losses.append(epoch_loss)
        val_losses.append(val_loss)
        print(f" {epoch:.4f}     {epoch_loss:.4f}      {val_loss:.4f}")

    mlp_cls[feature] = model
    train_loss_per_epoch[feature] = train_losses
    val_loss_per_epoch[feature] = val_losses

    return mlp_cls, train_loss_per_epoch, val_loss_per_epoch


def evaluate_model(feature, mlp_cls, test_dataloader, device, mimos):

    y_pred_proba = {}
    y_pred = {}
    y_true = {}
    test_loss = {}
    cms = {}

    mlp_cls[feature].eval()
    y_true_feature_specific = np.empty((0, 3))
    y_pred_proba_feature_specific = np.empty((0, 3))
    y_pred_feature_specific = np.empty(0)
    losses = []
    with torch.no_grad():
        for batch in test_dataloader[feature]:
            X, y = batch
            X = X.to(device)
            y = y.to(device)
            logits = mlp_cls[feature](X)
            losses.append(torch.nn.functional.cross_entropy(logits, y))

            y_true_feature_specific = np.vstack(
                (y_true_feature_specific, y.detach().cpu().numpy())
            )
            y_proba = (
                torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy()
            )
            y_pred_proba_feature_specific = np.vstack(
                (y_pred_proba_feature_specific, y_proba)
            )
            y_pred_feature_specific = np.hstack(
                (
                    y_pred_feature_specific,
                    logits.argmax(dim=1).detach().cpu().numpy(),
                )
            )

        y_pred_proba[feature] = y_pred_proba_feature_specific
        y_pred[feature] = y_pred_feature_specific
        y_true[feature] = y_true_feature_specific
        print(f"   > Mean test loss for {feature} MLP model: {np.array(losses).mean():.4f}")
        test_loss[feature] = np.array(losses).mean()

        y_true_feature_specific = np.argmax(y_true_feature_specific, axis=1)
        cm = confusion_matrix(y_true_feature_specific, y_pred_feature_specific)
        cms[feature] = pd.DataFrame(cm, mimos, mimos)

    return test_loss, y_true, y_pred_proba, y_pred, cms


class MimoMLP(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()

        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)



def build_dataloaders(data_split):

    train_sets = {
        feature: MDDataset(
            data_split[feature]["X_train"], data_split[feature]["y_train"]
        )
        for feature in data_split.keys()
    }
    val_sets = {
        feature: MDDataset(data_split[feature]["X_val"], data_split[feature]["y_val"])
        for feature in data_split.keys()
    }
    test_sets = {
        feature: MDDataset(data_split[feature]["X_test"], data_split[feature]["y_test"])
        for feature in data_split.keys()
    }
    train_loader = {
        feature: torch.utils.data.DataLoader(
            train_sets[feature], batch_size=256, shuffle=True
        )
        for feature in data_split.keys()
    }
    val_loader = {
        feature: torch.utils.data.DataLoader(
            val_sets[feature], batch_size=256, shuffle=True
        )
        for feature in data_split.keys()
    }
    test_loader = {
        feature: torch.utils.data.DataLoader(
            test_sets[feature], batch_size=256, shuffle=True
        )
        for feature in data_split.keys()
    }

    return train_loader, val_loader, test_loader


def run_mlp(data_split_type, include_esp, n_epochs):

    # Get datasets
    format_plots()
    mimos = ["mc6", "mc6s", "mc6sa"]
    data_loc = os.getcwd()
    df_charge, df_dist = load_data(mimos, include_esp, data_loc)
    plot_data(df_charge, df_dist, mimos)

    # Preprocess the data and split into train, validation, and test sets
    data_split, df_dist, df_charge = preprocess_data(df_charge, df_dist, mimos, data_split_type)
    # data_split, df_dist, df_charge = preprocess_data(df_charge, df_dist, mimos, data_split_type, val_frac=0.75, test_frac=0.875)

    # Build the train, validation, and test dataloaders
    train_loader, val_loader, test_loader = build_dataloaders(data_split)

    # Get input sizes for each dataset and build model architectures
    n_dist = data_split['dist']['X_train'].shape[1]
    n_charge = data_split['charge']['X_train'].shape[1]
    
    layers = {'dist': (torch.nn.Linear(n_dist, 155), torch.nn.ReLU(), 
                    torch.nn.Linear(155, 155), torch.nn.ReLU(), 
                    torch.nn.Linear(155, 155), torch.nn.ReLU(), 
                    torch.nn.Linear(155, 3)),
        'charge': (torch.nn.Linear(n_charge, 185), torch.nn.ReLU(), 
                    torch.nn.Linear(185, 185), torch.nn.ReLU(), 
                    torch.nn.Linear(185, 185), torch.nn.ReLU(), 
                    torch.nn.Linear(185, 3))
        }
    
    # Distance hyperparameters
    lr = 0.0012318
    l2 = 0.0003574
    mlp_cls_dist, train_loss_per_epoch_dist, val_loss_per_epoch_dist = train("dist", layers, lr, n_epochs, l2, train_loader, val_loader, 'cpu')

    # Charge hyperparameters
    lr = 0.0002025
    l2 = 0.0027472
    mlp_cls_charge, train_loss_per_epoch_charge, val_loss_per_epoch_charge = train("charge", layers, lr, n_epochs, l2, train_loader, val_loader, 'cpu')

    # Combine the results back together for efficient analysis
    mlp_cls = {**mlp_cls_dist, **mlp_cls_charge}
    train_loss_per_epoch = {**train_loss_per_epoch_dist, **train_loss_per_epoch_charge}
    val_loss_per_epoch = {**val_loss_per_epoch_dist, **val_loss_per_epoch_charge}
    plot_train_val_losses(train_loss_per_epoch, val_loss_per_epoch)
    
    # Evaluate model on test data
    test_loss, y_true_dist, y_pred_proba_dist, y_pred, cms_dist = evaluate_model("dist", mlp_cls, test_loader, 'cpu', mimos)
    test_loss, y_true_charge, y_pred_proba_charge, y_pred, cms_charge = evaluate_model("charge", mlp_cls, test_loader, 'cpu', mimos)

    # Combine values back together
    y_true = {**y_true_dist, **y_true_charge}
    y_pred_proba = {**y_pred_proba_dist, **y_pred_proba_charge}
    cms = {**cms_dist, **cms_charge}

    # Plot ROC-AUC curves, confusion matrices and SHAP dot plots
    plot_roc_curve(y_true, y_pred_proba, mimos)
    plot_confusion_matrices(cms, mimos)
    shap_analysis(mlp_cls, train_loader, test_loader, val_loader, df_dist, df_charge, mimos)

    # Clean up the newly generated files
    mlp_dir = "MLP"
    # Create the "rf/" directory if it doesn't exist
    if not os.path.exists(mlp_dir):
        os.makedirs(mlp_dir)

    # Move all files starting with "rf_" into the "rf/" directory
    for file in os.listdir():
        if file.startswith("mlp_"):
            shutil.move(file, os.path.join(mlp_dir, file))

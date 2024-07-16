import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

import tqdm
import scipy


def pearson_r2_metric(y_true, y_pred):
    return scipy.stats.pearsonr(y_true, y_pred)[0] ** 2
    

def get_kmers(seq, k=6, stride=1):
    return [seq[i:i + k] for i in range(0, len(seq), stride) if i + k <= len(seq)]


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )



def train_val_test_split(X, y, attention_mask=None, train=0.8, val=0.1, batch_size=128):
    if attention_mask is None:
        attention_mask = torch.ones_like(X)

    dataset = TensorDataset(X, attention_mask, y)

    # Define sizes for train, validation, test splits
    train_size = int(train * len(dataset))
    val_size = int(val * len(dataset))
    test_size = len(dataset) - train_size - val_size

    # Split dataset into train, validation, test sets
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Create dataloaders for each set
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=128)
    test_dataloader = DataLoader(test_dataset, batch_size=128)

    return train_dataloader, val_dataloader, test_dataloader


def train_model(train_loader,
                val_loader,
                model,
                criterion,
                lr=1e-4,
                n_epochs=10,
                device="cuda",
                early_stopping_patience=None,
                verbose=False):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    best_model_weights = model.state_dict()
    no_improvement_count = 0

    for epoch in range(n_epochs):
        model.train()
        for i, batch in tqdm.tqdm(enumerate(train_loader)):
            inputs, attention_mask, labels = batch
            #inputs, attention_mask, labels = inputs.to(device), attention_mask.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs, attention_mask=attention_mask).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                inputs, attention_mask, labels = batch
                #inputs, attention_mask, labels = inputs.to(device), attention_mask.to(device), labels.to(device)
                outputs = model(inputs, attention_mask=attention_mask).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = model.state_dict()
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if verbose:
            print("Epoch: {}, Training Loss: {}, Validation Loss: {}".format(epoch+1, loss.item(), val_loss))

        if early_stopping_patience is not None and no_improvement_count >= early_stopping_patience:
            print("Stopping early after {} epochs with no improvement in validation loss.".format(early_stopping_patience))
            model.load_state_dict(best_model_weights)
            break



def train(model, optimizer, loss_fn, num_epochs, train_dataloader, val_loader):
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_train_loss = 0.0
        for i, batch in tqdm.tqdm(enumerate(train_dataloader)):
            input_ids, attention_mask, labels = batch

            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask).squeeze()  # Shape: [batch_size]
            loss = loss_fn(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)

        # Print training statistics
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")


def test(model, loss_fn, test_dataloader):
    model.eval()  # Set the model to evaluation mode

    all_predictions = []
    all_labels = []
    total_test_loss = 0.0

    with torch.no_grad():  # Disable gradient computation
        for batch in test_dataloader:
            input_ids, attention_mask, labels = batch

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask).squeeze()  # Shape: [batch_size]

            # Compute loss
            test_loss = loss_fn(outputs, labels)
            total_test_loss += test_loss.item()

            # Store predictions and labels
            all_predictions.append(outputs)
            all_labels.append(labels)

    # Concatenate the results
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)

    avg_test_loss = total_test_loss / len(test_dataloader)

    print(f"Test Loss: {avg_test_loss:.4f}")

    return all_predictions, all_labels

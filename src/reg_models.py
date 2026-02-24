import torch
from torch import nn

class Regressor(nn.Module):
    def __init__(self, n_exp, n_neuromodul,hidden_dim=64):
        super().__init__()
        self.attention = nn.Linear(n_exp, 1)
        self.head = nn.Sequential(
            nn.Linear(n_exp,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,n_neuromodul),
            nn.Softplus()
        )

        def forward(self,x):
            w_att = self.attention(x).softmax(dim=1)
            x = (x*w_att).sum(dim=1)
            return self.head(x)

def train_regressor(model, train_data_loader,val_loader, loss, optimizer, device="cpu", num_epochs=100):
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        inputs, targets = inputs.to(device), targets.to(device)
        model.train()
        run_loss = 0.0

        for inputs, targets in train_data_loader:
            optimizer.zero_grad()
            out = model(inputs)
            l = loss(out,targets)
            l.backward()
            optimizer.step()

            run_loss += l.item()*inputs.size(0)
        
        ep_train_loss = run_loss/len(train_data_loader.dataset)
        train_losses.append(ep_train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs,targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                out = model(inputs)
                val_loss += loss(out,targets).item()*inputs.size(0)

            ep_val_loss = val_loss/len(val_loader.dataset)
            val_losses.append(ep_val_loss)
        if (epoch+1)%10 == 0:
            print(f"Epoch {epoch}/{num_epochs}| Train Loss: {ep_train_loss:.4f}| Val Loss: {ep_val_loss:.4f}")
    return train_losses

def test_regressor(model, test_data_loader, loss, device='cpu'):
    model.eval()
    test_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for inputs, targets in test_data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            out = model(inputs)

            l = loss(out, targets)
            test_loss += l.item() * inputs.size(0)

            all_preds.append(out.cpu())
            all_targets.append(targets.cpu())
        
    avg_loss = test_loss/len(test_data_loader)
    avg_loss = test_loss / len(test_data_loader.dataset)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    mae = (all_preds - all_targets).abs().mean(dim=0)

    return avg_loss, mae, all_preds, all_targets

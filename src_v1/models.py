import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau


class DynamixClassifier(nn.Module):
    def __init__(self, dynamix_model, n_classes=4,  freeze_exp=True, freeze_gating=False):
        self.dynamix = dynamix_model

        if freeze_exp:
            for param in self.dynamix.experts.parameters():
                param.required_grad=False

        if freeze_gating:
            for param in self.dynamix.gating_network.parameters():
                param.required_grad=False
        n_exp_params = int(len([i for i in self.dynamix.experts.parameters()])/3)

        self.n_exp = n_exp_params
        pass


class Regressor(nn.Module):
    def __init__(self, n_exp, n_neuromodul,hidden_dim1=64, hidden_dim2=128, pooling='attention'):
        super().__init__()
        self.pool = pooling

        input_dim = n_exp if pooling != 'none' else n_exp*1000

        self.attention = nn.Linear(n_exp, 1)
        self.head = nn.Sequential(
            nn.Linear(input_dim,hidden_dim1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim1,hidden_dim2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim2, n_neuromodul)#,
            #nn.Softplus()
        )

    def forward(self, x):
        if self.pool == 'attention':
            w_att = self.attention(x).softmax(dim=1)
            x = (x*w_att).sum(dim=1)
        elif self.pool == 'mean':
            x = x.mean(dim=1)
        elif self.pool == 'none':
            x = x.reshape(x.size(0),-1)

        return self.head(x)

def train_regressor(model, train_data_loader,val_loader, loss, optimizer, device="cpu", num_epochs=100):
    train_losses, val_losses = [], []

    scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5, patience=10)

    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        run_loss = 0.0

        for inputs, targets in train_data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
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
            scheduler.step(ep_val_loss)

        if (epoch+1)%10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}| Train Loss: {ep_train_loss:.4f}| Val Loss: {ep_val_loss:.4f}")

    return train_losses

def test_regressor(model, test_data_loader, loss, y_mean, y_std, device='cpu'):
    model.to(device)
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

    y_mean = torch.tensor(y_mean, dtype=torch.float32)
    y_std  = torch.tensor(y_std,  dtype=torch.float32)

    all_preds_nM   = all_preds   * y_std + y_mean
    all_targets_nM = all_targets * y_std + y_mean
    mae_nM = (all_preds_nM - all_targets_nM).abs().mean(dim=0)

    return avg_loss, mae, mae_nM, all_preds_nM, all_targets_nM

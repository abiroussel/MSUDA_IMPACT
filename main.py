import sys
import numpy as np
from sklearn.metrics import f1_score
from torch import nn
import torch
from torch.utils.data import DataLoader, TensorDataset
from pytorch_metric_learning import losses
from model import XModelv5
from functions import cumulate_EMA, computeMaxMin, rescale

def evaluation(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        tot_pred = []
        tot_labels = []
        for data in dataloader:
            x_batch = data[0].to(device)
            y_batch = data[1].to(device)
            pred = model(x_batch)[0]
            pred_npy = np.argmax(pred.cpu().detach().numpy(), axis=1)
            tot_pred.append( pred_npy )
            tot_labels.append( y_batch.cpu().detach().numpy())
        tot_pred = np.concatenate(tot_pred)
        tot_labels = np.concatenate(tot_labels)
    return tot_pred, tot_labels

momentum_ema = 0.95
b_size = 128
years = [2020, 2021, 2022, 2023, 2024]
n_years = len(years)
test_year = int(sys.argv[1])
source_data = []
source_label = []
source_year_dom = []
test_data = None
test_label = None
year_c = 0
for y in years:
    if y == test_year:
        print(y)
        test_data = np.load("%d_data_s2.npy"%y)
        test_label = np.load("%d_labels.npy"%y)
    else:
        t = np.load("%d_data_s2.npy"%y)
        source_year_dom.append( np.ones(t.shape[0]) * year_c )
        source_data.append(t)
        source_label.append(np.load("%d_labels.npy"%y))
        year_c+=1

source_data = np.concatenate(source_data, axis=0)
source_label = np.concatenate(source_label, axis=0)
source_year_dom = np.concatenate(source_year_dom, axis=0)

x_min, x_max = computeMaxMin(np.concatenate([source_data, test_data],axis=0))
source_data = rescale(source_data, x_min, x_max)
test_data = rescale(test_data, x_min, x_max)

x_pretrain = torch.tensor(np.concatenate([source_data, test_data], axis=0), dtype=torch.float32)
pretrain_dataset = TensorDataset(x_pretrain)
pretrain_dataloader = DataLoader(pretrain_dataset, shuffle=True, batch_size=b_size)

x_source = torch.tensor(source_data, dtype=torch.float32)
x_source_year = torch.tensor(source_year_dom, dtype=torch.int64)
y_source = torch.tensor(source_label, dtype=torch.int64)

x_test = torch.tensor(test_data, dtype=torch.float32)
y_test = torch.tensor(test_label, dtype=torch.int64)


source_dataset = TensorDataset(x_source, y_source, x_source_year)
test_dataset = TensorDataset(x_test, y_test)

source_dataloader = DataLoader(source_dataset, shuffle=True, batch_size=b_size)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1024)

x_target_data = torch.tensor(test_data, dtype=torch.float32)
target_dataset = TensorDataset(x_target_data)
target_dataloader = DataLoader(target_dataset, shuffle=True, batch_size=b_size)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = XModelv5(nb_classes=len(np.unique(source_label)), n_doms=n_years).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
optimizerPre = torch.optim.AdamW(model.parameters(), lr=0.0001)
loss_supCon = losses.SupConLoss()

best_model = None
best_epoch = None
total_losses = []
metric_evolution = []
ema_weights = None

criterion = nn.CrossEntropyLoss()
for epoch in range(300):
    model.train()
    total_loss = 0.0
    iteration = 0
    for x_batch, y_batch, year_batch in source_dataloader:
        tgt_x_batch = next(iter(target_dataloader))[0]

        # Go forward to get preds
        optimizer.zero_grad()
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        year_batch = year_batch.to(device)
        tgt_x_batch = tgt_x_batch.to(device)

        pred_source, pred_source_dom, emb_source = model(x_batch)
        _, pred_tgt_dom, _  = model(tgt_x_batch)

        #Task loss
        loss_cl = criterion(pred_source, y_batch)

        # Adversarial (CDAN) loss
        pred_dom = torch.cat([pred_source_dom, pred_tgt_dom], dim=0).to(device)
        y_dom_tgt = torch.ones(pred_tgt_dom.shape[0], dtype=torch.int64).to(device) * (n_years-1)
        y_dom = torch.cat([year_batch, y_dom_tgt], dim=0)
        y_dom = y_dom.to(device)
        loss_adv = criterion(pred_dom, y_dom)

        #Supervised Contrastive Loss
        loss_con = loss_supCon(emb_source, y_batch)

        loss = loss_cl + loss_adv + loss_con
        total_loss+=loss.cpu().detach().numpy()

        # Backpropagation
        loss.backward()
        optimizer.step()
        iteration+=1
    
    current_state_dict = model.state_dict()
    if epoch > 100:
        ema_weights = cumulate_EMA(model, ema_weights, momentum_ema)
        model.load_state_dict(ema_weights)

    tot_pred, tot_labels = evaluation(model, test_dataloader, device)
    f1_val = f1_score(tot_labels, tot_pred, average=None)*100
    f1_val_str = ["%.2f"%el for el in f1_val]
    f1_val_str = " ".join(f1_val_str)
    print("epoch %d with loss %.4f F1 val per-Class %s"%(epoch, total_loss/iteration, f1_val_str))
    model.load_state_dict(current_state_dict)
    sys.stdout.flush()
# Set the random seed manually for reproducibility.
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import sys
from tcn import TCN
from Assignment2.jsb_datasets import load_jsb
import numpy as np

cuda = False
clip = 0.2

torch.manual_seed(1111)
if torch.cuda.is_available():
    if not cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        
input_size = 1
X_train, X_valid, X_test = load_jsb(as_tensor=True)

n_channels = [150] * 4
kernel_size = 5
dropout = 0.25

model = TCN(input_size, input_size, n_channels, kernel_size, dropout)

if cuda:
    model.cuda()

criterion = nn.MSELoss()
lr = 1e-3
optimizer = getattr(optim, 'Adam')(model.parameters(), lr=lr)


def evaluate(X_data, name='Eval'):
    model.eval()
    eval_idx_list = np.arange(len(X_data), dtype="int32")
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for idx in eval_idx_list:
            data_line = X_data[idx]
            x, y = Variable(data_line[:-1]), Variable(data_line[1:])
            if cuda:
                x, y = x.cuda(), y.cuda()
            output = model(x.unsqueeze(0)).squeeze(0)
            loss = -torch.trace(torch.matmul(y, torch.log(output).float().t()) +
                                torch.matmul((1-y), torch.log(1-output).float().t()))
            total_loss += loss.item()
            count += output.size(0)
        eval_loss = total_loss / count
        print(name + " loss: {:.5f}".format(eval_loss))
        return eval_loss


def train(ep):
    model.train()
    total_loss = 0
    count = 0
    train_idx_list = np.arange(len(X_train), dtype="int32")
    np.random.shuffle(train_idx_list)
    for idx in train_idx_list:
        data_line = X_train[idx]
        x, y = Variable(data_line[:-1]), Variable(data_line[1:])
        if cuda:
            x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        output = model(x.unsqueeze(0)).squeeze(0)
        loss = -torch.trace(torch.matmul(y, torch.log(output).float().t()) +
                            torch.matmul((1 - y), torch.log(1 - output).float().t()))
        total_loss += loss.item()
        count += output.size(0)

        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        loss.backward()
        optimizer.step()
        if idx > 0 and idx % 100 == 0:
            cur_loss = total_loss / count
            print("Epoch {:2d} | lr {:.5f} | loss {:.5f}".format(ep, lr, cur_loss))
            total_loss = 0.0
            count = 0


if __name__ == "__main__":
    best_vloss = 1e8
    vloss_list = []
    model_name = "poly_music_Chorales.pt"
    for ep in range(1, 100 + 1):
        train(ep)
        vloss = evaluate(X_valid, name='Validation')
        tloss = evaluate(X_test, name='Test')
        if vloss < best_vloss:
            with open(model_name, "wb") as f:
                torch.save(model, f)
                print("Saved model!\n")
            best_vloss = vloss
        if ep > 10 and vloss > max(vloss_list[-3:]):
            lr /= 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        vloss_list.append(vloss)

    print('-' * 89)
    model = torch.load(open(model_name, "rb"))
    tloss = evaluate(X_test)
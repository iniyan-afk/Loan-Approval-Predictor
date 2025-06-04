import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

train_data[pd.get_dummies(train_data[['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']]).columns] = pd.get_dummies(train_data[['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']])
train_data_1 = train_data.drop(['person_age','id','person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file'], axis = 1)
test_data[pd.get_dummies(test_data[['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']]).columns] = pd.get_dummies(test_data[['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']])
test_data_1 = test_data.drop(['id','person_home_ownership','person_age','loan_intent', 'loan_grade', 'cb_person_default_on_file'], axis = 1)

y_train = train_data['loan_status']
train_data_1.drop('loan_status', axis=1, inplace=True)

train_data_1.replace({True:1, False:0}, inplace=True)
test_data_1.replace({True:1, False:0}, inplace=True)
train_data_1.fillna(0)
test_data_1.fillna(0)

train_data_1['person_income'] = (train_data_1['person_income'] - train_data_1['person_income'].mean())/train_data_1['person_income'].std()
train_data_1['loan_amnt'] = (train_data_1['loan_amnt'] - train_data_1['loan_amnt'].mean())/train_data_1['loan_amnt'].std()
test_data_1['person_income'] = (test_data_1['person_income'] - test_data_1['person_income'].mean())/test_data_1['person_income'].std()
test_data_1['loan_amnt'] = (test_data_1['loan_amnt'] - test_data_1['loan_amnt'].mean())/test_data_1['loan_amnt'].std()

train_data_numpy = train_data_1.to_numpy()
test_data_numpy = test_data_1.to_numpy()
y_train_numpy = y_train.to_numpy()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_data_tensor = torch.from_numpy(train_data_numpy).to(torch.float32).to(device)
test_data_tensor = torch.from_numpy(test_data_numpy).to(torch.float32).to(device)
y_train_tensor = torch.from_numpy(y_train_numpy).to(torch.float32).to(device)

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(25, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 256)
        self.l4 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
    def forward(self, X):
        X = self.relu(self.l1(X))
        X = self.relu(self.l2(X))
        X = self.relu(self.l3(X))
        X = self.l4(X)
        return X
model = Classifier().to(device)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

dataset = TensorDataset(train_data_tensor, y_train_tensor)
batch_size=64
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X, y in loader:
        y_pred = model(X).squeeze()
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss
    avg_loss = total_loss/len(loader)
    print("Epoch:", epoch , "Loss", loss.item())

model.eval()
y_predictions = model(test_data_tensor)
y_pred = torch.round(torch.sigmoid(y_predictions))
y_preds = y_pred.detach().cpu().numpy().squeeze()

output = pd.DataFrame({'id':test_data['id'], 'loan_status' : y_preds})
output.to_csv("out.csv", index=False)

                          

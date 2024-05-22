from DataLoader import DataLoader
from Neural_network import TimeSeriesClassifier
import os
import torch
from SkeletonTracker import SkeletonTracker

loader = DataLoader('')

list_of_correct_technique, list_of_wrong_technique = loader.load_data()
x_train, targets_train, x_val, targets_val = loader.split_data(test_ratio = 0.8)
n_class, n_correct, n_wrong, n_train, n_val = loader.info()

#tracker = SkeletonTracker(list_of_correct_technique[0],1, output_fps=1)
#tracker.visualize_video()

data_txt_names = os.listdir('corr') + os.listdir('wrg')

x_train_data = []
x_val_data = []

targets_train_data = []
targets_val_data = []

for i in range(n_train):
    for j in range(n_train+n_val):
        if x_train[i][0:len(x_train[i])-4] in data_txt_names[j]:
            if j < n_correct:
                x_train_data.append(loader.load_data_from_txt(f'.\\corr\\{data_txt_names[j]}'))
                targets_train_data.append(1)
            else:
                x_train_data.append(loader.load_data_from_txt(f'.\\wrg\\{data_txt_names[j]}'))
                targets_train_data.append(0)

for i in range(n_val):
    for j in range(n_train+n_val):
        if x_val[i][0:len(x_val[i])-4] in data_txt_names[j]:
            if j < n_correct:
                x_val_data.append(loader.load_data_from_txt(f'.\\corr\\{data_txt_names[j]}'))
                targets_val_data.append(1)
            else:
                x_val_data.append(loader.load_data_from_txt(f'.\\wrg\\{data_txt_names[j]}'))
                targets_val_data.append(0)

x_train_data = torch.tensor(x_train_data).unsqueeze(1)
x_val_data = torch.tensor(x_val_data).unsqueeze(1)
targets_train_data = torch.tensor(targets_train_data)
targets_val_data = torch.tensor(targets_val_data)

model = TimeSeriesClassifier(n_class)
print('Testing, if data and model are configured correctly')
model.train_neural_network(x_train_data, targets_train_data, num_epochs=10, learning_rate=0.001, weight_decay=1e-5, features_val=x_val_data, targets_val=targets_val_data)
model.confusion_matrix(x_val_data,targets_val_data)
print('Beginning k-fold validation')
model.k_fold_validation(x_train_data, targets_train_data, k=5, num_epochs=10, learning_rate=0.001, weight_decay=1e-5)
#print('Beginning leave-one-out validation')
#model.leave_one_out_validation(x_train_data, targets_train_data, num_epochs=10, learning_rate=0.001, weight_decay=1e-5)
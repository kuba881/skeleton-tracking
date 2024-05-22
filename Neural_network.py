import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from mlxtend.plotting import plot_confusion_matrix as mlxtend_plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, LeaveOneOut, train_test_split
import matplotlib.pyplot as plt
import os
import platform

if platform.uname().system=='Windows':
    par=r'\\'
else:
    par='/'

class TimeSeriesClassifier(nn.Module):
    def __init__(self, n_classes):
        super(TimeSeriesClassifier, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=(3, 3, 3), stride=1, padding=(0, 0, 0))
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1,1), padding=(0, 0))
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(3, 2, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(256*123*6, 256)
        self.fc2 = nn.Linear(256, n_classes)

        self.path=''
        self.results=f'Informations about neural network \n\n Basics: \n Step size: 0.001 \n Number of learning epochs: 1500 \n Optimizer: AdamW'

    def forward(self, x):
        #print(x.size())
        #x = self.batch_norm(x)
        x = F.relu(self.conv1(x))
        #print(x.size())
        x = self.pool1(x)
        #print(x.size())
        x=x.view(len(x), 64, 496, 30)
        #print(x.size())
        x = F.relu(self.conv2(x))
        #print(x.size())
        x = self.pool2(x)
        #print(x.size())

        x = F.relu(self.conv3(x))
        #print(x.size())
        x = self.pool3(x)
        #print(x.size())
        x = x.view(len(x), 256*123*6)
        
        #print(x.size())
        x = F.relu(self.fc1(x))
        #print(x.size())
        x = self.fc2(x)
        #print(x.size())
        return x
    
    #@jit(target_backend='cuda')
    def train_neural_network(self, features, targets, num_epochs=1000, learning_rate=0.01, weight_decay=1e-5, features_val=[0], targets_val=[0], val=None, k=0):
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)#SGD, Adam, RMSprop or Adagrad

        losses = []
        acc  =[]

        for epoch in range(num_epochs):
            # Forward pass
            outputs = self(features)

            loss = criterion(outputs, targets)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if len(features_val)!=1 and epoch % 1 ==0:
                metric = self.accuracy_during_training(features_val, targets_val)
                acc.append(metric)
                if metric>1:
                    self.save_weights(epoch,metric,val,k)

            # Save the loss for plotting
            losses.append(loss.item())
            # Print training progress
            #if (epoch + 1) % 50 == 0:
            #print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
                

        # Plot the training progress
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.savefig(self.path + f'.{par}training_curve.png')
        plt.clf()

        if len(features_val)!=1:
            plt.plot(acc)
            plt.xlabel('Epoch')
            plt.ylabel('Validation_accuracy')
            plt.title('Training Progress')
            plt.savefig(self.path + f'.{par}val_accuracy_curve.png')
            plt.clf()

        #metrics = self.evaluate_metrics(features, targets)
        #print(metrics)
    def accuracy_during_training(self,features, targets):
        outputs = self(features)
        _, predictions = torch.max(outputs, 1)  # Get the predicted class indices
        predictions = predictions.detach().numpy()

        accuracy = accuracy_score(targets, predictions)
        return accuracy

    def evaluate_metrics(self, features, targets):
        outputs = self(features)
        _, predictions = torch.max(outputs, 1)  # Get the predicted class indices
        predictions = predictions.detach().numpy()

        f1 = f1_score(targets, predictions)
        recall = recall_score(targets, predictions)
        accuracy = accuracy_score(targets, predictions)
        precision = precision_score(targets, predictions)

        return f1, recall, accuracy, precision
    
    def k_fold_validation(self, features, targets, k=5, num_epochs=100, learning_rate=0.01, weight_decay=1e-5):
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        metrics_per_fold = []

        for fold, (train_index, val_index) in enumerate(kf.split(features)):
            train_features, val_features = features[train_index], features[val_index]
            train_targets, val_targets = targets[train_index], targets[val_index]

            self.reset_parameters()  # Reset model parameters for each fold

            self.train_neural_network(train_features, train_targets, num_epochs, learning_rate, weight_decay,val_features,val_targets,'kfl',fold)
            metrics = self.evaluate_metrics(val_features, val_targets)

            metrics_per_fold.append(metrics)

            print(f'\nMetrics for Fold {fold + 1}:')
            print(f'F1 Score: {metrics[0]:.4f}, Recall: {metrics[1]:.4f}, Accuracy: {metrics[2]:.4f}, Precision: {metrics[3]:.4f}')

        avg_metrics = [np.mean(metric) for metric in zip(*metrics_per_fold)]
        confidence_interval = [1.96 * (np.std(metric) / np.sqrt(k)) for metric in zip(*metrics_per_fold)]
        lower_bound = [avg - ci for avg, ci in zip(avg_metrics, confidence_interval)]
        upper_bound = [avg + ci for avg, ci in zip(avg_metrics, confidence_interval)]
        
        self.results=self.results + f'\n\nAverage Metrics K-fold:'+ f'\n Average F1 Score: {avg_metrics[0]:.4f} +- {confidence_interval[0]:.4f}, \n Average Recall: {avg_metrics[1]:.4f} +- {confidence_interval[1]:.4f}, \n Average Accuracy: {avg_metrics[2]:.4f} +- {confidence_interval[2]:.4f}, \n Average Precision: {avg_metrics[3]:.4f} +- {confidence_interval[3]:.4f}'
        print('\nAverage Metrics Across Folds:')
        print(f'Average F1 Score: {avg_metrics[0]:.4f} +- {confidence_interval[0]:.4f}, Average Recall: {avg_metrics[1]:.4f} +- {confidence_interval[1]:.4f}, Average Accuracy: {avg_metrics[2]:.4f} +- {confidence_interval[2]:.4f}, Average Precision: {avg_metrics[3]:.4f} +- {confidence_interval[3]:.4f}')
        #print(self.results)
        self.plot_metrics_with_interval(['F1 Score', 'Recall', 'Accuracy', 'Precision'], avg_metrics[0:4], confidence_interval[0:4],'k_fold_')

    def leave_one_out_validation(self, features, targets, num_epochs=100, learning_rate=0.01, weight_decay=1e-5):
        loo = LeaveOneOut()
        metrics_per_instance = []
        k=0
        for instance, (train_index, val_index) in enumerate(loo.split(features)):
            train_features, val_features = features[train_index], features[val_index]
            train_targets, val_targets = targets[train_index], targets[val_index]
            
            train_features, val_features = train_features[0:200], train_features[200:-1]
            train_targets, val_targets = train_targets[0:200], train_targets[200:-1]

            self.reset_parameters()  # Reset model parameters for each fold

            self.train_neural_network(train_features, train_targets, num_epochs, learning_rate, weight_decay, val_features, val_targets,'loo',instance)
            
            metrics = self.evaluate_metrics(val_features, val_targets)

            metrics_per_instance.append(metrics)

            #print(f'\nMetrics for Instance {instance + 1}:')
            #print(f'F1 Score: {metrics[0]:.4f}, Recall: {metrics[1]:.4f}, Accuracy: {metrics[2]:.4f}, Precision: {metrics[3]:.4f}')
            k+=1
            if k >5:
                break

        avg_metrics = [np.mean(metric) for metric in zip(*metrics_per_instance)]
        confidence_interval = [1.96 * (np.std(metric) / np.sqrt(len(features))) for metric in zip(*metrics_per_instance)]
        lower_bound = [avg - ci for avg, ci in zip(avg_metrics, confidence_interval)]
        upper_bound = [avg + ci for avg, ci in zip(avg_metrics, confidence_interval)]

        self.results=self.results + f'\n\nAverage Metrics Leave one out:'+f'\n Average F1 Score: {avg_metrics[0]:.4f} +- {confidence_interval[0]:.4f}, \n Average Recall: {avg_metrics[1]:.4f} +- {confidence_interval[1]:.4f}, \n Average Accuracy: {avg_metrics[2]:.4f} +- {confidence_interval[2]:.4f}, \n Average Precision: {avg_metrics[3]:.4f} +- {confidence_interval[3]:.4f}'
        print('\nAverage Metrics Across Instances:')
        print(f'Average F1 Score: {avg_metrics[0]:.4f} +- {confidence_interval[0]:.4f}, Average Recall: {avg_metrics[1]:.4f} +- {confidence_interval[1]:.4f}, Average Accuracy: {avg_metrics[2]:.4f} +- {confidence_interval[2]:.4f}, Average Precision: {avg_metrics[3]:.4f} +- {confidence_interval[3]:.4f}')
        #print(self.results)
        self.plot_metrics_with_interval(['F1 Score', 'Recall', 'Accuracy', 'Precision'], avg_metrics[0:4], confidence_interval[0:4],'LOO_')

    def plot_metrics_with_interval(self, metrics_names, avg_metrics, confidence_interval,name):
        colors = ['blue', 'green', 'orange', 'red']

        plt.bar(metrics_names, avg_metrics, yerr=confidence_interval, color=colors, capsize=5, alpha=0.7)
        plt.ylabel('Score')
        plt.title('Final Metrics with Confidence Interval')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(self.path + '.' + par + name + 'final_metrics_graph.png')
        plt.clf()
        
    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def save_weights(self, epoch, acc, val=None, k=0):
        torch.save(self.state_dict(), self.path + f'.{par}model_weights_{val}{k}ep{"{0:04}".format(epoch)}_acc{acc:.4f}.pth')

    def save_info(self):
        #open text file
        text_file = open(self.path + f".{par}informations.txt", "w")
        #print(self.results)
        #write string to file
        text_file.write(self.results)
 
        #close file
        text_file.close()

    def confusion_matrix(self, features, targets):
        outputs = self(features)
        _, predictions = torch.max(outputs, 1)  # Get the predicted class indices
        predictions = predictions.detach().numpy()

        cm = confusion_matrix(targets, predictions)
        
        # Plot confusion matrix using mlxtend
        mlxtend_plot_confusion_matrix(conf_mat=cm, show_absolute=True, show_normed=True)
        plt.savefig(self.path + f'.{par}confusion_matrix.png')
        plt.clf()
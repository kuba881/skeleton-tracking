import os
import random
import torch

class DataLoader():
    def __init__(self, path):
        self.path = path

    def load_data(self):
        self.correct_technique = os.listdir(os.path.join(self.path,'correct'))
        self.wrong_technique = os.listdir(os.path.join(self.path,'wrong'))
        return self.correct_technique, self.wrong_technique

    def split_data(self,test_ratio=0.8):
        # Generate test indexes
        train_indexes_corr = random.sample(range(len(self.correct_technique)), round(len(self.correct_technique) * test_ratio))
        train_indexes_wrong = random.sample(range(len(self.wrong_technique)), round(len(self.wrong_technique) * test_ratio))

        # Generate train indexes
        val_indexes_corr = list(set(range(len(self.correct_technique))) - set(train_indexes_corr))
        val_indexes_wrong = list(set(range(len(self.wrong_technique))) - set(train_indexes_wrong))

        # Create train and test sets
        train_targets = [1] * len(train_indexes_corr) + [0] * len(train_indexes_wrong)
        self.train_values = [self.correct_technique[i] for i in train_indexes_corr] + [self.wrong_technique[i] for i in train_indexes_wrong]

        val_targets = [1] * len(val_indexes_corr) + [0] * len(val_indexes_wrong)
        self.val_values = [self.correct_technique[i] for i in val_indexes_corr] + [self.wrong_technique[i] for i in val_indexes_wrong]

        return self.train_values, train_targets, self.val_values, val_targets

    def info(self):
        n_class = 2
        n_correct = len(self.correct_technique)
        n_wrong = len(self.wrong_technique)
        n_train = len(self.train_values)
        n_val = len(self.val_values)
        return n_class, n_correct, n_wrong, n_train, n_val

    def load_data_from_txt(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        data_complete = []
        data = []
        for line in lines:
            if line.strip() and line[0]!='F':  # Skip empty lines
                coordinates = [float(coord) for coord in line.split(',')]
                data.append(coordinates)
                if len(data)==33:
                    data_complete.append(data)
                    data = []

        time_dimension = len(data_complete)
        if 500 - time_dimension != 0:
            for _ in range(500 - time_dimension):
                data_complete.append([[0,0,0] for _ in range(33)])
        return data_complete

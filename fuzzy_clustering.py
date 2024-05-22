import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from DataLoader import DataLoader
import os

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

data = np.array(x_train_data)
data = data.reshape(-1,500*33*3)

n_clusters = 2

# Apply fuzzy c-means clustering
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    data.T, n_clusters, 2, error=0.005, maxiter=1000, init=None
)

# Predict cluster membership for each data point
cluster_membership = np.argmax(u, axis=0)

# Print the cluster centers
print('Cluster Centers:', cntr)

# Print the cluster membership for each data point
print('Cluster Membership:', cluster_membership)

# Assuming true labels are provided in the array `true_labels`
true_labels = targets_train_data  # Replace with your true labels

# Calculate evaluation metrics
accuracy = metrics.accuracy_score(true_labels, cluster_membership)
precision = metrics.precision_score(true_labels, cluster_membership, average='weighted')
recall = metrics.recall_score(true_labels, cluster_membership, average='weighted')
f1 = metrics.f1_score(true_labels, cluster_membership, average='weighted')

print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1)

# Plot confusion matrix
confusion_matrix = metrics.confusion_matrix(true_labels, cluster_membership)
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

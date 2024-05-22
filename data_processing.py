from SkeletonTracker import SkeletonTracker
from DataLoader import DataLoader

loader = DataLoader('')
list_of_correct_technique, list_of_wrong_technique = loader.load_data()
x_train, targets_train, x_val, targets_val = loader.split_data(test_ratio = 0.8)

for i,record in enumerate(x_train):
    tracker = SkeletonTracker(record,targets_train[i], output_fps=1)
    tracker.process_video()

for i,record in enumerate(x_val):
    tracker = SkeletonTracker(record,targets_val[i], output_fps=1)
    tracker.process_video()

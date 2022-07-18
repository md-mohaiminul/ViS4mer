
import pickle

x = pickle.load(open('ucf101_feature.pkl', 'r'))

for a in x:
    print(a, x[a].shape)
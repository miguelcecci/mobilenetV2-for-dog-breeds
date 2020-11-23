import matplotlib.pyplot as plt
import tensorflow as tf
import pickle

loaded_model = './models/seila'

history = pickle.load(open('./trainHistoryDict', "rb"))
print(history)
def plot_graphs(history, string):
    plt.plot(history[string])
    plt.plot(history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()

plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')

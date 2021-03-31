import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

histories = ['evaluation/'+ x for x in os.listdir('evaluation')]

for history in histories:
    data = np.load(history, allow_pickle=True)
    eval = dict(enumerate(data.flatten(), 1))[1]
    print(eval.keys())

    plt.plot(range(0, len(eval['loss'])), eval['loss'], label="Loss")
    plt.plot(range(0, len(eval['accuracy'])), eval['accuracy'], label="tr_acc")
    plt.plot(range(0, len(eval['val_loss'])), eval['val_loss'], label="val_loss")
    plt.plot(range(0, len(eval['val_accuracy'])), eval['val_accuracy'], label="val_accuracy")
    plt.legend(loc="upper right", fontsize=8)
    # plt.show()
    plt.savefig(history[:-4]+'.png')
    plt.clf()
    del eval
    # print(eval)
    # exit()

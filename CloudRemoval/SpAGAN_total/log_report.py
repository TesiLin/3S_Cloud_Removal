import json
import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt


class LogReport():
    def __init__(self, log_dir, log_name='log'):
        self.log_dir = log_dir
        self.log_name = log_name
        self.log_ = []

    def __call__(self, log):
        self.log_.append(log)
        with open(os.path.join(self.log_dir, self.log_name), 'w', encoding='UTF-8') as f:
            json.dump(self.log_, f, indent=4)
    
    def save_lossgraph(self):
        epoch = []
        gen_loss = []
        dis_loss = []

        for l in self.log_:
            epoch.append(l['epoch'])
            gen_loss.append(l['gen/loss'])
            dis_loss.append(l['dis/loss'])

        epoch = np.asarray(epoch)
        gen_loss = np.asarray(gen_loss)
        dis_loss = np.asarray(dis_loss)

        plt.figure(1)
        plt.plot(epoch, gen_loss, 'r', label='gen_loss')
        plt.plot(epoch, dis_loss, 'b', label='dis_loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig(os.path.join(self.log_dir, 'lossgraph_gen_dis.pdf'))
        plt.close()


class TestReport():
    def __init__(self, log_dir, log_name='log_test'):
        self.log_dir = log_dir
        self.log_name = log_name
        self.log_ = []

    def __call__(self, log):
        self.log_.append(log)
        with open(os.path.join(self.log_dir, self.log_name), 'w', encoding='UTF-8') as f:
            json.dump(self.log_, f, indent=4)
    

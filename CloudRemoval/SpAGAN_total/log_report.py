import json
import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt


class LogReport():
    def __init__(self, log_dir, log_name='log.json'):
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
        
        epoch_max=epoch[-1]
        gen_sum=0
        dis_sum=0
        final_epoch=[]
        final_genloss=[]
        final_disloss=[]
        for i in range(0, epoch_max+1):
            epoch_count = epoch.count(i)
            if epoch_count != 0: # 确保元素在list内
                index_start = epoch.index(i)
                for j in range (0, epoch_count):
                    gen_sum += gen_loss[index_start + j]
                    dis_sum += dis_loss[index_start + j]
                gen_sum = gen_sum / epoch_count
                dis_sum = dis_sum / epoch_count
                
                final_epoch.append(i)
                final_genloss.append(gen_sum)
                final_disloss.append(dis_sum)

        final_epoch = np.asarray(final_epoch)
        final_genloss = np.asarray(final_genloss)
        final_disloss = np.asarray(final_disloss)

        plt.figure(1)
        plt.plot(final_epoch, final_genloss, 'r', label='gen_loss')
        plt.plot(final_epoch, final_disloss, 'b', label='dis_loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig(os.path.join(self.log_dir, 'lossgraph_gen_dis.pdf'))
        plt.close()

        with open(os.path.join(self.log_dir, "final_loss.txt"), 'w', encoding='UTF-8') as f:
            f.write(str(final_epoch[-1])+'\t'+str(final_genloss[-1])+'\t'+str(final_disloss[-1]))



class TestReport():
    def __init__(self, log_dir, log_name='log_test.json'):
        self.log_dir = log_dir
        self.log_name = log_name
        self.log_ = []

    def __call__(self, log):
        self.log_.append(log)
        with open(os.path.join(self.log_dir, self.log_name), 'w', encoding='UTF-8') as f:
            json.dump(self.log_, f, indent=4)
    

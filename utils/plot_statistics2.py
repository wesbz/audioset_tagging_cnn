import os
import sys
import numpy as np
import argparse
import h5py
import time
import pickle
import matplotlib.pyplot as plt
from sklearn import metrics

from utilities import (create_folder, get_filename, d_prime)
import config



def plot(args):
    
    # Arguments & parameters
    workspace = args.workspace
    select = args.select
    
    classes_num = config.classes_num
    max_plot_iteration = 3000000
    iterations = np.arange(0, max_plot_iteration, 2000)

    class_labels_indices_path = os.path.join('metadata', 'class_labels_indices.csv')
        
    save_out_path = 'results/{}.pdf'.format(select)
    create_folder(os.path.dirname(save_out_path))
    
    # Read labels
    labels = config.labels
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    lines = []
        
    def _load_metrics(filename, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, data_type, model_type, loss_type, balanced, augmentation, batch_size):
        statistics_path = os.path.join(workspace, 'statistics', filename, 
            'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
            sample_rate, window_size, hop_size, mel_bins, fmin, fmax), 
            'data_type={}'.format(data_type), model_type, 
            'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced), 
            'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size), 
            'statistics.pkl')

        statistics_dict = pickle.load(open(statistics_path, 'rb'))

        bal_map = np.array([statistics['average_precision'] for statistics in statistics_dict['bal']])    # (N, classes_num)
        bal_map = np.mean(bal_map, axis=-1)
        test_map = np.array([statistics['average_precision'] for statistics in statistics_dict['test']])    # (N, classes_num)
        test_map = np.mean(test_map, axis=-1)
        legend = '{}, {}, bal={}, aug={}, bs={}'.format(data_type, model_type, balanced, augmentation, batch_size)

        # return {'bal_map': bal_map, 'test_map': test_map, 'legend': legend}
        return bal_map, test_map, legend
        
    bal_alpha = 0.3
    test_alpha = 1.0
    lines = []

    if select == '3a':
        # (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
        #     320, 64, 50, 14000, 'full_train', 'Cnn14', 'clip_bce', 'balanced', 'mixup', 32)
        # line, = ax.plot(bal_map, color='r', alpha=bal_alpha)
        # line, = ax.plot(test_map, label='cnn14', color='r', alpha=test_alpha)
        # lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 16000, 512, 
            160, 64, 50, 8000, 'full_train', 'Cnn14_16k', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='r', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn14_16k', color='r', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 16000, 512, 
            160, 64, 50, 8000, 'full_train', 'Cnn14_small_16k', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='k', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn14_small_16k', color='k', alpha=test_alpha)
        lines.append(line)

        # (bal_map, test_map, legend) = _load_metrics('main', 16000, 512, 
        #     160, 64, 50, 8000, 'full_train', 'Cnn14_small_16k', 'clip_bce', 'balanced', 'mixup', 32)
        # line, = ax.plot(bal_map, color='k', alpha=bal_alpha)
        # line, = ax.plot(test_map, label='cnn14_small_16k', color='k', alpha=test_alpha)
        # lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 8000, 256, 
            80, 64, 50, 4000, 'full_train', 'Cnn14_8k', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='b', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn14_8k', color='b', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn14', 'clip_bce', 'alternate', 'mixup', 32)
        line, = ax.plot(bal_map, color='g', alpha=bal_alpha)
        line, = ax.plot(test_map, label='cnn14, alternate', color='g', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Wavegram_Logmel_Cnn14', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='y', alpha=bal_alpha)
        line, = ax.plot(test_map, label='Wavegram_Logmel_Cnn14', color='y', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 128, 50, 14000, 'full_train', 'Wavegram_Logmel128_Cnn14', 'clip_bce', 'alternate', 'mixup', 32)
        line, = ax.plot(bal_map, color='m', alpha=bal_alpha)
        line, = ax.plot(test_map, label='Wavegram_Logmel128_Cnn14, alternate', color='m', alpha=test_alpha)
        lines.append(line)

    if select == '3b':
        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn14_Transformer_pos', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='r', alpha=bal_alpha)
        line, = ax.plot(test_map, label='Cnn14_Transformer_pos', color='r', alpha=test_alpha)
        lines.append(line)

        (bal_map, test_map, legend) = _load_metrics('main', 32000, 1024, 
            320, 64, 50, 14000, 'full_train', 'Cnn14_Transformer_nopos', 'clip_bce', 'balanced', 'mixup', 32)
        line, = ax.plot(bal_map, color='g', alpha=bal_alpha)
        line, = ax.plot(test_map, label='Cnn14_Transformer_nopos', color='g', alpha=test_alpha)
        lines.append(line)
        

    ax.set_ylim(0, 1.)
    ax.set_xlim(0, len(iterations))
    ax.xaxis.set_ticks(np.arange(0, len(iterations), 25))
    ax.xaxis.set_ticklabels(np.arange(0, max_plot_iteration, 50000))
    ax.yaxis.set_ticks(np.arange(0, 1.01, 0.05))
    ax.yaxis.set_ticklabels(np.around(np.arange(0, 1.01, 0.05), decimals=2))        
    ax.grid(color='b', linestyle='solid', linewidth=0.3)
    plt.legend(handles=lines, loc=2)
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # ax.legend(handles=lines, bbox_to_anchor=(1.0, 1.0))

    plt.savefig(save_out_path)
    print('Save figure to {}'.format(save_out_path))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers(dest='mode')
    
    parser_plot = subparsers.add_parser('plot')
    parser_plot.add_argument('--workspace', type=str, required=True)
    parser_plot.add_argument('--select', type=str, required=True)

    args = parser.parse_args()
    
    if args.mode == 'plot':
        plot(args)

    else:
        raise Exception('Error argument!')
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 18:19:55 2018

@author: jsevillamol
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def summarize_multi_results():
    # get subdirectories
    train_dirs = [dir_ for dir_ in os.listdir() if os.path.isdir(dir_)]
    train_dirs.sort()
    
    # Read the results from the runs    
    first_iteration = True
    for train_dir in train_dirs:
        run_csv_fn = f'{train_dir}/training_history.csv'
        try:
            run_df = pd.read_csv(run_csv_fn)
            run_df['run_name'] = train_dir
            
        except (FileNotFoundError,pd.errors.EmptyDataError): 
            continue # If there is no training_history.csv we ignore this folder
        
        if first_iteration:
            metric_names = list(run_df)
            metric_names.remove('run_name')
            metric_names.remove('epoch')
            summary_df = run_df
            first_iteration = False
        else:
            summary_df = pd.concat([summary_df, run_df])
    
    # Save summary
    summary_csv_fn = 'multi_summary.csv'
    summary_df.to_csv(summary_csv_fn)
    
    # Plot the training progress per metric of each run
    for metric in metric_names:
        #plt.legend()
        plt.title(f'model {metric}')
        plt.ylabel(metric)
        plt.xlabel('epoch')
        
        for label, run_df in summary_df.groupby('run_name'):
            plt.plot(run_df[metric])
        
        plt.savefig(f'history_{metric}.png')
        plt.clf()
    
    # plot a histogram of the number of epochs per run
    n_epochs = summary_df.groupby('run_name').size().values
    bins = np.arange(0, n_epochs.max() + 1.5)
    fig, ax = plt.subplots()
    ax.hist(n_epochs, bins)
    ax.set_xticks(bins)
    plt.savefig('n_epochs.png')
    plt.clf()

def summarize_run_results(run_csv_fn, train_dir='.', metrics = ['loss', 'acc', 'auc']):
    df = pd.read_csv(run_csv_fn)
    for metric in metrics:
        df.plot.line(x='epoch', y = [metric, f'val_{metric}'])
        plt.savefig(f'{train_dir}/history_{metric}.png')
        plt.clf()
            
########################
# LAUNCH SCRIPT
########################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
              description=("Summarize the results of a training session"))

    parser.add_argument('path', default=None, help="")
    args = parser.parse_args()

    if args.path is None:
        summarize_multi_results()
    else: 
        summarize_run_results(args.path)

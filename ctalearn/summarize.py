#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 18:19:55 2018

@author: jsevillamol
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def summarize_multi_results():
    # get subdirectories
    train_dirs = [dir_ for dir_ in os.listdir() if os.path.isdir(dir_)]
    train_dirs.sort()
    
    # Read the results from the runs
    histories = []
    summary_csv_fn = 'multi_summary.csv'
    first_iteration = True
    for train_dir in train_dirs:
        run_csv_fn = f'{train_dir}/training_history.csv'
        try:
            with open(run_csv_fn) as csv_file:
                csv_reader = csv.reader(csv_file)
                csv_contents = []
                for row in csv_reader: 
                    csv_contents.append(row)
                if len(csv_contents) == 0: continue
                metric_names = csv_contents[0]
                history = np.array(csv_contents[1:])
                histories.append(history)
        except FileNotFoundError: continue # If there is no training_history.csv we ignore this folder
        
        # Write the names of the columns in the first iteration
        if first_iteration:
            with open(summary_csv_fn, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['run_name'] + metric_names)
            first_iteration = False
    
        # Write the final results in a summary file
        with open(summary_csv_fn, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([train_dir] + list(history[-1, :]))
        
    # Plot the runs
    for i, metric in enumerate(metric_names):
        plt.title(f'model {metric}')
        plt.ylabel(metric)
        plt.xlabel('epoch')
        for history in histories: 
            hist = list(map(float, history[:, i]))
            plt.plot(hist)
        plt.savefig(f'history_{metric}.png')
        plt.clf()
    
            
########################
# LAUNCH SCRIPT
########################

if __name__ == "__main__":
    summarize_multi_results()
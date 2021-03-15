#!/usr/bin/env python

import sys
import os
import numpy as np

# CLASSES = ['car', 'pedestrian', 'cyclist', 'van', 'truck', 'person_sitting', 'tram']
CLASSES = ['pedestrian']

# PARAMS = ['detection', 'orientation', 'iour', 'mppe']
PARAMS = ['detection', 'detection_1%', 'detection_5%', 'detection_10%', 'detection_3d', 'detection_ground', 'orientation']

DIFFICULTIES = ['easy', 'moderate', 'hard', 'all']

eval_type = ''

if len(sys.argv)<2:
    print('Usage: parser.py results_folder [evaluation_type]')

if len(sys.argv)==3:
    eval_type = sys.argv[2]

result_sha = sys.argv[1]
txt_dir = os.path.join('build','results', result_sha)

for class_name in CLASSES:
    for param in PARAMS:
        print("--{:s} {:s}--".format(class_name, param))
        if eval_type is '':
            txt_name = os.path.join(txt_dir, 'stats_' + class_name + '_' + param + '.txt')
        else:
            txt_name = os.path.join(txt_dir, 'stats_' + class_name + '_' + param + '_' + eval_type + '.txt')

        if not os.path.isfile(txt_name):
            print(txt_name, ' not found')
            continue

        cont = np.loadtxt(txt_name)

        averages = []
        for idx, difficulty in enumerate(DIFFICULTIES):
            sum = 0
            if param in PARAMS:
                for i in range(1, 41):
                    sum += cont[idx][i]

                average = sum/40.0
                
            #print class_name, difficulty, param, average
            averages.append(average)

        #print "\n"+class_name+" "+param
        print("Easy\tMod.\tHard\tAll")
        print("{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}".format(100*averages[0], 100*averages[1],100*averages[2],100*averages[3]))
        print("---------------------------------------------------------------------------------")
        if eval_type is not '' and param=='detection':
            break # No orientation for 3D or bird eye

    #print '================='

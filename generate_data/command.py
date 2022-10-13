import sys
sys.path.append("..")
import subprocess as sub
import os
import random

def run_cmd(dock_cmd = "", speed_run = False):

    # change working directory to predictions folder
    os.chdir('predictions')
    # perform 3D and docking prediction
    if speed_run:
        scores = [round(random.uniform(-10000.0,10000.0),4) for _ in range (0,5)]
        result = sub.Popen(('echo',f'Total weighted score: {scores[0]}\nTotal weighted score: {scores[1]}\nTotal weighted score: {scores[2]}\nTotal weighted score: {scores[3]}\nTotal weighted score: {scores[4]}'),stdout = sub.PIPE, stderr=sub.PIPE)
    else:                    
        result = sub.Popen(dock_cmd.split(),stdout = sub.PIPE, stderr = sub.PIPE)                      

    # change working directory back
    os.chdir('..')

    return result    
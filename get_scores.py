from generate_data.write_file import *
from generate_data.command import *
from arnie.mfe import mfe
import generate_data.read_in_data as rd
import pandas as pd
import multiprocessing as mp
import time
import math
from tqdm import tqdm
from shutil import rmtree as rm
from colorama import Fore, Style

# file names and directories
SUB_DIR = "predictions/"
FASTA = SUB_DIR+"fasta.txt"
SECONDARY = SUB_DIR+"secstruct.txt"
FLAGS = SUB_DIR+"flags"
CONSTRAINT = SUB_DIR+"constraint.cst"
SAVE_DIR = None
DEBUG_DIR = None
DOCK_CMD = None
FILE_TYPE = None
TRAIN_PERCENT = None

# Setting styles
RED = Fore.RED
GREEN = Fore.GREEN
CYAN = Fore.CYAN
YELLOW = Fore.YELLOW
MAGENTA = Fore.MAGENTA
RESET = Style.RESET_ALL


class Predictions:
    def __init__(self, command_str, train_percent, process_percent, file_type, score_output_dir, debug_dir):
        global DOCK_CMD
        global SAVE_DIR
        global DEBUG_DIR
        global FILE_TYPE
        global TRAIN_PERCENT

        DOCK_CMD = command_str
        TRAIN_PERCENT = train_percent
        FILE_TYPE = file_type
        SAVE_DIR = score_output_dir
        DEBUG_DIR = debug_dir
        self.max_num_processes = int(mp.cpu_count()*process_percent)

        # obtain all rna_denovo input
        # (Note: fasta and secondary file should not have any RNA info in it yet)
        self.fasta_data = rd.read_in(FASTA)
        self.secondary_data = rd.read_in(SECONDARY)
        self.flag_data = rd.read_in(FLAGS)
        self.constraint_data = rd.read_in(CONSTRAINT)

        # obtain the fasta header
        self.fasta_header = self.fasta_data[0].rstrip()
        # get last letter of fasta header (split list based on ":", then get the 2nd last 
        # element in the list and grab the last character - should be the last letter)
        self.last_letter = self.fasta_header.split(":")[-2][-1:]
        # should always provide the last chain as the rna sequence - no matter what
        # ord will return the unicode value, chr will convert it back into a letter
        self.rna_chain_letter = chr(ord(self.last_letter)+1)

        # obtain the protein sequence from the 2nd line in fasta_data 
        # by breaking it into a list divided by case, then returning the capital letters
        self.PROTEIN = self.fasta_data[1]
        # create as many dots as there are in the protein sequence length
        self.DOTS = "."*len(self.PROTEIN)

        
    def update_progress(self, progress_bar):
        progress_bar.update()

        
    def get_scores(self, start_data, output_file_name, output_debug_file_name, speed_run, debug, tqdm_colour, text_colour, reloading):
        data_save_name = output_file_name + FILE_TYPE
        troubleshoot_name = output_debug_file_name + FILE_TYPE

        datasize = len(start_data)
        loops = math.ceil(datasize/self.max_num_processes)
        
        start_time = time.time()
        all_data = []
        tqdm.write(f"{text_colour}Beginning secondary structure predictions")
        for sequence in start_data:
            # get the secondary structure
            sec_seq = mfe(sequence, package='eternafold')
            all_data.append([sequence, sec_seq])
        tqdm.write(f"{text_colour}Finished secondary structure prediction. Process took {time.time() - start_time} seconds for {datasize} sequences\nBeginning tertiary structure prediction.\n")
        
        progress = tqdm(total = datasize, desc=f"{text_colour}Scores", colour = tqdm_colour)
        data = []
        index = 0
        # loop data size/number of processes times
        for i in range(loops):
            data_debug = []
            # calculate number of processes remaining to run
            rem_proc = datasize - i * self.max_num_processes
            # if remaining number of processes to run is less than the number of processes, set to
            # remaining number of processes to finish it off. Else, loop through the number of 
            # processes
            sub_data_size = rem_proc if rem_proc < self.max_num_processes else self.max_num_processes

            # if loading in data from a crash, skip this entire process. Just advance the random states and update progress
            if reloading:
                for _ in range(sub_data_size):
                    label = "test" if random.random() > TRAIN_PERCENT else "train"
                    self.update_progress(progress)
                    time.sleep(0.01)
            
            # otherwise, do the actual predictions
            else:
                # running list is the currently running processes. 
                running_list = []

                # create tmp folder with current time in seconds (no decimal) under current directory
                temp_dir = ''.join(['tmp_',str(int(time.time()))])
                new_dir = SUB_DIR+temp_dir
                # note: os is imported from command.py
                os.mkdir(new_dir)

                # run all commands and get scores
                for _ in range(sub_data_size):
                    both = all_data.pop()
                    # get a random rna sequence
                    prim_seq = both[0]
                    # get the secondary structure
                    sec_seq = both[1]
                    # get the index for the middle of the RNA after the protein
                    mid_rna_index = str(len(self.PROTEIN) + int(len(prim_seq)/2))

                    # get the rna chain range and update the fasta header
                    rna_chain_range = str(len(self.PROTEIN)+1)+'-'+str(len(self.PROTEIN)+len(prim_seq))

                    # set WHERE the constraint should take place on the RNA
                    temp = self.constraint_data[0].split()
                    temp[4] = mid_rna_index
                    temp = " ".join(temp)
                    
                    # write out the constraint file for this subprocess
                    path_txt = [temp_dir, '/', 'constraint', '_', str(index), '.', 'cst']
                    const_file = ''.join(path_txt)
                    write_data(SUB_DIR+const_file,temp)

                    # write new fasta file
                    path_txt[2] = 'fasta'
                    path_txt[6] = 'txt'
                    rna_chain = ' ' + self.rna_chain_letter + ':' + rna_chain_range
                    fasta_file = ''.join(path_txt)
                    write_fasta(SUB_DIR+fasta_file,self.fasta_header,rna_chain,self.PROTEIN,prim_seq)
                    
                    # overwrite the secondary structure file with new data 
                    path_txt[2] = 'secstruct'
                    path_txt[6] = 'txt'
                    sec_file = ''.join(path_txt)
                    write_secondary(SUB_DIR+sec_file,self.DOTS,self.PROTEIN,sec_seq,prim_seq)
                    
                    # write out the flags file
                    path_txt[2] = 'flags'
                    path_txt[5] = ''
                    path_txt[6] = ''
                    self.flag_data[0] = f'-fasta {fasta_file}\n'
                    self.flag_data[1] = f'-secstruct_file {sec_file}\n'
                    self.flag_data[2] = f'-constraints:cst_file {const_file}\n'
                    flag_file = ''.join(path_txt)
                    write_data(SUB_DIR+flag_file,self.flag_data)

                    # assign a test or training label to the sequence
                    label = "test" if random.random() > TRAIN_PERCENT else "train"
                    # run the docking command for this sequence
                    rna_denovo_cmd = DOCK_CMD + "\"" + flag_file + "\""
                    a_process = run_cmd(rna_denovo_cmd, speed_run)  
                    status = a_process.poll()
                    # index, primary sequence, secondary sequence, process, scores, stderr, stdout, 
                    # return code, finished running flag, training label
                    running_list.append([index, prim_seq, sec_seq, a_process, "", None, "", status, False, label])

                    
                    # increase index count for correct index in larger list
                    index += 1

                # now read this group of processes' stdin and stderr to collect data and prevent zombification 
                finished = 0
                while True:
                    for cp in running_list:
                        # if process already finished running from last iteration, skip this child process
                        if cp[8] is True: continue
                        
                        # obtain a single line from the running child process
                        output = cp[3].stdout.readline().decode("utf-8")
                        # add output into cp stdout slot
                        cp[6] += " "+output

                        # if the output line starts with "total.." add this score to the score slot in cp
                        if output.lstrip().startswith("Total weighted score:"):
                            cp[4] += " "+output.split()[3]
                        
                        # check if process is still running
                        return_code = cp[3].poll()
                        # return code is "none" if process is still running, if not none, process stopped
                        if return_code is not None:
                            stdout = cp[3].stdout.readlines()
                            cp[5] = cp[3].stderr.readlines()
                            for line in stdout:
                                decoded = line.decode("utf-8")
                                # save remaining lines in stdout slot
                                cp[6] += " "+ decoded
                                # save remaning scores in score slot
                                if decoded.lstrip().startswith("Total weighted score:"):
                                    cp[4] += " "+decoded.split()[3]
                            
                            cp[7] = return_code
                            cp[8] = True
                            cp[3].terminate() 
                            cp[3] = None
                            # take score string, split it by spaces, map them to floats, 
                            # convert them to a list, then take minimum from the list.
                            # Result should be the smallest float
                            cp[4] = min(list(map(float,cp[4].split())))
                            self.update_progress(progress)
                            finished += 1
                    
                    if finished == sub_data_size: break
                
                # remove the temporary directory because this round of iterations are done
                rm(new_dir)

                if debug:
                    data_debug.extend(running_list)
                    debug_df = pd.DataFrame(data_debug, columns=['Index','Sequences','Secondary Structure','Process','Scores','StdErr','StdOut','Return Code','Finish Flag','Training Label'])
                    if i == 0:
                        tqdm.write(f"{text_colour}Writing troubleshoot csv")
                        debug_df.to_csv(DEBUG_DIR+troubleshoot_name, index = False)
                        tqdm.write(f"\n{text_colour}Complete")
                    else:
                        tqdm.write(f"{text_colour}Appending to troubleshoot csv (loop = {MAGENTA}{i}{text_colour})")
                        debug_df.to_csv(DEBUG_DIR+troubleshoot_name, mode = 'a', index = False, header = False)
                        tqdm.write(f"\n{text_colour}Complete")

                # remove columns in running list that should not be written to the dataset
                for row in running_list:
                    del row[8]
                    del row[7]
                    del row[6]
                    del row[5]
                    del row[3]
                
                # update and calculate remaining time
                progress.refresh()
                tot_dp = progress.total
                completed = progress.n
                elapsed_time = progress.format_dict['elapsed']
                total_est_time = tot_dp/completed*elapsed_time
                remaining_est_time = total_est_time - elapsed_time
                rem_est_time_formated = progress.format_interval(remaining_est_time)
                eta = time.asctime( time.localtime( time.time() + remaining_est_time ) )
                started = time.asctime( time.localtime( start_time ) )
                # do not display calculation if last iteration
                if i != loops-1:
                    tqdm.write(f"{text_colour}\n--- Scoring Iteration {MAGENTA}{i}{text_colour}: ---\nStart time: {started} \nElapsed time: {progress.format_interval(elapsed_time)} \nEstimated remaining time: {rem_est_time_formated} \nEstimated completion time: {eta} \nRemaining datapoints: {tot_dp-completed} ")
                
                # add data in order into a list
                data.extend(running_list)
                # write/append output data to csv
                df = pd.DataFrame(running_list, columns=['Index','Sequences','Secondary structure','Scores','Split'])
                if i == 0:
                    tqdm.write(f"\n{text_colour}Writing output")
                    df.to_csv(SAVE_DIR+data_save_name, index = False)
                    tqdm.write(f"{text_colour}Initial write complete\n---------------------------")
                else:
                    tqdm.write(f"\n{text_colour}Appending to output")
                    df.to_csv(SAVE_DIR+data_save_name, mode = 'a', index = False, header = False)
                    tqdm.write(f"{text_colour}Appending complete\n---------------------------")

        if reloading:
            tqdm.write(f"{text_colour}\n--- Returning {MAGENTA}0{RESET} scores due to reloading ---\n")
            return None
        else:
            tqdm.write(f"{text_colour}\n--- Returning scores ---\n")
            df = pd.DataFrame(data, columns=['Index','Sequences','Secondary structure','Scores','Split'])
            return df

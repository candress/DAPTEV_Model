import random
import os, shutil
import torch
import pandas as pd
from genetic_vae import DAPTEV 
from get_scores import Predictions as Pred
from generate_data.random_seq import get_rand_seq as rand_seq
from arnie.mfe import mfe
from tqdm import tqdm
import time
import pickle
from colorama import Fore, Style
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

# ------------------------------------------------------------------------------------------------------------------
# User-specified parameters
'''---User Parameters---'''
# regression style choice: 0 = BCE (uses score threshold), 1 = RMSE (does not use score threshold), 2 = MSE (does not use score threshold)
REGRESS_CHOICE = 0
SAMPLE_POP_SIZE = 100
# 5 for test, 10 for thesis, 1 for checking VAE only
GENERATIONS = 5
RUNS = 1
SEED = 200
# SEED = int(time.time())
# SPEED_RUN_FLAG is the flag to indicate a speed run or not (speed run will not run Rosetta and will simply generate random 
# scores based on a uniform distribution between [-1000, 1000) or [-1000, 1000] (depending on rounding) rounded to 4 decimal places
SPEED_RUN_FLAG = False
# if I want to print the entire output from rosetta or not
ROSETTA_DEBUG_FLAG = False
# only running the VAE to test parameters
VAE_ONLY = False
GA_ONLY = True
# Do I need to reload from a crash?
# TODO NOTE: this is not working for some reason.
RELOADING = False
# file output type (keep as .csv)
OUTPUT_TYPE = '.csv'
LOG_DIRECTORY = 'log/'
ROSETTA_SCORE_OUTPUT_DIRECTORY = 'data/output/'
TENSORBOARD_DIRECTORY = 'tensorboard/'
VAE_LOG_WRITER_DIRECTORY = 'vae_generations/'
SCORE_LOG_WRITER_DIRECTORY = 'dock_score_runs/'
ROSETTA_DEBUG_DIR = 'debug_data/'
SEQ_PRE_REPLACE_DIR = 'pre_replace/'
SEQ_POST_REPLACE_DIR = 'post_replace/'
SEQ_SORT_N_MERGE_DIR = 'sort_n_merged/'

'''---VAE Parameters---'''
# 30 epochs seems to be more than enough to train the VAE and reduce the loss function.
# Any more epochs just results in variations of the 3 loss components and doesn't reduce much more.
# PRE_EPOCHS = initial dataset training 
PRE_EPOCHS = 30
# POST_EPOCHS = number of epochs after the initial training (used for samples per generation)
POST_EPOCHS = 10
BATCH = 32
# determined emperically (3,500 = 25.25% of my training data, 5000 = 39.05%)
SCORE_THRESHOLD  = 5000 
START_LEARN_RATE = 0.003
END_LEARN_RATE   = 0.0003
START_KL_WEIGHT  = 0.001
END_KL_WEIGHT    = 0.05

'''---GA Parameters---'''
TOURN_SELECT_K = 2
SELECT_RATE = 0.95
ELITE_PERCENT = 0.01
CROSS_RATE = 0.9
# usually 1 percent, 10% destroys everything (last test seems to confirm that)
MUT_RATE = 0.01
VOCAB_COMBINATIONS = 3

'''---Rosetta Parameters---'''
MAX_SEQ_LEN = 40
MIN_SEQ_LEN = 20
GC_PERCENT = 0.50
TRAIN_PERCENT = 1.0
# percentage of processors to use
PROCESS_PERCENT = 0.5
# TODO: This parameter is "required" but not actually utilized in this program. It was simply carried over from the dataset creation script. Provided value is irrelevant
ALLOW_UNCONNECTED = False
# tqdm bar colour when predicting scores
ROSETTA_TQDM_COLOUR = "cyan"
CMD = "rna_denovo -constant_seed true -jran "+str(SEED)+" @"

'''---Setting styles---'''
RED = Fore.RED
GREEN = Fore.GREEN
CYAN = Fore.CYAN
YELLOW = Fore.YELLOW
MAGENTA = Fore.MAGENTA
RESET = Style.RESET_ALL

'''---Other settings---'''
# character length to cut off at if too many characters (for name of file)
CHAR_LEN = 6
# "too many characters" threshold
TOO_LONG = 9

# ------------------------------------------------------------------------------------------------------------------
# COMFORT HELPER FUNCTIONS
def get_last_run_folder_path(param_job_dir, suffix_folder_path, last_folder_w_file):
    # get a list of all the folders with current run in their name (between '[' and ']' - first occurence in string)
    # list will be sorted numerically, not alphabetically (alphabetically will prioritize lower following digits (e.g. 0, 1, 10, 2, ...))
    list_of_run_folders = sorted(os.listdir(param_job_dir), key=lambda x: int(x[x.find('[')+1 : x.find(']')]))
    # get the last run (which failed) to repeat generations
    last_run_folder_name = None
    # The file wouldn't exist if we broke on e.g. run 5 gen 0 after dir creation, but before saving the state and data.
    # thus, if we need the last folder with at least one file in it, walk through folders and determine which one this is
    if last_folder_w_file:
        for i, folder_name in enumerate(list_of_run_folders):
            # build a full path string to this folder
            full = param_job_dir + folder_name
            # perform walk to get files in every full path (above)
            for path, directory, files in os.walk(full):
                # if the length of files in this specific directory is greater than 0, then this folder was written to. 
                if len(files) > 0:
                    # Set this as the last folder
                    last_run_folder_name = list_of_run_folders[i]+'/'
                    # stop the walk process for this folder's sub folders
                    break
    # else, just get the last folder
    else:
        last_run_folder_name = list_of_run_folders[-1]+'/'
    
    # create the full path string for the OS to obtain files
    last_run_folder_path_w_suffix = param_job_dir + last_run_folder_name + suffix_folder_path
    # TODO what follows might be an issue for log directory because there are other folders within that do not adhere to the rfind
    
    return last_run_folder_path_w_suffix, last_run_folder_name

def make_dir(directory):
    # if directory doesn't exist, create it
    if not os.path.exists(directory):
        os.mkdir(directory)

def write_sequence_data(folder_name, file_name, data):
    name = folder_name + file_name
    data.to_csv(name, index = False)

def set_parameters(params):
    global RUNS, GENERATIONS, SAMPLE_POP_SIZE, SPEED_RUN_FLAG, VAE_ONLY, GA_ONLY, RELOADING, REGRESS_CHOICE, PRE_EPOCHS, POST_EPOCHS, BATCH, SCORE_THRESHOLD
    global START_LEARN_RATE, END_LEARN_RATE, START_KL_WEIGHT, END_KL_WEIGHT, VOCAB_COMBINATIONS, TOURN_SELECT_K, SELECT_RATE, ELITE_PERCENT, CROSS_RATE
    global MUT_RATE, MIN_SEQ_LEN, MAX_SEQ_LEN, GC_PERCENT, PROCESS_PERCENT, ALLOW_UNCONNECTED, SEED, VAE_PARAMS_DICT, GA_PARAMS_DICT, ROSETTA_PARAMS_DICT

    new_params = params.split(',')

    RUNS, GENERATIONS, SAMPLE_POP_SIZE, = int(new_params[0]), int(new_params[1]), int(new_params[2])
    SPEED_RUN_FLAG =  True if new_params[3].lower() == 'true' else False
    VAE_ONLY = True if new_params[4].lower() == 'true' else False
    ALLOW_UNCONNECTED = True if new_params[24].lower() == 'true' else False
    GA_ONLY = True if new_params[25].lower() == 'true' else False
    RELOADING = True if new_params[26].lower() == 'true' else False
    REGRESS_CHOICE, PRE_EPOCHS, POST_EPOCHS = int(new_params[5]), int(new_params[6]), int(new_params[7])
    BATCH, SCORE_THRESHOLD, START_LEARN_RATE = int(new_params[8]), int(new_params[9]), float(new_params[10])
    END_LEARN_RATE, START_KL_WEIGHT, END_KL_WEIGHT = float(new_params[11]), float(new_params[12]), float(new_params[13])
    VOCAB_COMBINATIONS, TOURN_SELECT_K = int(new_params[14]), int(new_params[15])
    SELECT_RATE, ELITE_PERCENT, CROSS_RATE, MUT_RATE = float(new_params[16]), float(new_params[17]), float(new_params[18]), float(new_params[19])
    MIN_SEQ_LEN, MAX_SEQ_LEN, GC_PERCENT = int(new_params[20]), int(new_params[21]), float(new_params[22])
    PROCESS_PERCENT, SEED = float(new_params[23]), int(new_params[-1])
    

    VAE_PARAMS_DICT = {
        'Pre_Epochs':PRE_EPOCHS,
        'Post_Epochs':POST_EPOCHS,
        'Batch':BATCH,
        'Score_Thresh':SCORE_THRESHOLD,
        'Start_LR':START_LEARN_RATE,
        'End_LR':END_LEARN_RATE,
        'Start_KLW':START_KL_WEIGHT,
        'End_KLW':END_KL_WEIGHT,
        'Vocab':VOCAB_COMBINATIONS
    }
    GA_PARAMS_DICT = {
        'Tourn_K':TOURN_SELECT_K,
        'Select_rate':SELECT_RATE,
        'Elite':ELITE_PERCENT,
        'Cross_Rate':CROSS_RATE,
        'Mut_Rate':MUT_RATE
    }
    ROSETTA_PARAMS_DICT = {
        'Min_Len':MIN_SEQ_LEN,
        'Max_Len':MAX_SEQ_LEN,
        'GC':GC_PERCENT,
        'CPU':PROCESS_PERCENT,
        'Unconnected':ALLOW_UNCONNECTED
    }

def write_hparams_tensorboard(param_dict, title, writer):
    header = '|'
    cols = '|'
    vals = '|'
    for item in param_dict.items():
        header += f'{item[0]}|'
        cols += ':-:|'
        vals += f'{item[1]}|'
        
    params = header + '\n' + cols + '\n' + vals
    writer.add_text('Parameters/'+title, params)
    
def write_hparams_scalar_tensorboard(run, writer):
    usr_param_dict = {
        'Runs':RUNS,
        'Cur_Run':run,
        'Gens':GENERATIONS,
        'Pop':SAMPLE_POP_SIZE,
        'Speed':SPEED_RUN_FLAG,
        'GA_Only':GA_ONLY,
        'VAE_Only': VAE_ONLY,
        'Regress':REGRESS_CHOICE,
        'Reloading':RELOADING,
        'Seed':SEED
    }
        
    param_dict = {}
    param_dict.update(usr_param_dict)
    param_dict.update(VAE_PARAMS_DICT)
    param_dict.update(GA_PARAMS_DICT)
    param_dict.update(ROSETTA_PARAMS_DICT)

    write_hparams_tensorboard(usr_param_dict, 'Usr Params', writer)
    write_hparams_tensorboard(VAE_PARAMS_DICT, 'VAE Params', writer)
    write_hparams_tensorboard(GA_PARAMS_DICT, 'GA Params', writer)
    write_hparams_tensorboard(ROSETTA_PARAMS_DICT, 'Rosetta Params', writer)

    descriptive_name = ''
    seed_text = ''
    for item in param_dict.items():
        if 'Seed' in item[0]: 
            seed_text += f'{item[0]}[{item[1]}]'
            continue
        descriptive_name += f'{item[0]}_{item[1]}-'
    descriptive_name += seed_text
    writer.add_scalar(f'AA_{descriptive_name}',0,0)

def write_scores_tensorboard(sorted_df, current_gen, writer, stats = None):
    tqdm.write("writing scores")
    best_selected_score = sorted_df.iloc[0]['Scores']
    worst_selected_score = sorted_df.iloc[-1]['Scores']
    average_selected_score = sorted_df['Scores'].mean()
    stddev_selected_score = sorted_df['Scores'].std()
    median_selected_score = sorted_df['Scores'].median()
    best_5_score_avg = sorted_df.head(5)['Scores'].mean()
    worst_5_score_avg = sorted_df.tail(5)['Scores'].mean()

# writing values to writer
    writer.add_scalar('Scores/Best_Score', best_selected_score, current_gen)
    writer.add_scalar('Scores/Worst_Score', worst_selected_score, current_gen)
    writer.add_scalar('Scores/Mean', average_selected_score, current_gen)
    writer.add_scalar('Scores/Median', median_selected_score, current_gen)
    writer.add_scalar('Scores/Standard_Deviation', stddev_selected_score, current_gen)
    writer.add_scalar('Scores/Top_5_Avg', best_5_score_avg, current_gen)
    writer.add_scalar('Scores/Worst_5_Avg', worst_5_score_avg, current_gen)
    
# create box plots to see visual of score range over time
    fig, ax = plt.subplots(figsize=(3.5,4.8), tight_layout=True)
    ax.boxplot(sorted_df['Scores'].to_list())
    plt.ylabel('Range of Scores')
    plt.title(f'Scores for Iteration: {current_gen}')
    writer.add_figure(f'Box_Plots/Iteration {current_gen}', fig, current_gen)

# write duplicated sequences and their associated indices 
    if stats is not None:
        writer.add_scalar('Z_Dup_Stats/C_All_pre_exist', stats[0], current_gen)
        writer.add_scalar('Z_Dup_Stats/D_Exists_in_initial_data', stats[1], current_gen)
        writer.add_scalar('Z_Dup_Stats/E_Exists_in_running_data', stats[2], current_gen)
        writer.add_scalar('Z_Dup_Stats/F_Duplicates', stats[3], current_gen)
        writer.add_scalar('Z_Dup_Stats/G_Dups_post_remove_exist', stats[4], current_gen)
        writer.add_scalar('Z_Dup_Stats/A_Novelty', stats[5], current_gen)
        writer.add_scalar('Z_Dup_Stats/B_Diversity', stats[6], current_gen)
        writer.add_scalar('Z_Dup_Stats/H_Total_replaced', stats[7], current_gen)
        # for i, dup in enumerate(duplicates):
        #     writer.add_text(f'Duplicates/Gen_{current_gen}_{i}', str(dup))

# Calling "flush" to make sure all pending events have been written to disk
    writer.flush()

def set_run_names_n_dir(cur_run):
    # convert percentages for output file name string
    gc_amnt = str(GC_PERCENT).replace('.','-')[:len(str(GC_PERCENT)) if len(str(GC_PERCENT)) < TOO_LONG else CHAR_LEN]
    cpu_amnt = str(PROCESS_PERCENT).replace('.','-')[:len(str(PROCESS_PERCENT)) if len(str(PROCESS_PERCENT)) < TOO_LONG else CHAR_LEN]
    elite = str(ELITE_PERCENT).replace('.','-')[:len(str(ELITE_PERCENT)) if len(str(ELITE_PERCENT)) < TOO_LONG else CHAR_LEN]
    cross = str(CROSS_RATE).replace('.','-')[:len(str(CROSS_RATE)) if len(str(CROSS_RATE)) < TOO_LONG else CHAR_LEN]
    mutate = str(MUT_RATE).replace('.','-')[:len(str(MUT_RATE)) if len(str(MUT_RATE)) < TOO_LONG else CHAR_LEN]
    lrs = str(START_LEARN_RATE).replace('.','-')[:len(str(START_LEARN_RATE)) if len(str(START_LEARN_RATE)) < TOO_LONG else CHAR_LEN]
    lre = str(END_LEARN_RATE).replace('.','-')[:len(str(END_LEARN_RATE)) if len(str(END_LEARN_RATE)) < TOO_LONG else CHAR_LEN]
    klws = str(START_KL_WEIGHT).replace('.','-')[:len(str(START_KL_WEIGHT)) if len(str(START_KL_WEIGHT)) < TOO_LONG else CHAR_LEN]
    klwe = str(END_KL_WEIGHT).replace('.','-')[:len(str(END_KL_WEIGHT)) if len(str(END_KL_WEIGHT)) < TOO_LONG else CHAR_LEN]
    sele_rte = str(SELECT_RATE).replace('.','-')[:len(str(SELECT_RATE)) if len(str(SELECT_RATE)) < TOO_LONG else CHAR_LEN]

    user_params_text = f'{RUNS}[{cur_run}]_{GENERATIONS}_{SAMPLE_POP_SIZE}_{SPEED_RUN_FLAG}_{VAE_ONLY}_{REGRESS_CHOICE}'
    vae_params_text = f'{PRE_EPOCHS}_{POST_EPOCHS}_{BATCH}_{SCORE_THRESHOLD}_{lrs}_{lre}_{klws}_{klwe}_{VOCAB_COMBINATIONS}'
    ga_params_text = f'{TOURN_SELECT_K}_{sele_rte}_{elite}_{cross}_{mutate}'
    rosetta_params_text = f'{MIN_SEQ_LEN}_{MAX_SEQ_LEN}_{gc_amnt}_{cpu_amnt}_{ALLOW_UNCONNECTED}'

    output_name = f'{user_params_text}_{vae_params_text}_{ga_params_text}_{rosetta_params_text}_{SEED}'
    rosetta_score_out_dir = f'{ROSETTA_SCORE_OUTPUT_DIRECTORY}{output_name}/'
    log_dir = f'{LOG_DIRECTORY}{output_name}/'

    rosetta_debug_dir = f'{rosetta_score_out_dir}{ROSETTA_DEBUG_DIR}'
    pre_replace_seq_dir = f'{rosetta_score_out_dir}{SEQ_PRE_REPLACE_DIR}'
    post_replace_score_dir = f'{rosetta_score_out_dir}{SEQ_POST_REPLACE_DIR}'
    sort_n_merge_dir = f'{rosetta_score_out_dir}{SEQ_SORT_N_MERGE_DIR}'

    # if directories don't exist, create them
    make_dir(ROSETTA_SCORE_OUTPUT_DIRECTORY)
    make_dir(LOG_DIRECTORY)
    make_dir(rosetta_score_out_dir)
    make_dir(log_dir)
    make_dir(rosetta_debug_dir)
    make_dir(pre_replace_seq_dir)
    make_dir(post_replace_score_dir)
    make_dir(sort_n_merge_dir)
    
    return log_dir, rosetta_debug_dir, pre_replace_seq_dir, post_replace_score_dir, sort_n_merge_dir, rosetta_score_out_dir

def display_eta(progress_bar, start_time, colour):
    total_data_points = progress_bar.total
    completed = progress_bar.n
    elapsed_time = progress_bar.format_dict['elapsed']
    total_est_time = total_data_points/completed*elapsed_time
    remaining_est_time = total_est_time - elapsed_time
    rem_est_time_formated = progress_bar.format_interval(remaining_est_time)
    eta = time.asctime( time.localtime( time.time() + remaining_est_time ) )
    started = time.asctime( time.localtime( start_time ) )
    tqdm.write(f"\n{colour}--- {progress_bar.desc}: {MAGENTA}{completed}{colour}: ---\nStart time: {started} \nElapsed time: {progress_bar.format_interval(elapsed_time)} \nEstimated remaining time: {rem_est_time_formated} \nEstimated completion time: {eta}{RESET}\n")

# ------------------------------------------------------------------------------------------------------------------
# OPERATION FUNCTIONS
def get_state(file_path):
    with open(file_path, 'rb') as infile:
        state_dict = pickle.load(infile)
        infile.close()
    return state_dict

def load_prev_data_del_tensorboard(param_job_seq_dir, param_job_log_dir):
    global reloaded_data
    global python_random_state
    global pytorch_random_state
    global last_run_value

    # suffix folder name for sequence data output
    seq_data_folder_name = 'sort_n_merged/'
    
    # get the last folder rosetta score folder that actually has a state written to it
    last_run_folder_path_w_suffix, last_run_csv_folder_name = get_last_run_folder_path(param_job_seq_dir, seq_data_folder_name, True)
    # create a list of file names with their associated prefix full paths from cwd IN THE LAST RUN FOLDER. 
    # Will be sorted first to last (0 - last gen)
    sorted_list_of_files = sorted(os.listdir(last_run_folder_path_w_suffix), key=lambda file_name: int(file_name[file_name.rfind('[')+1 : file_name.rfind(']')]))
    list_of_csv_file_paths = [last_run_folder_path_w_suffix + file_name for file_name in sorted_list_of_files]

    # load the reloaded_data list with csv dataframes
    reloaded_data = [pd.read_csv(csv_path) for csv_path in list_of_csv_file_paths]

    # get the name of the state file
    # TODO what if we broke on run 5 gen 0 after dir creation, but before saving the state or any data? The file wouldn't exist.
    # This has been addressed in the get_file_paths function
    # TODO what if we broke on run 5 gen 0 after dir creation and saving the state, but before saving data? Then the reloaded data is size 0, ending the "RELOADING" flag
    state_file_name = [name for name in os.listdir(param_job_seq_dir + last_run_csv_folder_name) if 'Run_' in name][0]
    # load in the last state and set variables
    loaded_state = get_state(param_job_seq_dir + last_run_csv_folder_name + state_file_name)
    last_run_value, python_random_state, pytorch_random_state = loaded_state.values()
    # TODO this line above might be the issue why this isn't working. It might also be due to issues with Pytorch not being able to guarantee same randomness.
    # TODO Try the following 3 lines
    # last_run_value = loaded_state['run']
    # python_random_state = loaded_state['rand_state']
    # pytorch_random_state = loaded_state['torch_state']

    # delete log event files
    # create a list of csv names with their associated prefix full paths from cwd
    last_run_tensorboard_path, _ = get_last_run_folder_path(param_job_log_dir, '', False)
    # delete the contents inside the RUN folder of the log directory
    last_run_tensorboard_path_w_suffix = last_run_tensorboard_path + TENSORBOARD_DIRECTORY
    if os.path.exists(last_run_tensorboard_path_w_suffix):
        shutil.rmtree(last_run_tensorboard_path_w_suffix)

def save_state(run, rand_state, pytorch_rand_state, state_dir):
    state_dict = {'run':run, 'rand_state':rand_state, 'torch_state':pytorch_rand_state}
    with open(state_dir+f'Run_{run}_state','wb') as outfile:
        pickle.dump(state_dict,outfile)
        outfile.close()

def replace_duplicates(children):
    global running_gen_seq
    
    # get ids of the children which exist in the running_gen_seq (including previous generations)
    idx_in_csv = []
    idx_in_running = []
    for i, seq in enumerate(children):
        in_original_data = (dataset_csv['Sequences'] == seq).any()
        if in_original_data:
            idx_in_csv.append(i)
        elif seq in running_gen_seq:
            idx_in_running.append(i)
    
    # get unfiltered children and obtain only the unique sequences
    new_sequences = []
    new_sequences.extend(children)
    # sets don't allow duplicates
    unique_sequences = list(set(new_sequences))

    # novelty = number of generated sequences that do not exist in training data vs total amount of generated samples
    novelty = (len(children)-len(idx_in_csv))/SAMPLE_POP_SIZE
    # diversity = percentage of generated unique sequences among total number of generated sequences
    diversity = len(unique_sequences)/SAMPLE_POP_SIZE
    # calculate number of duplicates (may include sequences in initial data set and running_gen_seq)
    num_total_dups = len(new_sequences)-len(unique_sequences)
    # num existing in JUST initial dataset
    num_exist_initial_dataset = len(idx_in_csv)
    # num existing in JUST running dataset
    num_exist_running_dataset = len(idx_in_running)
    # num children exist EITHER in initial data set or in running_gen_seq
    total_num_pre_exist = num_exist_initial_dataset + num_exist_running_dataset

    # remove sequences that already exist in EITHER the initial data set or in running_gen_seq and add them to discarded sequences
    idx_in_running.extend(idx_in_csv)
    idx_in_running.sort()
    discarded_sequences = []
    for i in reversed(idx_in_running):
        discarded_sequences.append(children.pop(i))

    # now that the already existing sequences have been removed, get unique sequences from this list to calc remaining duplicates
    new_sequences = []
    new_sequences.extend(children)
    unique_sequences = list(set(new_sequences))
    # calculate the number of duplicates there were from the post-existing to unique conversion
    num_remaining_dups = len(new_sequences)-len(unique_sequences)
    # add sequences that aren't already in the initial data set or in running_gen_seq, and aren't duplicated thereafter into the running sequence list
    running_gen_seq.extend(unique_sequences)

    children.clear()
    children.extend(unique_sequences)
    total_to_replace = SAMPLE_POP_SIZE - len(children)
    tqdm.write(f'{total_to_replace} sequences are being replaced. Please wait.')
    for _ in tqdm(range(total_to_replace), desc = 'Replacing Dups'):
        while True:
            sequence = rand_seq(MIN_SEQ_LEN, MAX_SEQ_LEN, GC_PERCENT) 
                # make sure sequence doesn't already exist in any lists
            in_original_data = (dataset_csv['Sequences'] == sequence).any()
            if not in_original_data:
                if sequence not in running_gen_seq:
                    if sequence not in children:
                        # for some reason, having this calculation and check right after getting the sequence stalled the program. 
                        # I suspect it's because the mfe does a popen command which laggs behind as python's random is too quick
                        sec_seq = mfe(sequence, package='eternafold')
                        # make sure there is atleast 1 connection in the secondary structure
                        if sec_seq.count("(") > 0:
                            break
        children.append(sequence)
        running_gen_seq.append(sequence)
    
    # all = list(zip(exist_seq_idx,reversed(discarded_sequences)))
    tqdm.write(f"{MAGENTA}{total_num_pre_exist}{RESET} sequences already existed in previously loaded and/or generated data.")
    tqdm.write(f'{MAGENTA}{num_total_dups}{RESET} total duplicates in this round of generated data.')
    tqdm.write(f'{MAGENTA}{num_exist_initial_dataset}{RESET} sequences exist in initial dataset.')
    tqdm.write(f'{MAGENTA}{num_exist_running_dataset}{RESET} sequences exist in running dataset.')
    tqdm.write(f'{MAGENTA}{num_remaining_dups}{RESET} remaining duplicates in this round of generated data.')
    tqdm.write(f'{MAGENTA}{total_to_replace}{RESET} sequences replaced with random sequences.')
    tqdm.write(f'Novelty: {MAGENTA}{novelty}{RESET}, number of generated sequences that do not exist in training data vs total amount of generated samples.')
    tqdm.write(f'Diversity: {MAGENTA}{diversity}{RESET}, percentage of generated unique sequences among total number of generated sequences.\n')

    return (total_num_pre_exist, num_exist_initial_dataset, num_exist_running_dataset, num_total_dups, num_remaining_dups, novelty, diversity, total_to_replace)

def score_sequences(predictor, data_list, sorted_parents, run, current_gen, score_writer, pre_dir, sort_n_merge_dir):
    global RELOADING
    global reloaded_data

    run_n_gen = f'Run[{run}]_Gen[{current_gen}]'
    # write sequences before replacing data
    temp_df = pd.DataFrame(data_list, columns=['Sequences'])
    write_sequence_data(pre_dir, run_n_gen+OUTPUT_TYPE, temp_df)

    dup_stats = replace_duplicates(data_list)   
    
    if VAE_ONLY:
        global data_rem
        data = data_rem.head(SAMPLE_POP_SIZE)
        data_rem = data_rem.iloc[SAMPLE_POP_SIZE:]
        data = data.sort_values(by=['Scores'])
    else:
        if len(reloaded_data) == 0:
            RELOADING = False
        
        tqdm.write(f"{CYAN}\n----- Obtaining scores from Rosetta -----")
        tqdm.write(f"{CYAN}This may take a while depending on the length of your sequences, the size of the docking target, and min(cpu amount used [{predictor.max_num_processes}], data size [{SAMPLE_POP_SIZE}])")
        # TODO right now this will create multiple MASSIVE files. Need to find a more space-efficient way of saving troubleshoot info (check IF error ?)
        data = predictor.get_scores(data_list, run_n_gen, run_n_gen, SPEED_RUN_FLAG, ROSETTA_DEBUG_FLAG, ROSETTA_TQDM_COLOUR, CYAN, RELOADING)
        tqdm.write(f"{YELLOW}\nScoring complete for Run {MAGENTA}{run}{YELLOW}, Generation {MAGENTA}{current_gen}.")
        tqdm.write(f"{YELLOW}Now merging datasets and retreiving best sequences.\n")

        if RELOADING:
            # TODO I'm popping a dataframe right now. Need to make sure it is what I expect to return
            data = reloaded_data.pop(0)
        else:
            parents_df = pd.DataFrame(sorted_parents, columns=['Sequences','Secondary structure','Scores','Split'])
            data = data.set_index('Index')
            data = pd.concat([parents_df,data], ignore_index = True)
            data = data.sort_values(by=['Scores'])
            data = data.head(SAMPLE_POP_SIZE)
            # write data again to get the the sorted, selected, data (will agree with tensorboard)
            write_sequence_data(sort_n_merge_dir, run_n_gen+OUTPUT_TYPE, data) 
    
    write_scores_tensorboard(data, current_gen + 1, score_writer, dup_stats)

    return data

# ------------------------------------------------------------------------------------------------------------------
#* MAIN FUNCTION
def run_parameter_set():
    global data    

    # Setting Seed
    random.seed(SEED)
    torch.manual_seed(SEED)

    # Obtain starting time
    run_start_time = time.time()

    run_progress = tqdm(range(RUNS), desc=f"{GREEN}Run", colour = "green", mininterval=0)
    for run in run_progress:
        if RELOADING:
            # get to correct run, set rng states after that
            if run < last_run_value:
                tqdm.write(f'{RED}SKIPPED run {run}')
                continue
            random.setstate(python_random_state)
            torch.set_rng_state(pytorch_random_state)
        else:
            # display estimated remaining time for all runs if this is not the first run
            if run != 0:
                display_eta(run_progress, run_start_time, GREEN)

        # create directories if they don't exitst, return important directory names
        log_directory, rosetta_debug_dir, pre_replace_seq_dir, post_replace_score_dir, sort_n_merge_dir, run_dir = set_run_names_n_dir(run)

        # save the current rng state if we aren't reloading from a crash
        if not RELOADING:
            save_state(run, random.getstate(), torch.get_rng_state(), run_dir)

        tqdm.write(f'\n{GREEN}----- Current Run = {MAGENTA}{run}{GREEN} -----')
        tqdm.write(f'Start Time: {MAGENTA}{time.asctime( time.localtime( run_start_time ) )}')
        score_writer = SummaryWriter(log_directory + TENSORBOARD_DIRECTORY + SCORE_LOG_WRITER_DIRECTORY + str(run))
        
        # Setting up and writing dictionaries for parameter choices
        write_hparams_scalar_tensorboard(run, score_writer)

        # write the initial score values from starting data set
        start_data_sorted_copy = data.copy().sort_values(by=['Scores'])
        write_scores_tensorboard(start_data_sorted_copy, 0, score_writer)    

        # create the model
        my_model = DAPTEV(PRE_EPOCHS, BATCH, VOCAB_COMBINATIONS, SCORE_THRESHOLD, SAMPLE_POP_SIZE, MAX_SEQ_LEN, TOURN_SELECT_K, ELITE_PERCENT, CROSS_RATE, MUT_RATE, SELECT_RATE,
                        START_LEARN_RATE, END_LEARN_RATE, START_KL_WEIGHT, END_KL_WEIGHT, REGRESS_CHOICE)

        # create a new instance of the prediction class (mainly to create new file names, read in protein data, and perform related calcs once)
        predictor = Pred(CMD, TRAIN_PERCENT, PROCESS_PERCENT, OUTPUT_TYPE, post_replace_score_dir, rosetta_debug_dir) 
        
        gen_progress = tqdm(range(GENERATIONS), desc=f"{YELLOW}Generation", colour = "yellow")
        gen_start_time = time.time()

        for current_gen in gen_progress:
            # display estimated remaing time for all generations in this run if this is not the first or last generation
            if current_gen != 0 and current_gen != GENERATIONS-1:
                display_eta(gen_progress, gen_start_time, YELLOW)
            
            if GA_ONLY:
                data, sorted_parents = my_model.train_model(data, ga_only = GA_ONLY)
            else:
                vae_writer = SummaryWriter(log_directory + TENSORBOARD_DIRECTORY + VAE_LOG_WRITER_DIRECTORY + str(current_gen))
                data, sorted_parents = my_model.train_model(data, vae_writer = vae_writer)
                my_model.my_trainer.n_epoch = POST_EPOCHS
            
            # score the sequences and modify provided data list in place 
            data = score_sequences(predictor, data, sorted_parents, run, current_gen, score_writer, pre_replace_seq_dir, sort_n_merge_dir)

        # just making sure the score writter closes (already flushed in genetic_vae.py)
        score_writer.close()
        
        
        


'''------------- BEGIN ITERATIONS -------------'''
parameters = open('parameters.csv','r').read().splitlines()
# do away with the header
parameters = parameters[1:]
num_jobs = 0
headings = []
for line in parameters:
    if line.startswith('-'):
        headings.append(line.strip('-'))
    else:
        num_jobs += 1

# get data
# read_in_data = pd.read_csv("data/start/Output_DataSize_2000_Min_20_Max_40_GCAmnt_50_CPUPercent_75_Uncon_False_Seed_200_Covid.csv",  index_col=0)
dataset_csv = pd.read_csv("data/start/Output_DataSize_12000_Min_20_Max_40_GCAmnt_50_CPUPercent_75_Uncon_False_Seed_200_Covid.csv",  index_col=0)
# list to store the sequences generated over time
running_gen_seq = []
# list to store the dataframes of sequences from the most recent parameter set (used for reloading data) 
reloaded_data = []
python_random_state, pytorch_random_state, last_run_value = None, None, None

temp_log_dir = LOG_DIRECTORY
temp_rosetta_scr_out_dir = ROSETTA_SCORE_OUTPUT_DIRECTORY

jobs_start_time = time.time()
parameter_progress = tqdm(range(num_jobs), desc=f"{RED}Job", colour = "red")
for i, line in enumerate(parameters):
    if line.startswith('-'):
        suffix = line.lstrip('-') + '/'
        LOG_DIRECTORY = temp_log_dir + suffix
        ROSETTA_SCORE_OUTPUT_DIRECTORY = temp_rosetta_scr_out_dir + suffix
        parameter_progress.desc = f"{RED}Job_{headings.pop(0)}"
        continue

    if RELOADING:
        tqdm.write(f'{MAGENTA}RELOADING is set to {CYAN}True{MAGENTA}.\nThis will skip Rosetta predictions and will use previous checkpoint data, overwriting VAE predictions')
        # read in all previous sort_n_merge data
        # Note: it is ok to do this because the user will modify parameters.csv to be the job that failed. If the first sort_n_merge didn't write, just start over.
        load_prev_data_del_tensorboard(ROSETTA_SCORE_OUTPUT_DIRECTORY, LOG_DIRECTORY)

    set_parameters(line)

    data = dataset_csv.copy()
    if VAE_ONLY:
        data_length = data.size
        tail_amnt = GENERATIONS * SAMPLE_POP_SIZE
        data_rem = data.tail(tail_amnt)
        data = data.head(data_length-tail_amnt)

    # running code
    run_parameter_set()
    
    # update progress
    parameter_progress.update()
    if i != len(parameters)-1:
        display_eta(parameter_progress, jobs_start_time, RED)

    LOG_DIRECTORY = temp_log_dir
    ROSETTA_SCORE_OUTPUT_DIRECTORY = temp_rosetta_scr_out_dir
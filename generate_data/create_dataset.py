from write_file import *
from random_seq import *
from command import *
from arnie.mfe import mfe
import pandas as pd
import multiprocessing as mp
import time
import math
from tqdm import tqdm
import read_in_data as rd
from shutil import rmtree as rm
from colorama import Fore, Style

# ------------------------------------------------------------------------------------------------------------------
# User-specified parameters
MIN_SEQ_LEN = 20
MAX_SEQ_LEN = 40
GC_PERCENT = 0.50
ALLOW_UNCONNECTED = False
SEED = 200
DATA_SIZE = 10000
TRAIN_PERCENT = 1.0
PROCESS_PERCENT = 0.5
# SPEED is the flag to indicate a speed run or not (speed run will not run Rosetta and will simply generate random 
# scores based on a uniform distribution between [-1000, 1000) or [-1000, 1000] (depending on rounding) rounded to 
# 4 decimal places
SPEED = False
# if I want to print the entire output from rosetta or not
ROSETTA_DEBUG_DATA = True
# ------------------------------------------------------------------------------------------------------------------
# user - DO NOT TOUCH ANYTHING AFTER THIS LINE ^

# file names
SUB_DIR = "predictions/"
FASTA = SUB_DIR+"fasta.txt"
SECONDARY = SUB_DIR+"secstruct.txt"
FLAGS = SUB_DIR+"flags"
CONSTRAINT = SUB_DIR+"constraint.cst"

# number of iterations
random.seed(SEED)
MAX_NUM_PROCESSES = int(mp.cpu_count()*PROCESS_PERCENT)
LOOPS = math.ceil(DATA_SIZE/MAX_NUM_PROCESSES)
DOCK_CMD = "rna_denovo "
CMD_SEED = "-constant_seed true -jran "+str(SEED)+" @"
DOCK_CMD = DOCK_CMD+CMD_SEED
CONNECT_MSG = "" if ALLOW_UNCONNECTED else "had at least one connection (a bracket) in each secondary structures, "

# obtain all rna_denovo input
# (Note: fasta and secondary file should not have any RNA info in it yet)
fasta_data = rd.read_in(FASTA)
secondary_data = rd.read_in(SECONDARY)
flag_data = rd.read_in(FLAGS)
constraint_data = rd.read_in(CONSTRAINT)

# obtain the fasta header
fasta_header = fasta_data[0].rstrip()
# get last letter of fasta header (split list based on ":", then get the 2nd last 
# element in the list and grab the last character - should be the last letter)
last_letter = fasta_header.split(":")[-2][-1:]
# should always include the last chain as the rna sequence - no matter what
# ord will return the unicode value, chr will convert it back into a letter
rna_chain_letter = chr(ord(last_letter)+1)
# convert percentages for output string (multiplied and turned into an integer)
gc_amnt = int(GC_PERCENT*100)
cpu_amnt = int(PROCESS_PERCENT*100)

# Setting styles
RED = Fore.RED
GREEN = Fore.GREEN
CYAN = Fore.CYAN
YELLOW = Fore.YELLOW
RESET = Style.RESET_ALL

# obtain the protein sequence from the 2nd line in fasta_data 
# by breaking it into a list divided by case, then returning the capital letters
'''PROTEIN = re.findall('[A-Z]+|[a-z]+', fasta_data[1])[0]'''
PROTEIN = fasta_data[1]
# create as many dots as there are in the protein sequence length
DOTS = "."*len(PROTEIN)
SAVE_DIR = 'data/start/'
LOG_DIR = 'log/'
DATA_CSV_NAME = f'Output_DataSize_{DATA_SIZE}_Min_{MIN_SEQ_LEN}_Max_{MAX_SEQ_LEN}_GCAmnt_{gc_amnt}_CPUPercent_{cpu_amnt}_Uncon_{ALLOW_UNCONNECTED}_Seed_{SEED}_Covid.csv'
TROUBLESHOOT_CSV = 'DebugData_'+DATA_CSV_NAME
# ------------------------------------------------------------------------------------------------------------------

# read in starting data
all_data = []
print(YELLOW+"\nPlease note: Currently, this program only supports RNA data (no DNA data). \nAny following instructions referring to DNA or sequences should be interpreted as RNA data only [2022-04]")
print("At any point throughout the running of this program, you may press ctrl+c to cancel execution. \nIf you do, you may need to delete a temporary folder found in the \"predictions\" folder")
while True:
    usr_input = input(GREEN+"\nDo you have data you would like to import (y/n)?\nNote: format must be a csv file with primary sequences as a string (e.g. augccaugga) and secondary structures in dot-bracket notation\n"+RESET).lower()
    if usr_input== 'y':
        print(YELLOW+f"\nCurrent working directory is:"+CYAN+f"\n{os.getcwd()}"+RESET)
        # keep allowing the user to read in more data
        while True:
            # if the user enters an incorrect path or file name, catch and try again
            while True:
                temp_usr_data = input(GREEN+"\nPlease specify the file name to read in.\nInclude folders names deeper within current working directory (e.g. folder_name/myfile.csv).\nNote: it is assumed your data has a header. If this is not included, the first row of your data will be treated as a header.\n"+RESET)
                try:
                    the_df = pd.read_csv(temp_usr_data)
                except:
                    print(RED+"\nAn error occured when trying to read in the csv file. \nPlease make sure the file exists, that is is of type csv, that the file is not empty, that you are specifying the file relative to your current working directory (including deeper folders), and that everything is spelled correctly."+RESET)
                    continue
                break

            df_list = the_df.values.tolist()
            # check that the file size is not 0
            if len(df_list) == 0:
                print(RED+"\nThe file is empty. Please double check the file and try again."+RESET)
                continue

            # which columns are for primary and secondary
            while True:
                # what are the columns for primary and secondary, repeat if not a positive integer (or if 0)
                while True:
                    PRIM_col = input(GREEN+"\nPlease specify column number of PRIMARY sequences (1-based indexing - e.g. 1, 2, 3, etc.):\n"+RESET)
                    if PRIM_col.isnumeric() is False or int(PRIM_col) == 0:
                        print(RED+"\nThe entered value is not a positive integer or is 0. Please try again."+RESET)
                    else: 
                        break
                while True:
                    SEC_col = input(GREEN+"\nPlease specify column number of SECONDARY structure (1-based indexing - e.g. 1, 2, 3, etc.):\n"+RESET)
                    if SEC_col.isnumeric() is False or int(SEC_col) == 0:
                        print(RED+"\nThe entered value is not a positive integer or is 0. Please try again."+RESET)
                    else: 
                        break
                
                # check that the entered column numbers for primary and secondary actually contain only primary and secondary data
                all_good = True
                PRIM_col = int(PRIM_col) - 1
                SEC_col = int(SEC_col) - 1
                for row in df_list:
                    primary = row[PRIM_col].lower()
                    secondary = row[SEC_col]
                    total_DNA = primary.count('a')+primary.count('c')+primary.count('g')+primary.count('t')
                    # if sequence is RNA, not DNA, then count for 't' will be 0. Can just add count for 'u'
                    total_RNA = total_DNA + primary.count('u')
                    total_secondary = secondary.count('.') + secondary.count('(') + secondary.count(')')
                    # if lenth of sequence string does not match the count of either the RNA or DNA sequence characters, there is an issue
                    if total_DNA != len(primary) and total_RNA != len(primary):
                        all_good = False
                        # for to be inclusive of DNA data when this is implimented
                        # print("\nColumn contains characters other than RNA or DNA sequence characters (e.g. a, c, g, t, u). Please try again.")
                        print(RED+"\nOne or more of your primary sequences contains characters other than RNA sequence characters (a, c, g, u). Please confirm column number of primary sequences try again."+RESET)
                        break
                    # if length of secondary string does not match the count of secondary characters, there is an issue
                    if total_secondary != len(secondary):
                        all_good = False
                        print(RED+"\nOne or more of your secondary structures contains characters other than \".\", \"(\", and \")\". Please confirm column number of secondary structures and try again."+RESET)
                        break
                    # if secondary character count does not match either RNA or DNA count, there is an issue
                    if total_secondary != total_DNA and total_secondary != total_RNA:
                        all_good = False
                        print(RED+"\nAmount of characters in at least one of your primary sequence entries and their associated secondary structure entries do not match. Please confirm the primary sequences and their associated secondary structures are each in the same row and try again."+RESET)
                        break
                # if any entered data has errors in matching (any 1 string is combo RNA and DNA or seq len doesn't match 2nd struct len), try again (error already printed)
                if all_good is False: continue

                # FILTER DATA!!!!!
                # else, if all is good so far, filter the entered data based on user-entered parameters
                else: 
                    print(YELLOW+f"\nFile size = "+CYAN+f"{len(df_list)}"+YELLOW+" sequences"+RESET)
                    # Note: Need accepted. accepted = number of items added to list for a provided file, List size = all sequences added over all files provided
                    accepted = 0
                    failed_max = 0
                    failed_min = 0
                    failed_connected = 0
                    already_exists = 0
                    for row in df_list:
                        # get primary sequence
                        primary = row[PRIM_col].lower()
                        # if entered data is not within min and max size, skip it
                        if len(primary) > MAX_SEQ_LEN :
                            failed_max += 1
                            continue
                        if len(primary) < MIN_SEQ_LEN: 
                            failed_min += 1
                            continue
                        # get secondary sequence
                        secondary = row[SEC_col]
                        # if the user chose not to allow unconnected secondary structures
                        if not ALLOW_UNCONNECTED:
                            # if all dots, skip it
                            if secondary.count("(") == 0: 
                                failed_connected += 1
                                continue
                        # if primary 
                        if [primary, secondary] in all_data:
                            already_exists += 1
                            continue
                        # if passed checks, then add to list
                        all_data.append([primary, secondary])
                        accepted += 1
                    
                    print(YELLOW+f"\nFrom this file, there was a total of "+CYAN+f"{accepted} "+YELLOW+f"sequences that {CONNECT_MSG}met the specified minimum ("+CYAN+f"{MIN_SEQ_LEN}"+YELLOW+") and maximum ("+CYAN+f"{MAX_SEQ_LEN}"+YELLOW+") sequence lengths, and did not already exist in either the provided file (duplicate) or in previously loaded data.")
                    print("The following is a breakdown of the sequences that were NOT accepted:")
                    print(f" - Number of sequences smaller than Min = "+CYAN+f"{failed_min}"+YELLOW)
                    print(f" - Number of sequences larger than Max = "+CYAN+f"{failed_max}"+YELLOW)
                    # if the user chose not to allow unconnected secondary structures
                    if not ALLOW_UNCONNECTED:
                        print(f" - Number of sequences with unconnected secondary structures = "+CYAN+f"{failed_connected}"+YELLOW)
                    print(f" - Number of duplicated sequences = "+CYAN+f"{already_exists}"+YELLOW)
                    print(f"\ncurrent size of accepted data is "+CYAN+f"{len(all_data)}"+YELLOW+" sequences"+RESET)
                    break
            
            # ask if the user wants to add more data or not
            while True:
                usr_choice = input(GREEN+"\nDo you have more data in another file to add (y/n)?\n"+RESET).lower()
                if usr_choice != 'y' and usr_choice != 'n':
                    print(RED+"\nSelection is not an option. Please try again."+RESET)
                else: break
            if usr_choice == 'y': continue
            else: break
        
    # else if the user does not want to enter data, leave the loop        
    elif usr_input == 'n': break
    else: print(RED+"\nSelection is not an option. Please try again."+RESET)

    # if the user does not want to enter additional data, leave the loop        
    if usr_choice == 'n': break
 
# ------------------------------------------------------------------------------------------------------------------

# fill rest of sequences and secondary structures
if len(all_data) > DATA_SIZE:
    print(YELLOW+f"\nLoaded too much data. Max data size = "+CYAN+f"{DATA_SIZE}"+YELLOW+". Current size = "+CYAN+f"{len(all_data)}"+YELLOW+". Shuffling list and removing remainder.")
    random.shuffle(all_data)
    # remove all datapoints after DATA_SIZE
    del all_data[DATA_SIZE:]
    print(YELLOW+f"\nNew data size = "+CYAN+f"{len(all_data)}"+YELLOW)
else:
    difference = DATA_SIZE - len(all_data)
    print(YELLOW+"\nComputing remaining number of sequences each with at least one connection in secondary structure (at least one set of brackets)")
    print(f"Remaining sequences to fill: data size "+CYAN+f"{DATA_SIZE}"+YELLOW+" - uploaded data amount that meets criteria "+CYAN+f"{len(all_data)} "+YELLOW+"= "+CYAN+f"{difference}"+YELLOW)
    print("Note: This might take a few seconds")
    for i in range(difference):
        # discard all sequences who's secondary structure is all dots (unconnected)
        while True:
            # get a random rna sequence
            prim_seq = get_rand_seq(MIN_SEQ_LEN,MAX_SEQ_LEN,GC_PERCENT)
            # get the secondary structure
            sec_seq = mfe(prim_seq,package='eternafold')
            # if there is atleast one connection in the secondary structure and the sequence does not already exist in dataset, break forever loop
            if sec_seq.count("(") > 0: 
                if [prim_seq, sec_seq] not in all_data:
                    break
        all_data.append([prim_seq,sec_seq])
    random.shuffle(all_data)

# ------------------------------------------------------------------------------------------------------------------

print("\n--------------------------------------------------------")
print("Now starting 3D structure prediction, docking simulation, and dataset creation"+RESET)
print()

progress = tqdm(total = DATA_SIZE)
data = []


# TODO WILL NEED TO MODIFY THIS LATER (also inlude new command line options for seed)
# add this to the "flags" file during the last round of scoring (may need to be done sequentially due to race conditions, or better yet change output location somehow each time)
# best if I append the iteration number so I can have multiple out files.
command_final = "rna_denovo @flags -out:file:silent iteration_x_.out"

'''# specifying the tutorial-specific information (to make testing quicker)
tutorial_dots = "................................................................................................................................................................................................................................................"
tutorial_sequence = "SMSKTIVLSVGEATRTLTEIQSTADRQIFEEKVGPLVGRLRLTASLRQNGAKTAYRVNLKLDQADVVDSGLPKVRYTQVWSHDVTIVANSTEASRKSLYDLTKSLVATSQVEDLVVNLVPLGRSKTIVLSVGEATRTLTEIQSTADRQIFEEKVGPLVGRLRLTASLRQNGAKTAYRVNLKLDQADVVPKVRYTQVWSHDVTIVANSTEASRKSLYDLTKSLVATSQVEDLVVNLVPLGR"
tutorial_fasta_header = ">2qud_unbound_align5.pdb  A:3-125 B:126-242 C:"
tutorial_C_chain = ""
# specifying the covid-specific information
covid_dots = "............................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................"
covid_protein_sequence = "LPFNDGVIFGTTLDSKTQSLLIIKVCEFNCTFEYVFKNIDGYFKIYLVDLPIGINITRFESIVRFPNITNCPFGEVFNATRFASVYAWNRKRISCVADYSVLYNSASFSTFKCYGVSPTKLNDLFTVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTCVIAWNSNNLDSKGNYNYLYRKPFERDIYFPLQSYGFQPTNVGYQPYRVVVLSFELLPATVCGPKKSTNLNNFNGLTPQRDPQTLEEAISDILSRLDPPEAEVQKDLPFNDGVYWIFGTTLDSKTQSLLIVVIKVCEFQNCTFEYSFVFKNIDGYFKIYSPLVDLPIGINITRFGYLQESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKGNYNYLYRKPFERDIYFPLQSYGFQPTNVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVNNFNGLTGPFQRDPQTLEEAINDILSRLDPPEAEVQKPFNDGVIFGTTLDSKTQSLLIVIKVCEFQNCTFEYVSVFKNIDGYFKIYSPLVDLPIGINITRFGLSVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTKIADYNYKLPDDFTGCVIAWNSNNLDSKGNYNYLYRKPFERDIYFPLQSYGFQPTNVGYQPYRVVVLSFELLHAPATVCGPKKSTNNNGLPQTTEAISNDILSRLDPPEAEVQI"
covid_fasta_header = ">6vxx_ABC_covid_align5.pdb  A:1-266 B:267-557 C:"
covid_C_chain = "558-828"'''

# ------------------------------------------------------------------------------------------------------------------

def update_progress():
    progress.update()
# ------------------------------------------------------------------------------------------------------------------

index = 0
for i in range(LOOPS):
    data_debug = []
    # calculate number of processes remaining to run
    rem_proc = DATA_SIZE - i * MAX_NUM_PROCESSES
    # if remaining number of processes to run is less than the number of processes, set to
    # remaining number of processes to finish it off. Else, loop through the number of 
    # processes
    iterations = rem_proc if rem_proc < MAX_NUM_PROCESSES else MAX_NUM_PROCESSES
    # running list is the currently running processes. 
    running_list = []

    # create tmp folder with current time in seconds (no decimal) under current directory
    temp_dir = ''.join(['tmp_',str(int(time.time()))])
    new_dir = SUB_DIR+temp_dir
    os.mkdir(new_dir)

    # run all commands and get scores
    for _ in range(iterations):
        both = all_data.pop()
        # get a random rna sequence
        prim_seq = both[0]
        # get the secondary structure
        sec_seq = both[1]
        # get the index for the middle of the RNA after the protein
        mid_rna_index = str(len(PROTEIN) + int(len(prim_seq)/2))

        # get the rna chain range and update the fasta header
        rna_chain_range = str(len(PROTEIN)+1)+'-'+str(len(PROTEIN)+len(prim_seq))

        # set WHERE the constraint should take place on the RNA
        temp = constraint_data[0].split()
        temp[4] = mid_rna_index
        temp = " ".join(temp)
        
        # write out the constraint file for this subprocess
        path_txt = [temp_dir, '/', 'constraint', '_', str(index), '.', 'cst']
        const_file = ''.join(path_txt)
        write_data(SUB_DIR+const_file,temp)

        # write new fasta file
        path_txt[2] = 'fasta'
        path_txt[6] = 'txt'
        rna_chain = ' ' + rna_chain_letter + ':' + rna_chain_range
        fasta_file = ''.join(path_txt)
        write_fasta(SUB_DIR+fasta_file,fasta_header,rna_chain,PROTEIN,prim_seq)
        
        # overwrite the secondary structure file with new data 
        path_txt[2] = 'secstruct'
        path_txt[6] = 'txt'
        sec_file = ''.join(path_txt)
        write_secondary(SUB_DIR+sec_file,DOTS,PROTEIN,sec_seq,prim_seq)
        
        # write out the flags file
        path_txt[2] = 'flags'
        path_txt[5] = ''
        path_txt[6] = ''
        flag_data[0] = f'-fasta {fasta_file}\n'
        flag_data[1] = f'-secstruct_file {sec_file}\n'
        flag_data[2] = f'-constraints:cst_file {const_file}\n'
        flag_file = ''.join(path_txt)
        write_data(SUB_DIR+flag_file,flag_data)

        # assign a test or training label to the sequence
        label = "test" if random.random() > TRAIN_PERCENT else "train"
        # run the docking command for this sequence
        rna_denovo_cmd = DOCK_CMD + "\"" + flag_file + "\""
        a_process = run_cmd(rna_denovo_cmd, SPEED)  
        status = a_process.poll()
        # index, primary sequence, secondary sequence, process, scores, stderr, stdout, 
        # return code, finished running flag, training label
        running_list.append([index, prim_seq, sec_seq, a_process, "", None, "", status, False, label])

        
        # increase index count for correct index in larger list
        index += 1

    # now read all processes' stdin and stderr to keep from zombification and collect data
    finished = 0
    while True:
        for cp in running_list:
            # if process is finished running, skip this child process
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
                update_progress()
                finished += 1
        
        if finished == iterations: break
    
    # remove the temporary directory because this round of iterations are done
    rm(new_dir)
    
    if ROSETTA_DEBUG_DATA:
        # save the entire running list for debug purposes
        # ----TODO THIS IS IF I WANT TO MAKE A DEEP COPY (WITHOUT REFERENCES)----
        '''for j in running_list:
            data_debug.append([])
            data_debug[len(data_debug)-1].extend(j)'''
        # ----TODO THIS IS IF I WANT TO MAKE A SHALLOW COPY JUST BEFORE WRITING ----
        data_debug.extend(running_list)
        debug_df = pd.DataFrame(data_debug, columns=['Index','Sequences','Secondary Structure','Process','Scores','StdErr','StdOut','Return Code','Finish Flag','Training Label'])
        if i == 0:
            tqdm.write("---- Writing troubleshoot csv ----")
            debug_df.to_csv(LOG_DIR+TROUBLESHOOT_CSV, index = False)
            tqdm.write("Complete")
        else:
            tqdm.write("---- Appending to troubleshoot csv (loop = {i}) ----")
            debug_df.to_csv(LOG_DIR+TROUBLESHOOT_CSV, mode = 'a', index = False, header = False)
            tqdm.write("Complete")

    # remove columns in running list that should not be written to the dataset
    for row in running_list:
        del row[8]
        del row[7]
        del row[6]
        del row[5]
        del row[3]
    
    # add data in order into a list
    data.extend(running_list)

    # write/append output data to csv
    df = pd.DataFrame(running_list, columns=['Index','Sequences','Secondary structure','Scores','Split'])
    if i == 0:
        tqdm.write("---- Writing output csv ----")
        df.to_csv(SAVE_DIR+DATA_CSV_NAME, index = False)
        tqdm.write("Complete")
    else:
        tqdm.write(f"---- Appending to output csv (loop = {i}) ----")
        df.to_csv(SAVE_DIR+DATA_CSV_NAME, mode = 'a', index = False, header = False)
        tqdm.write("Complete")
    
    # update and calculate remaining time
    progress.refresh()
    tot_dp = progress.total
    completed = progress.n
    elapsed_time = progress.format_dict['elapsed']
    total_est_time = tot_dp/completed*elapsed_time
    remaining_est_time = total_est_time - elapsed_time
    eta = progress.format_interval(remaining_est_time)
    # do not display calculation if last iteration
    if i != LOOPS:
        tqdm.write(f"\nElapsed time: {progress.format_interval(elapsed_time)} \nCompleted datapoints: {completed} \nTotal datapoints: {tot_dp} \nRemaining datapoints: {tot_dp-completed} \nETA: {eta}")
    

df = pd.DataFrame(data, columns=['Index','Sequences','Secondary structure','Scores','Split'])
tqdm.write("")
# df.to_csv(SAVE_DIR+DATA_CSV_NAME, index = False)
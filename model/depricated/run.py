import pandas as pd
from degen_model.data import *
from degen_model.samples import sample
from degen_model.trainer import Trainer
from degen_model.vae_model import VAE
from degen_model.ga_operations import GA_Ops
import random
from torch import manual_seed
from tensorboardX import SummaryWriter

# 30 epochs seems to be more than enough to train the VAE and reduce the loss function.
# Any more epochs just results in variations of the 3 loss components and doesn't reduce much more.
n_epoch = 30
n_batch = 32
combination_amount = 3
score_threshold = 5000 # determined emperically (3,500 = 25.25% of my training data)
n_returned_samples = 800
max_seq_len = 40
tourn_select_k = 2
elite_percent = 0.01
crossover_rate = 0.9
mutation_rate = 0.01
seed = 200
random.seed(seed)
manual_seed(seed)

log_directory ='log/'
tensorboard_directory = 'tensorboard/'
vae_log_directory = 'vae_generations/'
score_log_directory = 'dock_score_runs/'
output_name = 'VAE_test'
current_gen = 0
run = 0

vae_writer = SummaryWriter(log_directory + output_name + '/' + tensorboard_directory + vae_log_directory + str(current_gen))
score_writer = SummaryWriter(log_directory + output_name + '/' + tensorboard_directory + score_log_directory + str(run))

print("\n-----building data-----")
data = pd.read_csv("data/start/Output_DataSize_2000_Min_20_Max_40_GCAmnt_50_CPUPercent_75_Uncon_False_Seed_200_Covid.csv",  index_col=0)
d = Data(combinations = combination_amount)
d.update_training_data(data)
my_trainer = Trainer(data_module = d, num_epoch = n_epoch, batch_amnt = n_batch, score_threshold_to_binary = score_threshold)
print(d.vocabulary)
print("\n-----creating model - providing vocab and vector from Data class-----")
print(d.device)
model = VAE(d).to(d.device)
print("\n-----fitting the model with the training data-----")
my_trainer.fit(model, d.training_data, summary_writer= vae_writer)
print("\n-----setting model to evaluation mode-----")
model.eval()
print("\n-----sorting data-----")

# copy data
copy_data = [i for i in d.training_data]
# Sort data based on score (smallest to largest)
sorted_data = sorted(copy_data, key=lambda x: x[1])
# perform tournament selection
n_samples = min(len(sorted_data), n_returned_samples)
elite = int(n_samples * elite_percent)
elite = max(elite, 1)
selected_data = GA_Ops.tournament_selection(sorted_data, n_samples, elite, tourn_select_k)
# Sort selected data based on score (smallest to largest)
sorted_selected_data = sorted(selected_data, key = lambda x: x[1])

best_selected_score = sorted_selected_data[0][1]
worst_selected_score = sorted_selected_data[-1][1]
average_selected_score = sum(score for _, score in sorted_selected_data)/len(sorted_selected_data)
best_5_score_avg = sum(score for _, score in sorted_selected_data[:5])/5
worst_5_score_avg = sum(score for _, score in sorted_selected_data[-5:])/5

score_writer.add_scalar('Scores/Best_Score', best_selected_score, current_gen)
score_writer.add_scalar('Scores/Worst_Score', worst_selected_score, current_gen)
score_writer.add_scalar('Scores/Average_Score', average_selected_score, current_gen)
score_writer.add_scalar('Scores/Top_5_Avg', best_5_score_avg, current_gen)
score_writer.add_scalar('Scores/Worst_5_Avg', worst_5_score_avg, current_gen)

# Calling "flush" to make sure all pending events have been written to disk
vae_writer.flush()
score_writer.flush()
# don't need vae writer anymore
vae_writer.close()

print("\n-----grabbing training samples from the model-----")
samples = sample.take_data_samples(model, my_trainer, selected_data, n_batch, n_returned_samples, max_seq_len, crossover_rate, mutation_rate)
print("\n-----printing the sample grabbed-----")
print(samples[:5])
print(samples[-5:])

# %%
import os

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.distributions as tdist
import torch.nn.functional as F
import torch.nn as nn
import torchmetrics as tm

import pandas as pd
import sys
import random
import time
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler
# %%
# Hyperparametres
# ===============================================================================
# Kolik casu dozadu sledujeme k provedeni predikce.
num_time_steps = 1
num_features_intern0 = 512  # 640#640#512     # Pocty neuronu v dalsich vrstvach
num_features_intern1 = 256  # 256#256#256     # 160#256#160  
num_vars_in_LSTM = 128                        # Kolik vystupnich promennych maji LSTM bloky.
latent_dim = 100                              #
num_epochs = 100  # 256                       # Kolik epoch se ma pouzit pro uceni.
my_batch_size = 128                           #
anomaly_outer_margin = 16                     # Na kazdou casu od anomalie se tohle povazuje za nejistou oblast.
anomaly_inner_margin = 16                     #
my_learning_rate, my_amsgrad = 3e-4, False    # 0.0005 3e-4
load_trained_model = False                    # Loads up loss, weights and epochs
continue_training = True                      # Continues to train loaded trained model false if we want to sample from a loaded model
train_diff_dataset = False                    # If we want to train on different dataset with loaded model then it is good to save the newest best model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

scaler = MinMaxScaler()
dataScale = 1.0
# %%
# Jmeno souboru na testovani
test_file_name = "data/michal_anomal_91_dist.txt"
# Jmeno souboru s anotaci testu.
anot_test_file_name = "data/michal_annot.txt"
# Jmeno souboru na trenovani
train_file_name = "data/michal_normal_91_dist.txt"
# Jmeno souboru s ulozenym souborem
load_model_file_name = ""
save_model_file_prefix = "best_model" if not train_diff_dataset else "best_diff_model_save"
# %%
class RepeatVector(nn.Module):
    def __init__(self, times_repeated):
        super().__init__()
        self.times_repeated = times_repeated

    def forward(self, x):
        # https://stackoverflow.com/questions/57896357/how-to-repeat-tensor-in-a-specific-new-dimension-in-pytorch
        x = x.unsqueeze(1).repeat(1, self.times_repeated, 1)
        return x

# source: https://stackoverflow.com/questions/44130851/simple-lstm-in-pytorch-with-sequential-module/64265525#64265525
class ExtractTensor(nn.Module):
    def __init__(self, squeezed=False):
        super().__init__()
        self.squeezed = squeezed

    def forward(self, x):
        # Output shape (batch, features, hidden)
        tensor, _ = x
        # Reshape shape (batch, hidden)
        if self.squeezed:
            return tensor[:, -1, :]
        if not self.squeezed:
            return tensor[:, :, :]

# source: https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/3
# should be tensor flow equivalent in pytorch
class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False, relu_activation=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first
        self.relu = relu_activation

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        # (samples * timesteps, input_size)
        x_reshape = x.contiguous().view(-1, x.size(-1))

        y = self.module(x_reshape)
        if self.relu:
            y = F.relu(y)

        # We have to reshape Y
        if self.batch_first:
            # (samples, timesteps, output_size)
            y = y.contiguous().view(x.size(0), -1, y.size(-1))
        else:
            # (timesteps, samples, output_size)
            y = y.view(-1, x.size(1), y.size(-1))

        return y
    
# %%
class MyNet(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

        self.inference_net = nn.Sequential(
            TimeDistributed(nn.Linear(num_features, num_features_intern0)),# relu_activation=True),
            nn.ReLU(),
            TimeDistributed(nn.Linear(num_features_intern0, num_features_intern1)),
            nn.LSTM(num_features_intern1, num_vars_in_LSTM, bidirectional=True),
            ExtractTensor(squeezed=True),
            nn.ReLU(),
            nn.Linear(num_vars_in_LSTM * 2, self.latent_dim),
        )

        self.generative_net = nn.Sequential(
            RepeatVector(num_time_steps),
            nn.LSTM(self.latent_dim, num_vars_in_LSTM, bidirectional=True),
            ExtractTensor(),
            nn.ReLU(),
            TimeDistributed(nn.Linear(num_vars_in_LSTM * 2, num_features_intern0)),# relu_activation=True),
            nn.ReLU(),
            TimeDistributed(nn.Linear(num_features_intern0, num_features)),
        )

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def encode(self, x):
        return self.inference_net(x)

    def decode(self, z):
        return self.generative_net(z)

def prep_directories():
    """Prepares directories to store output data"""
    if (not os.path.exists("outputs")):
        os.mkdir("outputs")
    base_folder = os.path.join('outputs', f'{iso_date}_{case_id}')
    os.mkdir(base_folder)
    output_data_path = os.path.join(base_folder, 'output_data')
    os.mkdir(output_data_path)
    plots_path = os.path.join(base_folder, 'plots')
    os.mkdir(plots_path)
    return base_folder, output_data_path, plots_path

def create_case_id():
    """Generates identifying id for the current setup of the network"""
    case_id = base_case_id
    case_id += "_t" + str(num_time_steps) + "_d" + str(num_features_intern0) + "_" + str(num_features_intern1)
    case_id += "_" + str(num_vars_in_LSTM) + "_" + str(latent_dim) + "_s" + str(int(1000*dataScale))

    if my_amsgrad:
        case_id + "_a1"
    case_id += "_e" + str(num_epochs)

    print("case_id:", case_id)
    return case_id

def load_dataset(path, scaler, mode):
    """Loads up dataset returns dataframes values, and shapes"""
    df = pd.read_csv(path, header=None, delimiter=r"\s+")
    x = df.values.astype('float32')

    if mode == 'train':
        x_scaled = scaler.fit_transform(x)
    elif mode == 'test':
        x_scaled = scaler.transform(x)
    else:
        raise Exception("Invalid mode")

    df = pd.DataFrame(x_scaled)
    return df.values, df.shape[0], df.shape[1]

def num_of_batches_in_set(dataset):
    # Kolik je davek, ze kterych mohou byt vzorky.
    num_batches_in_set = (dataset.shape[0]-num_time_steps+1)//my_batch_size
    if (dataset.shape[0]-num_time_steps+1) % my_batch_size > 0: num_batches_in_set += 1
    print("num_batches_in_training_set=", num_batches_in_set)
    return num_batches_in_set

def build_model(latent_dim):
    """Setup of model and optimizer if load model is True loads up saved model from path"""
    model = MyNet(latent_dim=latent_dim)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=my_learning_rate, betas=(0.9, 0.999), eps=1e-7, amsgrad=my_amsgrad)
    # optimizer = torch.optim.SGD(model.parameters(), lr=my_learning_rate, momentum=0.9, nesterov=True)
    num_epochs_loaded = 0
    best_loss = 1.0e10

    if load_trained_model:
        num_epochs_loaded, loaded_loss = load_model(model, optimizer, load_model_file_name)
        if not train_diff_dataset:
            best_loss = loaded_loss

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return model, optimizer, num_epochs_loaded, best_loss

# Saves model for training or evaluating
def save_model(epoch, model, optimizer, loss, path):
    """Saves model"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        # maybe add lossi and training loss so that we can view graphs even after prolonged training
    }, path)

def load_model(model, optimizer, path):
    """Loads up model, optimizer states from given path and returns epoch when the model was save and it's loss"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    optimizer.param_groups[0]['lr'] = my_learning_rate
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f'Loss after last training {loss} and ended on epoch {epoch}')
    return epoch, loss

def checkpoint(epoch, loss_after_epoch, best_loss):
    """Checks if the current state of model is better than the best saved one if yes saves it"""
    weight_saved = 0
    if loss_after_epoch < best_loss:
        curent_best_model_path = f'{root}/{save_model_file_prefix}_{best_loss:.6f}_{iso_date}.pt'
        if os.path.isfile(curent_best_model_path):
            os.remove(curent_best_model_path)
        best_loss = loss_after_epoch
        save_model(epoch, model, optimizer, best_loss,
                   (f'{root}/{save_model_file_prefix}_{best_loss:.6f}_{iso_date}.pt'))
        weight_saved = 1
    return weight_saved, best_loss

# Show graph of model or atleast one part of it, does depend on forward function implemented in module MyNet
def tensorboard_model(model):
    """Graphs out tensorboard model"""
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter("torchlogs/")
    # rand_idx = random.randint(0, dataset.shape[0]//my_batch_size-1) not used
    batch_of_samples = get_random_batch_of_samples(
        dataset, my_batch_size, num_time_steps, num_features)
    batch_of_samples = torch.from_numpy(batch_of_samples).to(device)
    writer.add_graph(model, batch_of_samples)
    writer.close()

def get_ordered_batch_of_samples(sequences, batch_first_idx, batch_size, num_time_steps, num_features):
    batch_of_samples = list()
    # print( "getbatch_of_samples: batch_first_idx=", batch_first_idx, "  batchLastIdx=", batch_first_idx+batch_size-1 )
    for first_idx in range(batch_first_idx, batch_first_idx + batch_size):
        lastIdx = first_idx + num_time_steps
        if lastIdx >= len(sequences):
            # break # Pro tenhle index uz neni dost vzorku.
            return np.array(batch_of_samples)
        oneSample = sequences[first_idx:lastIdx, 0:num_features]
        batch_of_samples.append(oneSample)
    return np.array(batch_of_samples)

def get_random_batch_of_samples(sequences, batch_size, num_time_steps, num_features):
    batch_of_samples = list()
    for first_idx in range(batch_size):
        first_idx = random.randint(0, sequences.shape[0] - num_time_steps - 1)
        lastIdx = first_idx + num_time_steps
        oneSample = sequences[first_idx:lastIdx, 0:num_features]
        batch_of_samples.append(oneSample)
    return np.array(batch_of_samples)

def eval_batch_of_predictions(predicted_vals, true_vals):
    errs = list()
    for i in range(len(true_vals)):
        err = F.mse_loss(predicted_vals[i], true_vals[i])
        errs.append(err.item())
    return np.array(errs)

def eval_abnormal_vals(errs, anomaly_map):
    n = 0
    sum_abnormal_vals = 0.0
    for i in range(len(errs)):
        idx_in_anomaly_map = i+num_time_steps-1
        if anomaly_map[idx_in_anomaly_map] == 2:
            sum_abnormal_vals += errs[i]
            n += 1
    if n == 0:
        print("Problem: No anomalies were catched. Check the annotation file and the value of anomaly_inner_margin.")
        return 0.0
    return sum_abnormal_vals/n

def eval_normal_vals(errs, anomaly_map):
    n = 0
    sum_normal_vals = 0.0
    for i in range(len(errs)):
        idx_in_anomaly_map = i+num_time_steps-1
        if anomaly_map[idx_in_anomaly_map] == 1:
            sum_normal_vals += errs[i]
            n += 1
    return sum_normal_vals/n

def eval_set_err(model, num_batches, dataset, num_features):
    """Evaluates errors in chosen dataset"""
    for i in range(num_batches):
        true_values = get_ordered_batch_of_samples(
            dataset, i*my_batch_size, my_batch_size, num_time_steps, num_features)
        true_values = torch.from_numpy(true_values).to(device)
        # print( "i=", i, " ", true_values.shape )
        batch_errs = eval_batch_of_predictions(model(true_values), true_values)
        if i == 0:
            errs = batch_errs
        else:
            errs = np.concatenate((errs, batch_errs), axis=0)
    return errs

def calculate_prediction_times(total_length, num_time_steps):
    prediction_times = np.array(range(total_length - num_time_steps))
    prediction_times = prediction_times + num_time_steps - 1
    prediction_times = prediction_times.reshape((len(prediction_times), 1))
    return prediction_times

def compute_loss(model, x):
    """Computes MSE loss"""
    y = model(x)
    val = F.mse_loss(x, y)  # To udela MSE jenom za jednotlive casy
    return val

def train_loop_fn():
    """Trains 1 epoch on training data"""
    train_loss = tm.MeanMetric().to(device)
    for i in range(0, num_batches_in_training_set):
        # rand_idx = random.randint(0, dataset.shape[0]//my_batch_size-1) Not used actually ask what is this
        batch_of_samples = get_random_batch_of_samples(
            dataset, my_batch_size, num_time_steps, num_features)
        batch_of_samples = torch.from_numpy(batch_of_samples).to(device)
        loss = (compute_loss(model, batch_of_samples))
        with torch.no_grad():
            train_loss(loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return train_loss.compute()

def validation_loop():
    """One epoch of validation"""
    loss = tm.MeanMetric().to(device)
    for i in range(0, num_batches_in_training_set):
        batch_of_samples = get_ordered_batch_of_samples(
            dataset, i*my_batch_size, my_batch_size, num_time_steps, num_features)
        batch_of_samples = torch.from_numpy(batch_of_samples).to(device)
        loss(compute_loss(model, batch_of_samples))
    return loss.compute()

def fit_model(best_loss):
    """Train the model for num epochs set in hyperparams, plots out the learning curve"""
    model.train()
    lossi = range(1, num_epochs + 1)
    losses_t = []
    losses_v = []
    for epoch in range(1 + num_epochs_loaded, num_epochs + 1 + num_epochs_loaded):
        start_time = time.time()
        if torch.cuda.device_count() > 1:
            torch.cuda.synchronize(device)
        # training loop
        train_loss = train_loop_fn()
        losses_t.append(train_loss.item())
        # validation loop
        with torch.no_grad():
            loss_after_epoch = validation_loop()
            losses_v.append(loss_after_epoch.item())
            weight_saved, best_loss = checkpoint(epoch, loss_after_epoch, best_loss)

            end_time = time.time()

        print(f'Epoch: {epoch:3d}, loss: {loss_after_epoch:.6f}, computational time {(end_time - start_time):.2f} s, weight_saved={weight_saved}')

    plot_train_val_loss(lossi, losses_t, losses_v, case_id, iso_date)
    return best_loss

def plot_train_val_loss(lossi, losses_t, losses_v, case_id, iso_date):
    """Plots out training and validation loss on y axis and epochs on x axis"""
    plt.title("Training and Validation loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(lossi, losses_t, label='Training loss')
    plt.plot(lossi, losses_v, label='Validation loss')
    last_loss = f'{losses_v[-1]:.6f}'
    plt.annotate(last_loss, (lossi[-1], losses_t[-1]),
                 textcoords='offset points', xytext=(0, 10), ha='center')
    plt.legend(loc='best')
    plt.savefig(f'{output_plots_path}/lossi_loss_{case_id}_{iso_date}.png')

def read_anot_file():
    anomaly_locations = []
    with open(anot_test_file_name, 'r') as file:
        for row in file:
            row = row.strip()
            if len(row) > 0:
                res = row.split()
                if res[0] == '#': continue
                anomaly_locations.append([int(res[0]), int(res[1])])
    return anomaly_locations

def create_anomaly_map(num_rows_test):
    # Precteme anotacni soubor.
    anomaly_locations = read_anot_file()

    # Vytvorime mapu: 1-normalni, 2-anomalie, 0-nejsite misto.
    anomaly_map = [1] * num_rows_test

    for anomalyLocation in anomaly_locations:
        # Posuvy o (num_time_steps//2) berou v uvahu, co lze jeste rozumne ocekavat. Primo v deklarovanem zacatku tu
        # anomalii videt nelze, protoze vysledek je zpozdeny o num_time_steps. V deklarovanem zacatku + (num_time_steps//2)
        # vidi sit uz pulku (num_time_steps//2) okamziku, kde je aomalie. To by mohlo stacit. Podobne je to posunute dozadu.
        l = anomalyLocation[0] + (num_time_steps//2) - anomaly_outer_margin
        r = anomalyLocation[1] + (num_time_steps//2) + anomaly_outer_margin
        if l < 0: l = 0
        if r > num_rows_test: r = num_rows_test
        for i in range(l, r): anomaly_map[i] = 0
        l = anomalyLocation[0] + (num_time_steps//2) + anomaly_inner_margin
        r = anomalyLocation[1] + (num_time_steps//2) - anomaly_inner_margin
        if l < 0: l = 0
        if l > num_rows_test: l = num_rows_test
        if r < 0: r = 0
        if r > num_rows_test: r = num_rows_test
        for i in range(l, r): anomaly_map[i] = 2
    return anomaly_map

def calculate_threshold(eval_errs):
    maxA = max(eval_errs)
    minA = min(eval_errs)
    myTh = (maxA + minA)/2.0
    return myTh

def is_in_range(n):
    is_in = 0
    anot_file_lines = []
    with open(anot_test_file_name) as anotFile:
        anot_file_lines = [line.rstrip().split(" ") for line in anotFile]

    for i in range(len(anot_file_lines)):
        a = int(anot_file_lines[i][0])
        b = int(anot_file_lines[i][1])
        if n in range(a, b):
            is_in = 1

    return is_in

# %%
iso_date = datetime.now().isoformat('T')
base_case_id = "rad_bas_bidir" # nejaky retezec, ktery se pouzije jako zacatek vsech jmen souboru (a dalsi se doplni automaticky).
case_id = create_case_id()
root, output_data_path, output_plots_path = prep_directories()
dataset, total_length_train, num_features = load_dataset(train_file_name, scaler, 'train')
num_batches_in_training_set = num_of_batches_in_set(dataset) # Kolik je davek, ze kterych mohou byt vzorky.
print("training dataset shape: ", dataset.shape)

model, optimizer, num_epochs_loaded, best_loss = build_model(latent_dim)
# %% Training of the model
if continue_training or not load_trained_model:
    best_loss = fit_model(best_loss)

# %% ===============================================================================
# Vyzkousime predikci na trenovaci sekvenci.
model.eval()

errs = eval_set_err(model, num_batches_in_training_set, dataset, num_features)
file_name = f"{case_id}_{iso_date}_training_errs.txt"
# np.savetxt(f'output_data/training_errs/{file_name}', errs, fmt="%.8f")
np.savetxt(f'{output_data_path}/{file_name}', errs, fmt="%.8f")
mean_training_err = np.mean(errs)
print("> mean_training_err=", mean_training_err)

# Ke kresleni obrazku pripravime casy, kde mame predikce (to nejsou vsechny).
prediction_times = calculate_prediction_times(total_length_train, num_time_steps)
# Vykreslime vysledek pro trenovaci sekvenci.
fig, ax = plt.subplots(1, 1, figsize=(32, 10))
ax.plot(prediction_times, errs, linewidth=1, color='g')

# %% Precteme data pro testovani a predikce.
dataset, num_rows_test, num_features = load_dataset(test_file_name, scaler, 'test')
num_batches_in_test_set = num_of_batches_in_set(dataset)

errs = eval_set_err(model, num_batches_in_test_set, dataset, num_features)
file_name = f"{case_id}_{iso_date}_reconst_errs.txt"
# np.savetxt(f'output_data/reconstructed_errs/{file_name}', errs, fmt="%.8f")
np.savetxt(f'{output_data_path}/{file_name}', errs, fmt="%.8f")
mean_reconst_err = np.mean(errs)
print("> mean_reconst_err=", mean_reconst_err)

# %% Obrazek chyby predikce.
prediction_times = calculate_prediction_times(num_rows_test, num_time_steps)
ax.plot(prediction_times, errs, linewidth=1, color='r')

anomaly_map = create_anomaly_map(num_rows_test)

abnormal_eval = eval_abnormal_vals(errs, anomaly_map)
normal_eval = eval_normal_vals(errs, anomaly_map)
print('abnormal_eval={:.6f}, normal_eval={:.6f}, SNR={:.3f}'.format(abnormal_eval, normal_eval, abnormal_eval/normal_eval))

for i in range(num_rows_test):
    if anomaly_map[i] == 1:
        ax.plot(i, 0.0,  marker='.',  markersize=1, color='y')
    if anomaly_map[i] == 2:
        ax.plot(i, 0.0,  marker='.',  markersize=1, color='b')

plt.suptitle("training (green) and reconst (red) errors")
file_name = f'{case_id}_{iso_date}_training_and_reconst_errs.png'
# plt.savefig(f'plots/reconstructed_errs/{file_name}')
plt.savefig(f'{output_plots_path}/{file_name}')
print("The image with errs has been saved to:", file_name)
# %%
eval_errs = [i * 10.0 for i in errs]

myTh = calculate_threshold(eval_errs)

true_positives = 0
true_negatives = 0
false_positives = 0
false_negatives = 0
for i in range(len(eval_errs)):
    err = eval_errs[i]
    is_anom = 0
    is_gt = 0

    if (err > myTh):
        is_anom = 1
    # pro kazdy snimek urcim jestli se shoduje odpoved ze site s anotaci nebo ne
    is_gt = is_in_range(i)
    if ((is_gt == 1) and (is_anom == 1)):
        true_positives = true_positives + 1
    elif ((is_gt == 0) and (is_anom == 0)):
        true_negatives = true_negatives + 1
    elif ((is_gt == 1) and (is_anom == 0)):
        false_negatives = false_negatives + 1
    elif ((is_gt == 0) and (is_anom == 1)):
        false_positives = false_positives + 1

acc = ((true_positives + true_negatives) / len(eval_errs)) * 100
print("True positives", true_positives)
print("True negatives", true_negatives)
print("False negatives", false_negatives)
print("False positives", false_positives)
print(f"Accuracy {acc:.2f}")

with open(f'{output_data_path}/log_{case_id}_{iso_date}.txt', 'w+') as log:
    log.write(f'Mean training err={mean_training_err:.5f}, Mean reconst error={mean_reconst_err:.5f}\
                \nAbnormal eval={abnormal_eval:.5f}, Normal eval={normal_eval:.5f}, SNR={(abnormal_eval/normal_eval):.4f}\
                \nTrue postives {true_positives}\nTrue negatives {true_negatives}\nFalse negatives {false_negatives}\nFalse positives {false_positives}\nAccuracy {acc:.4f}\nBest loss {best_loss}')

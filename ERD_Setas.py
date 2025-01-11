import os
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import scipy.io as spio
import funcoes_processamento as fp
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
%matplotlib qt5

# Carregar dados de movimento real
seta_grande = []
seta_pequena = []
voluntarios = [i for i in range(11)]  # Grupo controle tem 13 voluntarios
seta = 'Direita'

if seta == 'Direita':
    lado = '2'
    epoca_grande = 'right/big'
    epoca_pequena = 'right/small'
else:
    lado ='1'
    epoca_grande ='left/big'
    epoca_pequena ='left/small'

# Carregar dados da seta grande
for i in voluntarios:
    if i < 9:
        data_folder = 'F:\Juan\LFCOG\ProcessamentodeSinais\Padrao/S0'+str(i+1)
        voluntario = '/S0'+str(i+1)+'_'+lado+'0-epo'
    else:
        data_folder = 'F:\Juan\LFCOG\ProcessamentodeSinais\Padrao/S'+ str(i+1)
        voluntario = '/S' + str(i+1) + '_'+lado+'0-epo'
    epoca = mne.read_epochs(data_folder + voluntario + '.fif')
    seta_grande.append(epoca[epoca_grande].get_data(picks='eeg'))

# Carregar dados da seta pequena
for i in voluntarios:
    if i < 9:
        data_folder = 'F:\Juan\LFCOG\ProcessamentodeSinais\Padrao/S0'+str(i+1)
        voluntario = '/S0'+str(i+1)+ '_'+lado+'1-epo'
    else:
        data_folder = 'F:\Juan\LFCOG\ProcessamentodeSinais\Padrao/S'+ str(i+1)
        voluntario = '/S' + str(i+1) + '_'+lado+'1-epo'
    epoca = mne.read_epochs(data_folder + voluntario + '.fif')
    seta_pequena.append(epoca[epoca_pequena].get_data(picks='eeg'))

ch_name = epoca.ch_names
fs = epoca.info['sfreq']

n_ch = np.size(seta_grande[0], 1)
n = np.size(seta_grande[0], 2)

freq1 = 8
freq2 = 13

t1 = -2
t2 = 8

# Extrair os dados dos canais específicos
canais_interesse = [2, 1, 18, 22, 0, 23, 19, 5, 3, 4, 20, 25, 28, 29, 21, 26, 6, 7]  # Índices dos canais especificados

dados_grande_voluntarios = []
dados_pequena_voluntarios = []

# Calcula as médias dos dados para cada voluntário
for i in range(len(seta_grande)):
    dados_grande_voluntarios.append(np.mean(seta_grande[i][:, canais_interesse, :], axis=0))
    dados_pequena_voluntarios.append(np.mean(seta_pequena[i][:, canais_interesse, :], axis=0))

# Calcula a média das médias dos dados para obter um único conjunto de dados para cada condição
dados_grande = np.mean(dados_grande_voluntarios, axis=0)
dados_pequena = np.mean(dados_pequena_voluntarios, axis=0)

# Canais de interesse agrupados
canais = [0, 2, 1,  # F3, Fz, F4
        18, 17, 22, 23, # Fc3, Fc1, Fc2, Fc4
    20, 3, 19, 5, 24, 4, 25, # C5, C3, C1, Cz, C2, C4, C6
        21, 28, 27, 29, 26, # Cp5, Cp1, Cpz, Cp2, Cp6
           6, 8, 7, # P3, Pz, P4
              16]  # Oz

grupos_canais = [
    canais[:3],      # Grupo 1: F3, Fz, F4
    canais[3:5],     # Grupo 2: Fc1, Fc2
    canais[5:7],     # Grupo 3: Fc3, Fc4
    canais[7:10],    # Grupo 4: C1, Cz, C2
    canais[10:12],   # Grupo 5: C3, C4
    canais[12:14],   # Grupo 6: C5, C6
    canais[14:17],   # Grupo 7: Cp1, Cpz, Cp2
    canais[17:19],   # Grupo 8: Cp5, Cp6
    canais[19:22],   # Grupo 9: P3, Pz, P4
    canais[22:]      # Grupo 10: Oz
]

# Definir intervalo de tempo desejado (de -2 a 8 segundos)
tempo_inicial = int(fs * t1)
tempo_final = int(fs * t2)
tempo = np.arange(t1, t2, 1/fs)[:tempo_final-tempo_inicial]

# Plotar os dados para cada grupo de canais
for i, grupo in enumerate(grupos_canais, start=1):
    num_canais = len(grupo)
    fig, axs = plt.subplots(num_canais, 1, figsize=(10, 6*num_canais))

    if num_canais == 1:  # Se houver apenas um canal, axs será um objeto Axes, não um array
        axs = [axs]

    for j, canal_indice in enumerate(grupo):
        canal_nome = ch_name[canal_indice]
        dados_grande_grupo = dados_grande[:, canal_indice][tempo_inicial:tempo_final]
        dados_pequena_grupo = dados_pequena[:, canal_indice][tempo_inicial:tempo_final]
        tempo_grande = np.arange(t1, t2, 1/fs)[:len(dados_grande_grupo)]
        tempo_pequena = np.arange(t1, t2, 1/fs)[:len(dados_pequena_grupo)]

        axs[j].plot(tempo_grande, dados_grande_grupo, label='Grande', color='blue')
        axs[j].plot(tempo_pequena, dados_pequena_grupo, label='Pequena', color='red')

        axs[j].set_title(canal_nome)
        axs[j].set_xlabel('Tempo (s)')
        axs[j].set_ylabel('Amplitude')
        axs[j].legend()

    plt.tight_layout()
    plt.show()
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
voluntarios = [i for i in range(12)]  # Grupo controle tem 13 voluntarios
seta = 'Esquerda'

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
canais = [0, 2, 1, # F3, Fz, F4
        18, 17, 22, 23, # Fc3, Fc1, Fc2, Fc4
    20, 3, 19, 5, 24, 4, 25, # C5, C3, C1, Cz, C2, C4, C6
        21, 28, 27, 29, 26, # Cp5, Cp1, Cpz, Cp2, Cp6
           6, 8, 7, # P3, Pz, P4
              16]  # Oz

pares_canais = [0,   1, 17, 22, 18, 23, # F3 vs F4 // Fc1 vs Fc2 // Fc3 vs Fc4
                19, 24,  3,  4, 20, 25, # C1 vs C2 // C3 vs C4 // C5 vs C6
                28, 29, 21, 26,  6, 7] # Cp1 vs Cp2 // Cp5 vs Cp6 // P3 vs P4

t1 = -2
t2 =  8

MV_erds_grande, MV_erp_grande, MV_erds_vol_grande, t_vol_grande, MV_erp_vol_grande, MV_erds_vol_MA_grande = fp.erdsvol(seta_grande, t1, t2, freq1, freq2, fs)

MV_erds_pequena, MV_erp_pequena, MV_erds_vol_pequena, t_vol_pequena, MV_erp_vol_pequena, MV_erds_vol_MA_pequena = fp.erdsvol(seta_pequena, t1, t2, freq1, freq2, fs)

y1 = 50
c = 'lightgray'
ch_box = np.array(canais)
ch_box2 = np.array(pares_canais)
voluntarios = np.array(voluntarios)
titulo_grande = 'Seta ' + seta + ' (Grande)'
titulo_pequeno = 'Seta ' + seta + ' (Pequena)'

# Mapa estatistico para a mesma condição
tabela_MV_grande, tmin_grande = fp.mapa_estatistico_mesma_condicao(MV_erds_grande, MV_erds_vol_grande, t_vol_grande, y1, titulo_grande, ch_name, ch_box2, voluntarios, c)
tabela_MV_pequeno, tmin_pequeno = fp.mapa_estatistico_mesma_condicao(MV_erds_pequena, MV_erds_vol_pequena, t_vol_pequena, y1, titulo_pequeno, ch_name, ch_box2, voluntarios, c)

if seta == 'Direita':
    pares_canais = [16, 21, 3, 17]
    titulo_seta = 'Seta para a Direita'
if seta == 'Esquerda':
    pares_canais = [16, 26, 4, 22]
    titulo_seta = 'Seta para a Esquerda'
tabela_MVIM, tmin = fp.mapa_estatistico_MVIMG(MV_erds_grande, MV_erds_vol_grande, MV_erds_pequena, MV_erds_vol_pequena, t_vol_grande,  y1, titulo_seta, ch_name, pares_canais, c)
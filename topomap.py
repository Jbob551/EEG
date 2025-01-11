import mne
import numpy as np
import scipy.stats
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import pipeline_functions as pf
%matplotlib qt5
from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter1d


# Loading raw dataset  
path = r'F:\Juan\LFCOG\ProcessamentodeSinais\Padrao\S01/S01.edf'
dados_raw, info = pf.abre_preprocessa_reamostra(path)

layout = mne.channels.find_layout(dados_raw.info)

condicoes = {'left': {'small': {'data_folder': r'F:\Juan\LFCOG\ProcessamentodeSinais\Padrao', 'voluntarios': [i+1 for i in range(12)], 'epoca': '11-epo'},
                     'big': {'data_folder': r'F:\Juan\LFCOG\ProcessamentodeSinais\Padrao', 'voluntarios': [i+1 for i in range(12)], 'epoca': '10-epo'}},
            
             'right': {'small': {'data_folder': r'F:\Juan\LFCOG\ProcessamentodeSinais\Padrao', 'voluntarios': [i+1 for i in range(12)], 'epoca': '21-epo'},
                     'big': {'data_folder': r'F:\Juan\LFCOG\ProcessamentodeSinais\Padrao', 'voluntarios': [i+1 for i in range(12)], 'epoca': '20-epo'}}}

condicoes_epocas = {}

def gerar_erp(condicoes):
    dados_erp = {}
    dados_erp_mean = {}
    
    for movimento, condicoes_right in condicoes.items():
        for condicao, info in condicoes_right.items():
            dados_epocas = []
            for i in info['voluntarios']:
                data_folder = info['data_folder'] + (f'\S0{i}' if i <= 9 else f'\S{i}')
                voluntario = (f'\S0{i}' if i <= 9 else f'\S{i}') + '_' + info['epoca']

                print(data_folder + voluntario)

                epocas = mne.read_epochs(data_folder + voluntario + '.fif')
                dados_epocas.append(epocas.get_data(picks='eeg'))

            condicoes_epocas[condicao] = epocas
            voluntarios = len(dados_epocas)
            dados_erp[condicao] = np.zeros((voluntarios, 30, 6001))

            for indice in range(voluntarios):
                dados_erp[condicao][indice,:,:] = np.mean(dados_epocas[indice], axis=0)

            dados_erp_mean[condicao] = np.mean(dados_erp[condicao], axis=0)

        dados_erp[f'{movimento}_small'], dados_erp_mean[f'{movimento}_small'] = dados_erp['small'], dados_erp_mean['small']
        dados_erp[f'{movimento}_big'], dados_erp_mean[f'{movimento}_big'] = dados_erp['big'], dados_erp_mean['big']
       
    return (dados_erp['left_small'], dados_erp_mean['left_small'], 
            dados_erp['left_big'], dados_erp_mean['left_big'], 
            dados_erp['right_small'], dados_erp_mean['right_small'], 
            dados_erp['right_big'], dados_erp_mean['right_big'])

dados_erp = {}
dados_erp_mean = {}

(dados_erp['left_small'], dados_erp_mean['left_small'], 
 dados_erp['left_big'], dados_erp_mean['left_big'], 
 dados_erp['right_small'], dados_erp_mean['right_small'], 
 dados_erp['right_big'], dados_erp_mean['right_big']) = gerar_erp(condicoes)

# epocas_controle_exe = condicoes_epocas['controle_exe']
# epocas_controle_wait = condicoes_epocas['controle_wait']
# epocas_small = condicoes_epocas['imaginacao_exe']
# epocas_big = condicoes_epocas['imaginacao_img']

# If you want to access a single subject, use these variables
subject = 0
dados_erp_small_subject_left = dados_erp['left_small'][subject,:,:]
dados_erp_big_subject_left = dados_erp['left_big'][subject,:,:]

dados_erp_small_subject_right = dados_erp['right_small'][subject,:,:]
dados_erp_big_subject_right = dados_erp['right_big'][subject,:,:]


condicoes = {}

condicoes['left_small_1'] = dados_erp_mean['left_small'][:,1350:1800]
condicoes['left_small_2'] = dados_erp_mean['left_small'][:,2400:3000]
condicoes['left_big_1'] = dados_erp_mean['left_big'][:,1350:1800]
condicoes['left_big_2'] = dados_erp_mean['left_big'][:,2400:3000]


condicoes['right_small_1'] = dados_erp_mean['right_small'][:,1350:1800]
condicoes['right_small_2'] = dados_erp_mean['right_small'][:,2400:3000]
condicoes['right_big_1'] = dados_erp_mean['right_big'][:,1350:1800]
condicoes['right_big_2'] = dados_erp_mean['right_big'][:,2400:3000]

pontos = {}

for condicao, dados in condicoes.items(): 
    pontos[condicao+'_pos_i1'] = (np.argmax(np.max(dados,axis=0))+1350)//600
    pontos[condicao+'_neg_i1'] = (np.argmin(np.min(dados,axis=0))+1350)//600
    pontos[condicao+'_pos_i2'] = (np.argmax(np.max(dados,axis=0))+2400)//600
    pontos[condicao+'_neg_i2'] = (np.argmin(np.min(dados,axis=0))+2400)//600

# Evoked objects creation (Intervals in samples)
evoked_small_left = mne.EvokedArray(dados_erp_mean['left_small'][:,:3000], info=info, comment='Small', 
tmin=0, baseline=(0,2)) 
evoked_big_left = mne.EvokedArray(dados_erp_mean['left_big'][:,:3000], info=info, comment='Big', 
tmin=0, baseline=(0,2)) 

evoked_small_right = mne.EvokedArray(dados_erp_mean['right_small'][:,:3000], info=info, comment='Small', 
tmin=0, baseline=(0,2)) 
evoked_big_right = mne.EvokedArray(dados_erp_mean['right_big'][:,:3000], info=info, comment='Big', 
tmin=0, baseline=(0,2)) 

evokeds = dict(left_small=evoked_small_left, left_big=evoked_big_left, 
               right_small=evoked_small_right, right_big=evoked_big_right)

# Left
evoked_small_left.plot_topomap(times=[1.5, 2.215, 2.272, 2.374, 4.3, 4.393], 
                                   average=0.05, scalings=None, ch_type='eeg',
                                   sphere=None, image_interp='cubic', time_unit='s', time_format="%0.2f s", vlim=(-2.0,2.0))
evoked_big_left.plot_topomap(times=[1.5, 2.2, 2.263, 2.404, 4.282, 4.362], 
                                   average=0.05, scalings=None, ch_type='eeg',
                                   sphere=None, image_interp='cubic', time_unit='s', time_format="%0.2f s", vlim=(-2.0,2.0))

# Right
evoked_small_right.plot_topomap(times=[1.5, 2.2, 2.3, 2.4, 4.3, 4.4], 
                                   average=0.05, scalings=None, ch_type='eeg',
                                   sphere=None, image_interp='cubic', time_unit='s', time_format="%0.2f s", vlim=(-2.0,2.0))
evoked_big_right.plot_topomap(times=[1.5, 2.2, 2.3, 2.4, 4.3, 4.4], 
                                   average=0.05, scalings=None, ch_type='eeg',
                                   sphere=None, image_interp='cubic', time_unit='s', time_format="%0.2f s", vlim=(-2.0,2.0))

evoked_small_left.plot_joint(times=[1.5, 2.215, 2.272, 2.374, 4.3, 4.393], 
                                    title='ERP: Antebraço Esquerdo (Seta Pequena)')
evoked_big_left.plot_joint(times=[1.5, 2.2, 2.263, 2.404, 4.282, 4.362],
                                    title='ERP: Antebraço Esquerdo (Seta Grande)')

evoked_small_right.plot_joint(times=[1.5, 2.219, 2.268, 2.409, 4.287, 4.388],
                                    title='ERP: Antebraço Direito (Seta Pequena)')
evoked_big_right.plot_joint(times=[1.5, 2.2, 2.3, 2.4, 4.3, 4.371],
                                    title='ERP: Antebraço Direito (Seta Grande)')

mne.viz.plot_compare_evokeds(evokeds, picks='eeg', combine='mean', title='')
mne.viz.plot_evoked_topo(evokeds,layout=layout) # Multi-ERP plot (interactive)

import os
import mne
import numpy as np
import scipy.stats
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import pipeline_functions as pf
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
%matplotlib qt5

# Parâmetros
caminho_arquivo = r'F:\Juan\LFCOG\ProcessamentodeSinais\Padrao\S01/S01.edf'
dados_brutos, info = pf.abre_preprocessa_reamostra(caminho_arquivo)

layout = mne.channels.find_layout(dados_brutos.info)

condicoes = {
    'left': {
        'small': {
            'pasta_dados': r'F:\Juan\LFCOG\ProcessamentodeSinais\Padrao', 
            'voluntarios': [i+1 for i in range(12)], 
            'epoca': '11-epo'
        },
        'big': {
            'pasta_dados': r'F:\Juan\LFCOG\ProcessamentodeSinais\Padrao', 
            'voluntarios': [i+1 for i in range(12)], 
            'epoca': '10-epo'
        }
    },
    'right': {
        'small': {
            'pasta_dados': r'F:\Juan\LFCOG\ProcessamentodeSinais\Padrao', 
            'voluntarios': [i+1 for i in range(12)], 
            'epoca': '21-epo'
        },
        'big': {
            'pasta_dados': r'F:\Juan\LFCOG\ProcessamentodeSinais\Padrao', 
            'voluntarios': [i+1 for i in range(12)], 
            'epoca': '20-epo'
        }
    }
}

condicoes_epocas = {}

def gerar_erp(condicoes):
    dados_erp = {}
    dados_erp_media = {}
    
    for movimento, condicoes_movimento in condicoes.items():
        for condicao, info in condicoes_movimento.items():
            epocas_dados = []
            for i in info['voluntarios']:
                pasta_dados = info['pasta_dados'] + (f'\S0{i}' if i <= 9 else f'\S{i}')
                voluntario = (f'\S0{i}' if i <= 9 else f'\S{i}') + '_' + info['epoca']
                epocas = mne.read_epochs(pasta_dados + voluntario + '.fif')
                epocas_dados.append(epocas.get_data(picks='eeg'))
                
            condicoes_epocas[condicao] = epocas
            num_voluntarios = len(epocas_dados)
            dados_erp[condicao] = np.zeros((num_voluntarios, 30, 6001))
            
            for indice in range(num_voluntarios):
                dados_erp[condicao][indice,:,:] = np.mean(epocas_dados[indice], axis=0)
                
            dados_erp_media[condicao] = np.mean(dados_erp[condicao], axis=0)
        
        dados_erp[f'{movimento}_small'], dados_erp_media[f'{movimento}_small'] = dados_erp['small'], dados_erp_media['small']
        dados_erp[f'{movimento}_big'], dados_erp_media[f'{movimento}_big'] = dados_erp['big'], dados_erp_media['big']
    
    return (dados_erp['left_small'], dados_erp_media['left_small'], 
            dados_erp['left_big'], dados_erp_media['left_big'], 
            dados_erp['right_small'], dados_erp_media['right_small'], 
            dados_erp['right_big'], dados_erp_media['right_big'])

dados_erp = {}
dados_erp_media = {}

(dados_erp['left_small'], dados_erp_media['left_small'], 
 dados_erp['left_big'], dados_erp_media['left_big'], 
 dados_erp['right_small'], dados_erp_media['right_small'], 
 dados_erp['right_big'], dados_erp_media['right_big']) = gerar_erp(condicoes)

# Parâmetros
canal = 'Oz'
inicio_tempo_pontos = 0
fim_tempo_pontos = 6000
rastro_imagem = (0.02)  # Tempo +- ao redor do pico na imagem
rastro_estatistica = (0.05)  # Tempo ao redor do pico na estatística
ponto = 600

intervalos = {
    'N200 de planejamento': (1320, 1320),
    'P300 de planejamento': (1320, 1400),
    'N400 de planejamento': (1390, 1440),
    'P300 de movimento': (2580, 2640),
    'N400 de movimento': (2640, 2640)
}

cores_eventos = {
    'N200 de planejamento': 'orange',
    'P300 de planejamento': 'green',
    'N400 de planejamento': 'red',
    'P300 de movimento': 'blue',
    'N400 de movimento': 'purple'
}

condicoes = ['left_small', 'right_small', 'left_big', 'right_big']
dados_canal = {condicao: [] for condicao in condicoes}

# Coleta dos dados dos voluntários e separação por condições
for condicao in condicoes:
    for sujeito in range(12):
        dados_sujeito = dados_erp[condicao][sujeito, layout.names.index(canal), inicio_tempo_pontos:fim_tempo_pontos]
        if dados_sujeito is not None and dados_sujeito.size > 0:
            dados_canal[condicao].append(dados_sujeito)

# Janela de ampliação para análise de picos
janela_ampliacao = 30

# Funções para encontrar picos
def encontrar_pico_maior_amplitude(dados, intervalo, janela_ampliacao=60):
    intervalo_ampliado = (
        max(0, intervalo[0] - janela_ampliacao),
        min(len(dados), intervalo[1] + janela_ampliacao)
    )
    dados_intervalo = dados[intervalo_ampliado[0]:intervalo_ampliado[1]]
    picos_negativos, _ = find_peaks(-dados_intervalo)
    if len(picos_negativos) > 0:
        amplitudes = [-dados_intervalo[pico] for pico in picos_negativos]
        indice_pico_maior_amplitude = picos_negativos[np.argmax(amplitudes)]
        tempo_pico_maior_amplitude = intervalo_ampliado[0] + indice_pico_maior_amplitude
        return tempo_pico_maior_amplitude
    return None

def encontrar_pico_positivo_maior_amplitude(dados, intervalo, janela_ampliacao=60):
    intervalo_ampliado = (
        max(0, intervalo[0] - janela_ampliacao),
        min(len(dados), intervalo[1] + janela_ampliacao)
    )
    dados_intervalo = dados[intervalo_ampliado[0]:intervalo_ampliado[1]]
    picos_positivos, _ = find_peaks(dados_intervalo)
    if len(picos_positivos) > 0:
        amplitudes = [dados_intervalo[pico] for pico in picos_positivos]
        indice_pico_maior_amplitude = picos_positivos[np.argmax(amplitudes)]
        tempo_pico_maior_amplitude = intervalo_ampliado[0] + indice_pico_maior_amplitude
        return tempo_pico_maior_amplitude
    return None

# Sincronizar picos e ajustar dados
def sincronizar_picos(dados, intervalo, tipo_pico='positivo'):
    pico_voluntarios = []
    dados_ajustados = []
    for sujeito in range(12):
        dados_sujeito = dados[sujeito]
        if dados_sujeito is not None and dados_sujeito.size > 0:
            if tipo_pico == 'positivo':
                tempo_pico = encontrar_pico_positivo_maior_amplitude(dados_sujeito, intervalo, janela_ampliacao)
            else:
                tempo_pico = encontrar_pico_maior_amplitude(dados_sujeito, intervalo, janela_ampliacao)
            if tempo_pico is not None:
                pico_voluntarios.append(tempo_pico)
                deslocamento = int(tempo_pico - (intervalo[0] + (intervalo[1] - intervalo[0]) // 2))
                dados_ajustados.append(np.roll(dados_sujeito, -deslocamento))
    return dados_ajustados

# Separar e sincronizar dados de planejamento e movimento
dados_ajustados_planning = {condicao: sincronizar_picos(dados_canal[condicao], intervalos['P300 de planejamento'], 'positivo') for condicao in condicoes}
dados_ajustados_movement = {condicao: sincronizar_picos(dados_canal[condicao], intervalos['P300 de movimento'], 'positivo') for condicao in condicoes}

# Calcular o Grand Average ERP
grand_average_erp_planning = {condicao: np.mean(dados_ajustados_planning[condicao], axis=0) for condicao in condicoes}
grand_average_erp_movement = {condicao: np.mean(dados_ajustados_movement[condicao], axis=0) for condicao in condicoes}

# Função de média móvel para suavização dos dados
def aplicar_media_movel(ERP, w=3):
    return np.convolve(ERP, np.ones(w) / w, mode='valid')

# Aplicar a média móvel para todos os canais nas condições de planejamento e movimento
grand_average_erp_planning_suave = {condicao: aplicar_media_movel(grand_average_erp_planning[condicao], w=3) for condicao in condicoes}
grand_average_erp_movement_suave = {condicao: aplicar_media_movel(grand_average_erp_movement[condicao], w=3) for condicao in condicoes}

# Plot 1: Todas as setas
fig1, axes1 = plt.subplots(2, 1, figsize=(10, 12), sharey=True)
conditions = ['right_small', 'right_big', 'left_small', 'left_big']

for condicao in conditions:
    media_condicao = grand_average_erp_planning_suave[condicao]  # Usando dados suavizados
    axes1[0].plot(np.arange(1200, 1800) / ponto, media_condicao[1200:1800], label='Seta Pequena' if 'small' in condicao else 'Seta Grande')
    for evento, intervalo in intervalos.items():
        if 'planejamento' in evento:
            if 'N' in evento:
                tempo_pico_maior_amplitude = encontrar_pico_maior_amplitude(media_condicao, intervalo, janela_ampliacao)
            else:
                tempo_pico_maior_amplitude = encontrar_pico_positivo_maior_amplitude(media_condicao, intervalo, janela_ampliacao)
            if tempo_pico_maior_amplitude is not None:
                color = cores_eventos[evento]
                axes1[0].axvspan(tempo_pico_maior_amplitude / ponto - rastro_imagem, tempo_pico_maior_amplitude / ponto + rastro_imagem, facecolor=color, alpha=0.5)
                axes1[0].axvline(x=tempo_pico_maior_amplitude / ponto, color=color, linestyle='', linewidth=1)
axes1[0].set_title(f'Canal {canal} - Planejamento (Grande e Pequena)')
axes1[0].set_xlabel('Tempo (s)')
axes1[0].set_ylabel("Amplitude (uV)")
axes1[0].legend(loc='upper right')

for condicao in conditions:
    media_condicao = grand_average_erp_movement_suave[condicao]  # Usando dados suavizados
    axes1[1].plot(np.arange(2400, 3000) / ponto, media_condicao[2400:3000], label='Seta Pequena' if 'small' in condicao else 'Seta Grande')
    for evento, intervalo in intervalos.items():
        if 'movimento' in evento:
            if 'N' in evento:
                tempo_pico_maior_amplitude = encontrar_pico_maior_amplitude(media_condicao, intervalo, janela_ampliacao)
            else:
                tempo_pico_maior_amplitude = encontrar_pico_positivo_maior_amplitude(media_condicao, intervalo, janela_ampliacao)
            if tempo_pico_maior_amplitude is not None:
                color = cores_eventos[evento]
                axes1[1].axvspan(tempo_pico_maior_amplitude / ponto - rastro_imagem, tempo_pico_maior_amplitude / ponto + rastro_imagem, facecolor=color, alpha=0.5)
                axes1[1].axvline(x=tempo_pico_maior_amplitude / ponto, color=color, linestyle='', linewidth=1)
axes1[1].set_title(f'Canal {canal} - Movimento (Grande e Pequena)')
axes1[1].set_xlabel('Tempo (s)')
axes1[1].set_ylabel("Amplitude (uV)")
axes1[1].legend(loc='upper right')

# Ajustar os limites de amplitude para ambos os gráficos
max_abs = max(np.max(np.abs(grand_average_erp_planning_suave[cond])) for cond in condicoes)
max_abs = max(max_abs, max(np.max(np.abs(grand_average_erp_movement_suave[cond])) for cond in condicoes))
for ax in axes1:
    ax.set_ylim([-max_abs * 1.25, max_abs * 1.25])

plt.tight_layout()
plt.show()

# Plot 2: Apenas com Esquerda
fig2, axes2 = plt.subplots(2, 1, figsize=(10, 12), sharey=True)
left_conditions = ['left_small', 'left_big']

for condicao in left_conditions:
    media_condicao = grand_average_erp_planning_suave[condicao]
    axes2[0].plot(np.arange(1200, 1800) / ponto, media_condicao[1200:1800], label='Seta Pequena' if 'small' in condicao else 'Seta Grande')
    for evento, intervalo in intervalos.items():
        if 'planejamento' in evento:
            if 'N' in evento:
                tempo_pico_maior_amplitude = encontrar_pico_maior_amplitude(media_condicao, intervalo, janela_ampliacao)
            else:
                tempo_pico_maior_amplitude = encontrar_pico_positivo_maior_amplitude(media_condicao, intervalo, janela_ampliacao)
            if tempo_pico_maior_amplitude is not None:
                color = cores_eventos[evento]
                axes2[0].axvspan(tempo_pico_maior_amplitude / ponto - rastro_imagem, tempo_pico_maior_amplitude / ponto + rastro_imagem, facecolor=color, alpha=0.5)
                axes2[0].axvline(x=tempo_pico_maior_amplitude / ponto, color=color, linestyle='', linewidth=1)
axes2[0].set_title(f'Canal {canal} - Planejamento para a seta Esquerda (Grande e Pequena)')
axes2[0].set_xlabel('Tempo (s)')
axes2[0].set_ylabel("Amplitude (uV)")
axes2[0].legend(loc='upper right')

for condicao in left_conditions:
    media_condicao = grand_average_erp_movement_suave[condicao]
    axes2[1].plot(np.arange(2400, 3000) / ponto, media_condicao[2400:3000], label='Seta Pequena' if 'small' in condicao else 'Seta Grande')
    for evento, intervalo in intervalos.items():
        if 'movimento' in evento:
            if 'N' in evento:
                tempo_pico_maior_amplitude = encontrar_pico_maior_amplitude(media_condicao, intervalo, janela_ampliacao)
            else:
                tempo_pico_maior_amplitude = encontrar_pico_positivo_maior_amplitude(media_condicao, intervalo, janela_ampliacao)
            if tempo_pico_maior_amplitude is not None:
                color = cores_eventos[evento]
                axes2[1].axvspan(tempo_pico_maior_amplitude / ponto - rastro_imagem, tempo_pico_maior_amplitude / ponto + rastro_imagem, facecolor=color, alpha=0.5)
                axes2[1].axvline(x=tempo_pico_maior_amplitude / ponto, color=color, linestyle='', linewidth=1)
axes2[1].set_title(f'Canal {canal} - Movimento para a seta Esquerda (Grande e Pequena)')
axes2[1].set_xlabel('Tempo (s)')
axes2[1].set_ylabel("Amplitude (uV)")
axes2[1].legend(loc='upper right')

max_abs = max(np.max(np.abs(grand_average_erp_planning_suave[cond])) for cond in condicoes)
max_abs = max(max_abs, max(np.max(np.abs(grand_average_erp_movement_suave[cond])) for cond in condicoes))
for ax in axes2:
    ax.set_ylim([-max_abs * 1.25, max_abs * 1.25])

plt.tight_layout()
plt.show()

# Plot 3: Apenas com Direita
fig3, axes3 = plt.subplots(2, 1, figsize=(10, 12), sharey=True)
right_conditions = ['right_small', 'right_big']

for condicao in right_conditions:
    media_condicao = grand_average_erp_planning_suave[condicao]  # Usando dados suavizados
    axes3[0].plot(np.arange(1200, 1800) / ponto, media_condicao[1200:1800], label='Seta Pequena' if 'small' in condicao else 'Seta Grande')
    for evento, intervalo in intervalos.items():
        if 'planejamento' in evento:
            if 'N' in evento:
                tempo_pico_maior_amplitude = encontrar_pico_maior_amplitude(media_condicao, intervalo, janela_ampliacao)
            else:
                tempo_pico_maior_amplitude = encontrar_pico_positivo_maior_amplitude(media_condicao, intervalo, janela_ampliacao)
            if tempo_pico_maior_amplitude is not None:
                color = cores_eventos[evento]
                axes3[0].axvspan(tempo_pico_maior_amplitude / ponto - rastro_imagem, tempo_pico_maior_amplitude / ponto + rastro_imagem, facecolor=color, alpha=0.5)
                axes3[0].axvline(x=tempo_pico_maior_amplitude / ponto, color=color, linestyle='', linewidth=1)
axes3[0].set_title(f'Canal {canal} - Planejamento para a seta Direita (Grande e Pequena)')
axes3[0].set_xlabel('Tempo (s)')
axes3[0].set_ylabel("Amplitude (uV)")
axes3[0].legend(loc='upper right')

for condicao in right_conditions:
    media_condicao = grand_average_erp_movement_suave[condicao]  # Usando dados suavizados
    axes3[1].plot(np.arange(2400, 3000) / ponto, media_condicao[2400:3000], label='Seta Pequena' if 'small' in condicao else 'Seta Grande')
    for evento, intervalo in intervalos.items():
        if 'movimento' in evento:
            if 'N' in evento:
                tempo_pico_maior_amplitude = encontrar_pico_maior_amplitude(media_condicao, intervalo, janela_ampliacao)
            else:
                tempo_pico_maior_amplitude = encontrar_pico_positivo_maior_amplitude(media_condicao, intervalo, janela_ampliacao)
            if tempo_pico_maior_amplitude is not None:
                color = cores_eventos[evento]
                axes3[1].axvspan(tempo_pico_maior_amplitude / ponto - rastro_imagem, tempo_pico_maior_amplitude / ponto + rastro_imagem, facecolor=color, alpha=0.5)
                axes3[1].axvline(x=tempo_pico_maior_amplitude / ponto, color=color, linestyle='', linewidth=1)
axes3[1].set_title(f'Canal {canal} - Movimento para a seta Direita (Grande e Pequena)')
axes3[1].set_xlabel('Tempo (s)')
axes3[1].set_ylabel("Amplitude (uV)")
axes3[1].legend(loc='upper right')

# Ajustar os limites de amplitude para ambos os gráficos
max_abs = max(np.max(np.abs(grand_average_erp_planning_suave[cond])) for cond in condicoes)
max_abs = max(max_abs, max(np.max(np.abs(grand_average_erp_movement_suave[cond])) for cond in condicoes))
for ax in axes3:
    ax.set_ylim([-max_abs * 1.25, max_abs * 1.25])

plt.tight_layout()
plt.show()

# Plot 4: Direita Planejamento x Esquerda Planejamento
fig4, axes4 = plt.subplots(2, 1, figsize=(10, 12), sharey=True)
for lado, conditions in zip(['Esquerda', 'Direita'], [left_conditions, right_conditions]):
    for condicao in conditions:
        media_condicao = grand_average_erp_planning_suave[condicao]
        axes4[0 if lado == 'Esquerda' else 1].plot(np.arange(1200, 1800) / ponto, media_condicao[1200:1800], label='Seta Pequena' if 'small' in condicao else 'Seta Grande')
        for evento, intervalo in intervalos.items():
            if 'planejamento' in evento:
                if 'N' in evento:
                    tempo_pico_maior_amplitude = encontrar_pico_maior_amplitude(media_condicao, intervalo, janela_ampliacao)
                else:
                    tempo_pico_maior_amplitude = encontrar_pico_positivo_maior_amplitude(media_condicao, intervalo, janela_ampliacao)
                if tempo_pico_maior_amplitude is not None:
                    color = cores_eventos[evento]
                    axes4[0 if lado == 'Esquerda' else 1].axvspan(tempo_pico_maior_amplitude / ponto - rastro_imagem, tempo_pico_maior_amplitude / ponto + rastro_imagem, facecolor=color, alpha=0.5)
                    axes4[0 if lado == 'Esquerda' else 1].axvline(x=tempo_pico_maior_amplitude / ponto, color=color, linestyle='', linewidth=1)
    axes4[0 if lado == 'Esquerda' else 1].set_title(f'Canal {canal} - Planejamento para a seta {lado}')
    axes4[0 if lado == 'Esquerda' else 1].set_xlabel('Tempo (s)')
    axes4[0 if lado == 'Esquerda' else 1].set_ylabel("Amplitude (uV)")
    axes4[0 if lado == 'Esquerda' else 1].legend(loc='upper right')

# Ajustar os limites de amplitude para ambos os gráficos
max_abs = max(np.max(np.abs(grand_average_erp_planning_suave[cond])) for cond in condicoes)
for ax in axes4:
    ax.set_ylim([-max_abs * 1.25, max_abs * 1.25])

plt.tight_layout()
plt.show()

# Plot 5: Direita Movimento x Esquerda Movimento
fig5, axes5 = plt.subplots(2, 1, figsize=(10, 12), sharey=True)
for lado, conditions in zip(['Esquerda', 'Direita'], [left_conditions, right_conditions]):
    for condicao in conditions:
        media_condicao = grand_average_erp_movement_suave[condicao]
        axes5[0 if lado == 'Esquerda' else 1].plot(np.arange(2400, 3000) / ponto, media_condicao[2400:3000], label='Seta Pequena' if 'small' in condicao else 'Seta Grande')
        for evento, intervalo in intervalos.items():
            if 'movimento' in evento:
                if 'N' in evento:
                    tempo_pico_maior_amplitude = encontrar_pico_maior_amplitude(media_condicao, intervalo, janela_ampliacao)
                else:
                    tempo_pico_maior_amplitude = encontrar_pico_positivo_maior_amplitude(media_condicao, intervalo, janela_ampliacao)
                if tempo_pico_maior_amplitude is not None:
                    color = cores_eventos[evento]
                    axes5[0 if lado == 'Esquerda' else 1].axvspan(tempo_pico_maior_amplitude / ponto - rastro_imagem, tempo_pico_maior_amplitude / ponto + rastro_imagem, facecolor=color, alpha=0.5)
                    axes5[0 if lado == 'Esquerda' else 1].axvline(x=tempo_pico_maior_amplitude / ponto, color=color, linestyle='', linewidth=1)
    axes5[0 if lado == 'Esquerda' else 1].set_title(f'Canal {canal} - Movimento para a seta {lado}')
    axes5[0 if lado == 'Esquerda' else 1].set_xlabel('Tempo (s)')
    axes5[0 if lado == 'Esquerda' else 1].set_ylabel("Amplitude (uV)")
    axes5[0 if lado == 'Esquerda' else 1].legend(loc='upper right')

# Ajustar os limites de amplitude para ambos os gráficos
max_abs = max(np.max(np.abs(grand_average_erp_movement_suave[cond])) for cond in condicoes)
for ax in axes5:
    ax.set_ylim([-max_abs * 1.25, max_abs * 1.25])

plt.tight_layout()
plt.show()

# Realize o teste de Wilcoxon com um espaço de 50ms ao redor dos picos
alpha = 0.05
comparacoes = [
    ('left_small', 'left_big'),
    ('right_small', 'right_big')
]

print(f"Canal {canal}:")
print("=" * 30)

# Dicionário para traduzir os nomes das condições
condicao_nomes = {
    'left_small': 'Seta Esquerda Pequena',
    'right_small': 'Seta Direita Pequena',
    'left_big': 'Seta Esquerda Grande',
    'right_big': 'Seta Direita Grande'
}

# Calcula os tempos médios de pico para cada evento
media_tempos = {
    evento: (intervalo[0] + intervalo[1]) / 2 / ponto for evento, intervalo in intervalos.items()
}

for evento, tempo_medio in media_tempos.items():
    intervalo = (int(tempo_medio * 600) - 30, int(tempo_medio * 600) + 30)
    print(evento)
    for comparacao in comparacoes:
        condicao1, condicao2 = comparacao
        nome_condicao1 = condicao_nomes[condicao1]
        nome_condicao2 = condicao_nomes[condicao2]
        dados_condicao1 = np.concatenate([dados[range(intervalo[0], intervalo[1])] for dados in dados_canal[condicao1]])
        dados_condicao2 = np.concatenate([dados[range(intervalo[0], intervalo[1])] for dados in dados_canal[condicao2]])
        
        # Teste de Wilcoxon
        _, p_value = wilcoxon(dados_condicao1, dados_condicao2, alternative='two-sided')
        diferenca_significativa = "Houve" if p_value < alpha else "Não houve"
        
        print(f"Entre {nome_condicao1} e {nome_condicao2}: valor de P = {p_value:.4f} ({diferenca_significativa} diferença estatística)")
    print("=" * 30)

def calcular_metrica_amplitude(dados, intervalo):
    """
    Calcula métricas para as amplitudes dentro do intervalo:
    - Média
    - Amplitude do Pico
    - Tempo do Pico
    - Desvio Padrão
    - Integral da Amplitude
    """
    # Determinar o início e o fim do intervalo em pontos
    inicio = max(0, int(intervalo[0] - 50 * ponto / 1000))  # -50ms em pontos
    fim = min(len(dados[0]), int(intervalo[1] + 50 * ponto / 1000))  # +50ms em pontos

    # Coletar amplitudes dentro do intervalo
    amplitudes = np.concatenate([dado[inicio:fim] for dado in dados if len(dado[inicio:fim]) > 0])

    if len(amplitudes) > 0:
        # Média
        media = np.mean(amplitudes)
        # Pico (máximo valor absoluto)
        amplitude_pico = np.max(np.abs(amplitudes))
        # Tempo do Pico (em pontos)
        indice_pico = np.argmax(np.abs(amplitudes)) + inicio
        tempo_pico = indice_pico / ponto  # Converter para segundos
        # Desvio Padrão
        desvio_padrao = np.std(amplitudes)
        # Integral da Amplitude
        integral = np.sum(amplitudes) / ponto  # Soma normalizada pelo número de pontos
    else:
        media = amplitude_pico = tempo_pico = desvio_padrao = integral = None

    return media, amplitude_pico, tempo_pico, desvio_padrao, integral

# Função para exibir todas as métricas para uma condição

def calcular_metricas_condicao_compacta(condicao, eventos, dados_canal):
    print(f"Condição {condicao}:")
    print("=" * 50)
    resultados = []
    for evento, intervalo in intervalos.items():
        # Calcular métricas
        metricas = calcular_metrica_amplitude(dados_canal[condicao], intervalo)

        if all(m is not None for m in metricas):
            media, pico, tempo_pico, desvio, integral = metricas

            # Ajustar polaridade para N200/N400 (negativo) e P300 (positivo)
            if "N400" in evento or "N200" in evento:
                media, pico, integral = -abs(media), -abs(pico), -abs(integral)
            elif "P300" in evento:
                media, pico, integral = abs(media), abs(pico), abs(integral)

            resultados.append(
                f"  {evento} -> Média: {media * 1e6:.2f} µV // "
                f"Pico: {pico * 1e6:.2f} µV // "
                f"Tempo do Pico: {tempo_pico * 1e3:.2f} ms // "
                f"Desvio: {desvio * 1e6:.2f} µV // "
                f"Integral: {integral * 1e6:.2f} µV·s"
            )
        else:
            resultados.append(f"  {evento} -> Nenhum dado disponível")
    
    # Exibir resultados
    print("\n".join(resultados))
    print("=" * 50)

# Chamada para cada condição, separadamente
print("\nResultados Compactos para Left Small:")
calcular_metricas_condicao_compacta("left_small", intervalos, dados_canal)

print("\nResultados Compactos para Right Small:")
calcular_metricas_condicao_compacta("right_small", intervalos, dados_canal)

print("\nResultados Compactos para Left Big:")
calcular_metricas_condicao_compacta("left_big", intervalos, dados_canal)

print("\nResultados Compactos para Right Big:")
calcular_metricas_condicao_compacta("right_big", intervalos, dados_canal)
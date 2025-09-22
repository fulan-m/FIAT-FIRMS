# O comentário com o seguinte símbolo: " # %% " é uma função disponível no VS Code que integra
# cadernos Jupyter (Jupyter Notebooks) com scripts .py. O código funcionará normalmente sem o
# símbolo, mas é recomendado remover a porção final que carrega visualizações possíveis apenas
# em cadernos.

# %%
import os
import ast
import json
import rasterio
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import HTML
from matplotlib.animation import FuncAnimation

# Configurar estilo visual - AUMENTAR AQUI TAMBÉM
plt.style.use('default')
mpl.rcParams['font.size'] = 14
mpl.rcParams['font.family'] = 'arial'
mpl.rcParams['axes.titlesize'] = 16  # Título do gráfico
mpl.rcParams['axes.labelsize'] = 14  # Rótulos dos eixos
mpl.rcParams['xtick.labelsize'] = 14  # Labels do eixo X (classes)
mpl.rcParams['ytick.labelsize'] = 14  # Labels do eixo Y
mpl.rcParams['legend.fontsize'] = 14  # Legenda (se houver)

# Função para ler o JSON com a legenda
def load_legend(json_path):
    """
    Carrega o arquivo JSON com a legenda do MapBiomas
    
    Parâmetros:
    json_path (str): Caminho para o arquivo JSON
    
    Retorna:
    dict: Dicionário com a legenda (code_id como chave)
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            legend_data = json.load(f)
        return legend_data
    except Exception as e:
        print(f"Erro ao carregar o arquivo JSON: {e}")
        return {}

# Função para processar o raster
def read_raster_and_get_value_counts(raster_path, legend_data):
    """
    Lê um raster e retorna um DataFrame com estatísticas dos valores únicos,
    ignorando o valor 0 (no data) e adicionando informações da legenda.
    
    Parâmetros:
    raster_path (str): Caminho para o arquivo raster
    legend_data (dict): Dicionário com dados da legenda
    
    Retorna:
    pandas.DataFrame: DataFrame com colunas 'classe', 'cont_px', 'porcent', 'nome_pt', 'cor_hex'
    """
    with rasterio.open(raster_path) as src:
        raster_data = src.read(1)  # Read the first band
        unique, counts = np.unique(raster_data, return_counts=True)
        
        # Filtrar para remover o valor 0 (no data)
        mask = unique != 0
        unique_filtered = unique[mask]
        counts_filtered = counts[mask]
        
        # Calcular o total de pixels válidos (excluindo 0)
        total_pixels = np.sum(counts_filtered)
        
        # Calcular porcentagens
        percentages = [(count / total_pixels) * 100 for count in counts_filtered]
        
        # Adicionar informações da legenda
        nomes_pt = []
        cores_hex = []
        
        for code in unique_filtered:
            code_str = str(code)
            if code_str in legend_data:
                nomes_pt.append(legend_data[code_str]['PT'])
                cores_hex.append(legend_data[code_str]['HEX_COL'])
            else:
                nomes_pt.append(f"Classe {code} (não encontrada)")
                cores_hex.append("#808080")  # Cinza para classes não encontradas
        
        # Criar DataFrame
        df = pd.DataFrame({
            'classe': unique_filtered,
            'cont_px': counts_filtered,
            'porcent': percentages,
            'nome_pt': nomes_pt,
            'cor_hex': cores_hex
        })
        
        # Ordenar por contagem de pixels (decrescente)
        df = df.sort_values('cont_px', ascending=False).reset_index(drop=True)
    
    return df

# Função para criar um dataframe unificado com todas as classes
def create_unified_dataframe(dataframes, years, top_n=5):
    """
    Cria um DataFrame unificado com todas as classes presentes em qualquer ano,
    preenchendo com zeros quando uma classe não está presente em um ano específico.
    
    Parâmetros:
    dataframes (list): Lista de DataFrames para cada ano
    years (list): Lista de anos
    top_n (int): Número de classes principais a considerar
    
    Retorna:
    pandas.DataFrame: DataFrame unificado com todas as classes
    """
    # Coletar todas as classes únicas em todos os anos
    all_classes = set()
    for df in dataframes:
        all_classes.update(df['classe'].head(top_n).tolist())
    
    # Criar DataFrame unificado
    unified_data = []
    for year, df in zip(years, dataframes):
        # Obter as top_n classes para este ano
        top_classes = df.head(top_n)['classe'].tolist()
        
        # Para cada classe nas top_n globais
        for classe in all_classes:
            # Verificar se a classe está presente neste ano
            if classe in df['classe'].values:
                row = df[df['classe'] == classe].iloc[0]
                unified_data.append({
                    'year': year,
                    'classe': classe,
                    'porcent': row['porcent'],
                    'nome_pt': row['nome_pt'],
                    'cor_hex': row['cor_hex'],
                    'rank': top_classes.index(classe) + 1 if classe in top_classes else top_n + 1
                })
            else:
                # Se a classe não está presente, adicionar com porcentagem zero
                # Precisamos encontrar o nome e cor desta classe em outro ano
                found = False
                for other_df in dataframes:
                    if classe in other_df['classe'].values:
                        other_row = other_df[other_df['classe'] == classe].iloc[0]
                        unified_data.append({
                            'year': year,
                            'classe': classe,
                            'porcent': 0,
                            'nome_pt': other_row['nome_pt'],
                            'cor_hex': other_row['cor_hex'],
                            'rank': top_n + 1
                        })
                        found = True
                        break
                
                if not found:
                    # Se não encontramos a classe em nenhum ano (improvável)
                    unified_data.append({
                        'year': year,
                        'classe': classe,
                        'porcent': 0,
                        'nome_pt': f"Classe {classe}",
                        'cor_hex': "#808080",
                        'rank': top_n + 1
                    })
    
    return pd.DataFrame(unified_data)

# Função para criar a animação suavizada
def create_smooth_animation(years, unified_df, top_n=5, output_path=None, frames_per_year=10):
    """
    Cria uma animação suavizada dos histogramas ao longo dos anos
    
    Parâmetros:
    years (list): Lista de anos
    unified_df (DataFrame): DataFrame unificado com dados de todos os anos
    top_n (int): Número de classes principais a mostrar
    output_path (str): Caminho para salvar a animação (opcional)
    frames_per_year (int): Número de frames de transição entre anos
    
    Retorna:
    matplotlib.animation.FuncAnimation: Objeto de animação
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Preparar dados para animação
    all_years = []
    for i in range(len(years) - 1):
        # Adicionar anos inteiros
        all_years.append(years[i])
        
        # Adicionar anos interpolados para transição suave
        for j in range(1, frames_per_year):
            all_years.append(years[i] + j * (1.0 / frames_per_year))
    
    all_years.append(years[-1])
    
    # Preparar dados interpolados
    interpolated_data = []
    for year in all_years:
        if year in years:  # Ano exato
            year_data = unified_df[unified_df['year'] == year].copy()
            interpolated_data.append(year_data)
        else:  # Ano interpolado
            # Encontrar os anos entre os quais estamos interpolando
            year_prev = int(year)
            year_next = year_prev + 1
            
            # Obter dados para ambos os anos
            data_prev = unified_df[unified_df['year'] == year_prev].copy()
            data_next = unified_df[unified_df['year'] == year_next].copy()
            
            # Calcular fator de interpolação (0 a 1)
            alpha = year - year_prev
            
            # Interpolar porcentagens
            interpolated = data_prev.copy()
            for idx, row in interpolated.iterrows():
                classe = row['classe']
                if classe in data_next['classe'].values:
                    next_val = data_next[data_next['classe'] == classe]['porcent'].values[0]
                    interpolated.at[idx, 'porcent'] = row['porcent'] * (1 - alpha) + next_val * alpha
            
            interpolated_data.append(interpolated)
    
    # Função de atualização para animação
    def update(frame):
        ax.clear()
        year_data = interpolated_data[frame]
        year = all_years[frame]
        
        # Ordenar por porcentagem (decrescente)
        year_data = year_data.sort_values('porcent', ascending=False)
        
        # Selecionar apenas as top_n classes (SEM adicionar "Outras classes")
        year_data_top = year_data.head(top_n).copy()
        
        # Converter cores hex para formato matplotlib
        colors = [ast.literal_eval(f"'{color}'") for color in year_data_top['cor_hex']]
        
        # Criar gráfico de barras
        bars = ax.bar(year_data_top['nome_pt'], year_data_top['porcent'], color=colors, alpha=0.8)
        
        # Configurar o gráfico - MOSTRAR APENAS O ANO INTEIRO NO TÍTULO
        ax.set_title(f"Evolução das {top_n} maiores classes - {int(year)}", 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Classes', fontsize=12)
        ax.set_ylabel('Porcentagem (%)', fontsize=12)
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        # Definir limite do eixo Y para manter consistência
        max_percent = unified_df['porcent'].max()
        ax.set_ylim(0, max_percent * 1.1)
        
        # Adicionar valores nas barras
        for i, (bar, porcent) in enumerate(zip(bars, year_data_top['porcent'])):
            height = bar.get_height()
            if height > 1:  # Só adicionar texto se a barra for suficientemente alta
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{porcent:.1f}%', ha='center', va='bottom', rotation=0, fontsize=9)
        
        # REMOVIDO: Adicionar ano no canto superior direito (caixa amarela)
        # Apenas o título já mostra o ano
        
        # Ajustar layout
        plt.tight_layout()
        
        return ax
    
    # Criar animação
    ani = FuncAnimation(fig, update, frames=len(all_years), interval=50, repeat=True)
    
    if output_path:
        # Salvar como GIF
        ani.save(output_path, writer='pillow', fps=180, dpi=300)
    
    return ani

# Caminhos dos arquivos (substitua com seus caminhos)
# Disponível no repositório
json_path = "mapbiomas_colec_10.json"

# Supondo que seus arquivos estão no padrão: classificacao_1985.tif, classificacao_1986.tif, etc.
# A imagem deve ser um raster com 1 banda, contendo valores inteiros representando as classes.
base_raster_path = "classificacao_{}.tif"

# Carregar a legenda
legend_data = load_legend(json_path)

# Processar dados para cada ano (1985-1990)
years = list(range(1985, 2025))
dataframes = []

for year in years:
    raster_path = base_raster_path.format(year)
    
    # Verificar se o arquivo existe
    if not os.path.exists(raster_path):
        print(f"Arquivo não encontrado: {raster_path}")
        continue
    
    print(f"Processando ano {year}...")
    df_resultado = read_raster_and_get_value_counts(raster_path, legend_data)
    dataframes.append(df_resultado)
    
    print(f"Ano {year} processado. Total de pixels válidos: {df_resultado['cont_px'].sum():,}")
    print(f"Top 5 classes: {df_resultado.head(5)['nome_pt'].tolist()}\n")

# Criar DataFrame unificado com todas as classes
unified_df = create_unified_dataframe(dataframes, years, top_n=5)

# Criar animação suavizada
ani = create_smooth_animation(years, unified_df, top_n=5, 
                             output_path="histogram_animation_smooth_1985_2024.gif",
                             frames_per_year=10)

# Mostrar animação no notebook (se estiver usando Jupyter)
HTML(ani.to_jshtml())

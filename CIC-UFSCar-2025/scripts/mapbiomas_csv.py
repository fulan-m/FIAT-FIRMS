import os
import json
import rasterio
import numpy as np
import pandas as pd

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

# Função para processar o raster e calcular estatísticas
def process_raster_for_csv(raster_path, year, legend_data):
    """
    Lê um raster e retorna um DataFrame com as estatísticas solicitadas
    
    Parâmetros:
    raster_path (str): Caminho para o arquivo raster
    year (int): Ano do raster
    legend_data (dict): Dicionário com dados da legenda
    
    Retorna:
    pandas.DataFrame: DataFrame com colunas 'ano', 'classe', 'num_px', 'porc_rel'
    """
    with rasterio.open(raster_path) as src:
        raster_data = src.read(1)  # Lê a primeira banda
        unique, counts = np.unique(raster_data, return_counts=True)
        
        # Filtrar para remover o valor 0 (no data)
        mask = unique != 0
        unique_filtered = unique[mask]
        counts_filtered = counts[mask]
        
        # Calcular o total de pixels válidos (excluindo 0)
        total_pixels = np.sum(counts_filtered)
        
        # Calcular porcentagens relativas
        percentages = [(count / total_pixels) * 100 for count in counts_filtered]
        
        # Criar DataFrame com os dados solicitados
        df = pd.DataFrame({
            'ano': year,
            'classe': unique_filtered,
            'num_px': counts_filtered,
            'porc_rel': percentages
        })
        
        # Ordenar por número de pixels (decrescente)
        df = df.sort_values('num_px', ascending=False).reset_index(drop=True)
    
    return df

# Caminhos dos arquivos (substitua com seus caminhos)
json_path = "C:\\Users\\mateu\\OneDrive\\Projetos\\FIAT_FIRMS\\FIAT_FIRMS-dados\\dicionarios_json\\mapbiomas_colec_10.json"
base_raster_path = "C:\\Users\\mateu\\OneDrive\\Projetos\\FIAT_FIRMS\\FIAT_FIRMS-dados\\tiff\\mapbiomas_cerradoSP_1985_2024_30m\\classificacao_{}.tif"

# Carregar a legenda
legend_data = load_legend(json_path)

# Lista para armazenar todos os dataframes
all_dataframes = []

# Processar dados para cada ano (2012-2024 conforme seu exemplo)
years = list(range(1985, 2025))

for year in years:
    raster_path = base_raster_path.format(year)
    
    # Verificar se o arquivo existe
    if not os.path.exists(raster_path):
        print(f"Arquivo não encontrado: {raster_path}")
        continue
    
    print(f"Processando ano {year}...")
    
    try:
        df_ano = process_raster_for_csv(raster_path, year, legend_data)
        all_dataframes.append(df_ano)
        
        print(f"Ano {year} processado. Total de pixels válidos: {df_ano['num_px'].sum():,}")
        print(f"Número de classes encontradas: {len(df_ano)}")
        
    except Exception as e:
        print(f"Erro ao processar ano {year}: {e}")
        continue

# Concatenar todos os dataframes
if all_dataframes:
    final_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Ordenar por ano e depois por número de pixels (decrescente)
    final_df = final_df.sort_values(['ano', 'num_px'], ascending=[True, False]).reset_index(drop=True)
    
    # Mostrar informações do dataframe final
    print(f"\nDataframe final criado com sucesso!")
    print(f"Total de registros: {len(final_df):,}")
    print(f"Anos processados: {final_df['ano'].unique()}")
    print(f"Classes únicas: {final_df['classe'].nunique()}")
    
    # Mostrar as primeiras linhas
    print("\nPrimeiras 10 linhas do dataframe:")
    print(final_df.head(10))
    
    # Salvar como CSV
    csv_path = "C:\\Users\\mateu\\OneDrive\\Projetos\\FIAT_FIRMS\\FIAT_FIRMS-CIC_2025\\resultados\\classes_mapbiomas_1985-2024.csv"
    
    # Criar diretório se não existir
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    final_df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"\nArquivo CSV salvo em: {csv_path}")
    
    # Estatísticas resumidas
    print("\nEstatísticas resumidas:")
    print(f"Período: {final_df['ano'].min()} - {final_df['ano'].max()}")
    print(f"Total de anos: {final_df['ano'].nunique()}")
    print(f"Total de classes únicas: {final_df['classe'].nunique()}")
    print(f"Total de registros: {len(final_df):,}")
    
else:
    print("Nenhum dado foi processado. Verifique os caminhos dos arquivos.")

# Opcional: Mostrar estatísticas por ano
if all_dataframes:
    print("\nEstatísticas por ano:")
    stats_por_ano = final_df.groupby('ano').agg({
        'classe': 'count',
        'num_px': 'sum',
        'porc_rel': 'sum'
    }).rename(columns={'classe': 'qtd_classes', 'num_px': 'total_pixels', 'porc_rel': 'soma_porcentagens'})
    
    print(stats_por_ano)

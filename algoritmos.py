from concurrent.futures import ThreadPoolExecutor
import numpy as np
import os
from pyarrow.parquet import ParquetFile
import pyarrow.parquet as pq
# import pyarrow as pa
import pandas as pd
import random


# Algoritmo 1: Contagem de arquivos por tipo

# Complexidade O(n)
def contar_arquivos_por_tipo_linear(data_lake_path):
    tipos_arquivos = set()
    for root, _, files in os.walk(data_lake_path):
        for file in files:
            _, ext = os.path.splitext(file)
            tipos_arquivos.add(ext)
    return len(tipos_arquivos)


# Complexidade O(n^2)
def contar_arquivos_por_tipo_quadratico(data_lake_path):
    tipos_arquivos = set()
    for root, _, files in os.walk(data_lake_path):
        for file1 in files:
            _, ext1 = os.path.splitext(file1)
            for file2 in files:
                _, ext2 = os.path.splitext(file2)
                if ext1 == ext2:
                    tipos_arquivos.add(ext1)
    return len(tipos_arquivos)


# Complexidade O(log n)
def contar_arquivos_por_tipo_logaritmico(data_lake_path):
    tipos_arquivos = {}
    for root, _, files in os.walk(data_lake_path):
        for file in files:
            _, ext = os.path.splitext(file)
            if ext in tipos_arquivos:
                tipos_arquivos[ext] += 1
            else:
                tipos_arquivos[ext] = 1
    return len(tipos_arquivos)


# Algoritmo 2: Descoberta de esquema de dados

# Complexidade O(n)
def descobrir_esquema_dados(data_lake_path):
    esquemas = []
    for root, _, files in os.walk(data_lake_path):
        for file in files:
            if file.endswith('.parquet'):
                file_path = os.path.join(root, file)
                parquet_file = ParquetFile(file_path)
                esquemas.append(parquet_file.schema)
    return esquemas


# Complexidade O(n)
def descobrir_esquema_dados_pyarrow(data_lake_path):
    esquemas = []
    for root, _, files in os.walk(data_lake_path):
        for file in files:
            if file.endswith('.parquet'):
                file_path = os.path.join(root, file)
                parquet_file = ParquetFile(file_path)
                esquemas.append(parquet_file.schema)
    return esquemas


# Complexidade O(n^2)
def descobrir_esquema_dados_quadratico(data_lake_path):
    esquemas = []
    arquivos = []

    for root, _, files in os.walk(data_lake_path):
        for file in files:
            if file.endswith('.parquet'):
                arquivos.append(os.path.join(root, file))

    for file1 in arquivos:
        parquet_file1 = ParquetFile(file1)
        schema1 = parquet_file1.schema
        for file2 in arquivos:
            parquet_file2 = ParquetFile(file2)
        esquemas.append(schema1)

    return esquemas


# Complexidade O(log n)
def descobrir_esquema_dados_logaritmico(data_lake_path):
    esquemas = []
    for root, _, files in os.walk(data_lake_path):
        for file in files:
            if file.endswith('.parquet'):
                file_path = os.path.join(root, file)
                parquet_file = ParquetFile(file_path)
                esquemas.append(parquet_file.schema)
    return esquemas


# Algoritmo 3: Análise de dados faltantes por bloco de colunas

def obter_caminhos_parquet(data_lake_path):
    caminhos_parquet = []

    for root, dirs, files in os.walk(data_lake_path):
        for file in files:
            if file.endswith(".parquet"):
                caminhos_parquet.append(os.path.join(root, file))

    return caminhos_parquet

# Variação 1: Análise de um bloco de colunas por vez (O(n))
def analisar_dados_faltantes_variacao1(data_lake_path, tamanho_bloco):
    caminhos_parquet = obter_caminhos_parquet(data_lake_path)
    dados_faltantes = {}

    for caminho_parquet in caminhos_parquet:
        table = pq.read_table(caminho_parquet)
        dataframe = table.to_pandas()

        blocos = np.array_split(dataframe.columns, len(dataframe.columns) // tamanho_bloco + 1)

        for bloco in blocos:
            bloco_dados = dataframe[bloco]
            dados_faltantes_bloco = bloco_dados.isnull().sum()
            dados_faltantes.update(dados_faltantes_bloco)

    return dados_faltantes

# Variação 2: Análise de um bloco de colunas por vez, com otimização de acesso aos dados (O(n))
def analisar_dados_faltantes_variacao2(data_lake_path, tamanho_bloco):
    caminhos_parquet = obter_caminhos_parquet(data_lake_path)
    dados_faltantes = {}

    for caminho_parquet in caminhos_parquet:
        table = pq.read_table(caminho_parquet)
        dataframe = table.to_pandas()

        colunas = dataframe.columns
        blocos = np.array_split(colunas, len(colunas) // tamanho_bloco + 1)

        for bloco in blocos:
            bloco_dados = dataframe[bloco]
            dados_faltantes_bloco = bloco_dados.isnull().sum()
            dados_faltantes.update(dados_faltantes_bloco)

    return dados_faltantes

# Variação 3: Análise paralela por bloco de colunas (O(n/p))
def analisar_dados_faltantes_variacao3(data_lake_path, tamanho_bloco):
    caminhos_parquet = obter_caminhos_parquet(data_lake_path)
    dados_faltantes = {}

    def processar_parquet(caminho_parquet):
        table = pq.read_table(caminho_parquet)
        dataframe = table.to_pandas()

        colunas = dataframe.columns
        blocos = np.array_split(colunas, len(colunas) // tamanho_bloco + 1)
        dados_faltantes_bloco = {}

        for bloco in blocos:
            bloco_dados = dataframe[bloco]
            dados_faltantes_bloco.update(bloco_dados.isnull().sum())

        return dados_faltantes_bloco

    with ThreadPoolExecutor() as executor:
        resultados = executor.map(processar_parquet, caminhos_parquet)

    for resultado in resultados:
        dados_faltantes.update(resultado)

    return dados_faltantes


# Algoritmo 4: Consulta de dados

# Complexidade O(n)
def consultar_dados(data_lake_path, criterios_pyarrow):
    dados_encontrados = []
    for root, _, files in os.walk(data_lake_path):
        for file in files:
            if file.endswith('.parquet'):
                file_path = os.path.join(root, file)
                parquet_file = ParquetFile(file_path)
                table = parquet_file.read()
                
                if 'val_geracaopctmwmed' in table.columns:
                    query_result = table.filter(criterios_pyarrow)
                    dados_encontrados.extend(query_result)
                else:
                    # A coluna não está presente neste arquivo, pule para o próximo
                    continue
    
    return dados_encontrados

# Complexidade O(n)
def consultar_dados_pyarrow(data_lake_path, criterios):
    dados_encontrados = []
    for root, _, files in os.walk(data_lake_path):
        for file in files:
            if file.endswith('.parquet'):
                file_path = os.path.join(root, file)
                parquet_file = pq.ParquetFile(file_path)
                table = parquet_file.read()
                df = table.to_pandas()
                if 'val_geracaopctmwmed' in df.columns:
                    query_result = df.query(criterios)
                    if not query_result.empty:
                        dados_encontrados.append(query_result)
    if dados_encontrados:
        return pd.concat(dados_encontrados)
    else:
        return pd.DataFrame()


# Complexidade O(n^2)
def consultar_dados_quadratico(data_lake_path, criterios):
    dados_encontrados = []
    for root, _, files in os.walk(data_lake_path):
        for file1 in files:
            if file1.endswith('.parquet'):
                file_path = os.path.join(root, file1)
                try:
                    table = pq.read_table(file_path)
                    df = table.to_pandas()
                    try:
                        query_result = df.query(criterios)
                        dados_encontrados.extend(query_result.to_dict('records'))
                    except pd.core.computation.ops.UndefinedVariableError:
                        # Tratar exceção quando a coluna não estiver presente
                        continue
                except:
                    # Tratar exceção quando ocorrer algum erro na leitura do arquivo Parquet
                    continue
    return dados_encontrados


# Complexidade O(log n)
def consultar_dados_logaritmico(data_lake_path, criterios):
    dados_encontrados = []
    for root, _, files in os.walk(data_lake_path):
        for file in files:
            if file.endswith('.parquet'):
                file_path = os.path.join(root, file)
                try:
                    table = pq.read_table(file_path)
                    df = table.to_pandas()
                    query_result = df.query(criterios)
                    dados_encontrados.extend(query_result.to_dict('records'))
                except:
                    # Tratar exceção quando ocorrer algum erro na leitura do arquivo Parquet
                    continue
    return dados_encontrados


# Algoritmo 5: Extração de amostras aleatórias

# Complexidade O(n)
def extrair_amostra_aleatoria(data_lake_path, porcentagem_amostra):
    amostra = []
    for root, _, files in os.walk(data_lake_path):
        for file in files:
            if file.endswith('.parquet'):
                file_path = os.path.join(root, file)
                try:
                    parquet_file = pq.ParquetFile(file_path)
                    table = parquet_file.read()
                    df = table.to_pandas()
                    num_rows = df.shape[0]
                    num_amostras = int(num_rows * porcentagem_amostra)
                    amostra.extend(df.sample(num_amostras))
                except:
                    # Tratar exceção quando ocorrer algum erro na leitura do arquivo Parquet
                    continue
    return amostra


# Complexidade O(n)
def extrair_amostra_aleatoria_pyarrow(data_lake_path, porcentagem_amostra):
    amostra = []
    for root, _, files in os.walk(data_lake_path):
        for file in files:
            if file.endswith('.parquet'):
                file_path = os.path.join(root, file)
                parquet_file = pq.ParquetFile(file_path)
                table = parquet_file.read()
                df = table.to_pandas()
                num_rows = df.shape[0]
                num_amostras = int(num_rows * porcentagem_amostra)
                random_indices = random.sample(range(num_rows), num_amostras)
                amostra.extend(df.iloc[random_indices])
    return amostra


# Complexidade O(n^2)
def extrair_amostra_aleatoria_quadratico(data_lake_path, porcentagem_amostra):
    amostra = []
    for root, _, files in os.walk(data_lake_path):
        for file1 in files:
            if file1.endswith('.parquet'):
                file_path = os.path.join(root, file1)
                parquet_file = pq.ParquetFile(file_path)
                table = parquet_file.read()
                df = table.to_pandas()
                num_rows = df.shape[0]
                num_amostras = int(num_rows * porcentagem_amostra)
                random_indices = random.sample(range(num_rows), num_amostras)
                amostra.extend(df.iloc[random_indices])
    return amostra


# Complexidade O(log n)
def extrair_amostra_aleatoria_logaritmico(data_lake_path, porcentagem_amostra):
    amostra = []
    for root, _, files in os.walk(data_lake_path):
        for file in files:
            if file.endswith('.parquet'):
                file_path = os.path.join(root, file)
                parquet_file = pq.ParquetFile(file_path)
                table = parquet_file.read()
                df = table.to_pandas()
                num_rows = df.shape[0]
                num_amostras = int(num_rows * porcentagem_amostra)
                random_indices = random.sample(range(num_rows), num_amostras)
                amostra.extend(df.iloc[random_indices])
    return amostra


# Algoritmo 6: Análise de distribuição de dados

# Complexidade O(n)
def analisar_distribuicao_dados(data_lake_path, coluna):
    distribuicao = []
    for root, _, files in os.walk(data_lake_path):
        for file in files:
            if file.endswith('.parquet'):
                file_path = os.path.join(root, file)
                parquet_file = ParquetFile(file_path)
                if coluna in parquet_file.schema.names:
                    table = parquet_file.read(columns=[coluna])
                    distribuicao.extend(table.to_pandas()[coluna])
    return distribuicao


# Complexidade O(n)
def analisar_distribuicao_dados_pyarrow(data_lake_path, coluna):
    distribuicao = []
    for root, _, files in os.walk(data_lake_path):
        for file in files:
            if file.endswith('.parquet'):
                file_path = os.path.join(root, file)
                parquet_file = ParquetFile(file_path)
                table = parquet_file.read(columns=[coluna])
                try:
                    distribuicao.extend(table.to_pandas()[coluna])
                except KeyError:
                    continue
    return distribuicao


# Complexidade O(n^2)
def analisar_distribuicao_dados_quadratico(data_lake_path, coluna):
    distribuicao = []
    for root, _, files in os.walk(data_lake_path):
        for file1 in files:
            if file1.endswith('.parquet'):
                file_path = os.path.join(root, file1)
                parquet_file = ParquetFile(file_path)
                table = parquet_file.read(columns=[coluna])
                try:
                    distribuicao.extend(table.to_pandas()[coluna])
                except KeyError:
                    continue
    return distribuicao


# Complexidade O(log n)
def analisar_distribuicao_dados_logaritmico(data_lake_path, coluna):
    distribuicao = []
    for root, _, files in os.walk(data_lake_path):
        for file in files:
            if file.endswith('.parquet'):
                file_path = os.path.join(root, file)
                parquet_file = ParquetFile(file_path)
                schema = parquet_file.schema
                if coluna in schema.names:
                    table = parquet_file.read(columns=[coluna])
                    distribuicao.extend(table.to_pandas()[coluna])
    return distribuicao

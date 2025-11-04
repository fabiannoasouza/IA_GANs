import argparse
import os
import sys
import requests
import time

def instalarReplicate():
    """Verifica e instala a biblioteca replicate, se necessário."""
    try:
        import replicate
        print("Biblioteca 'replicate' já está instalada.")
    except ImportError:
        import subprocess
        print("Instalando a biblioteca 'replicate'...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "replicate"])
        print("Instalação concluída.")

def restaurarComAPI(caminhoInput, apiToken):
    """
    Restaura rostos em uma imagem usando a API do Replicate para o GFPGAN.
    """
    # Define o token da API como uma variável de ambiente
    os.environ["REPLICATE_API_TOKEN"] = apiToken

    import replicate

    print(f"Processando a imagem: {caminhoInput}")

    # O identificador do modelo GFPGAN no Replicate
    modelo = "tencentarc/gfpgan:9283608cc6b7be6b65a8e44983db012355fde4132009bf99d976b2f0896856a3"

    try:
        with open(caminhoInput, "rb") as imagemArquivo:
            # Inicia a predição na API do Replicate
            print("Enviando imagem para a API do Replicate...")
            outputUrl = replicate.run(
                modelo,
                input={"img": imagemArquivo}
            )
            print(f"Processamento concluído. URL da imagem restaurada: {outputUrl}")

        # --- Baixar o Resultado ---
        if outputUrl:
            print("Baixando a imagem restaurada...")
            resposta = requests.get(outputUrl, stream=True)
            resposta.raise_for_status() # Lança um erro se a requisição falhar

            # Cria o diretório de saída se ele não existir
            pastaOutput = 'resultados'
            if not os.path.exists(pastaOutput):
                os.makedirs(pastaOutput)

            # Define o caminho do arquivo de saída
            nomeBase = os.path.basename(caminhoInput)
            nomeArquivo, extensaoArquivo = os.path.splitext(nomeBase)
            caminhoOutput = os.path.join(pastaOutput, f"{nomeArquivo}_restaurado_api{extensaoArquivo}")

            # Salva a imagem
            with open(caminhoOutput, 'wb') as f:
                for chunk in resposta.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"Imagem restaurada com sucesso e salva em: {caminhoOutput}")
        else:
            print("Erro: A API não retornou uma URL de saída.")

    except Exception as e:
        print(f"Ocorreu um erro durante a comunicação com a API: {e}")


if __name__ == "__main__":
    # Garante que a biblioteca 'replicate' está instalada
    instalarReplicate()

    parser = argparse.ArgumentParser(description="Restaura rostos em imagens usando a API do GFPGAN no Replicate.")
    parser.add_argument(
        'caminhoInput',
        type=str,
        help="O caminho para a imagem de entrada que você deseja restaurar."
    )

    argumentos = parser.parse_args()

    # Pega o token da variável de ambiente
    apiToken = os.getenv("REPLICATE_API_TOKEN")
    if not apiToken:
        print("Erro: O token da API do Replicate não foi encontrado.")
        print("Por favor, defina a variável de ambiente REPLICATE_API_TOKEN.")
        sys.exit(1)

    if os.path.isfile(argumentos.caminhoInput):
        restaurarComAPI(argumentos.caminhoInput, apiToken)
    else:
        print(f"Erro: O arquivo '{argumentos.caminhoInput}' não foi encontrado.")
        sys.exit(1)

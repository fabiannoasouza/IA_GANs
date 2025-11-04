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

def traduzirComAPI(caminhoInput, apiToken):
    """
    Traduz uma imagem usando a API do Replicate para o modelo Pix2Pix.
    """
    os.environ["REPLICATE_API_TOKEN"] = apiToken
    import replicate

    print(f"Processando a imagem de entrada: {caminhoInput}")

    # Identificador de um modelo popular de image-to-image (Instruct Pix2Pix)
    modelo = "timothybrooks/instruct-pix2pix:605e9a4f82a7f50a4d86899b823e5a41f8a70905391694f227b8c8d88e7b991b"

    try:
        with open(caminhoInput, "rb") as imagemArquivo:
            print("Enviando imagem para a API do Replicate...")
            # Para este modelo, o input é 'image'. Para o Pix2Pix clássico, poderia ser 'img'.
            # A instrução é opcional, então não vamos fornecer uma.
            outputUrl = replicate.run(
                modelo,
                input={"image": imagemArquivo}
            )
            print(f"Processamento concluído. URL da imagem traduzida: {outputUrl}")

        if outputUrl:
            print("Baixando a imagem resultante...")
            resposta = requests.get(outputUrl, stream=True)
            resposta.raise_for_status()

            pastaOutput = 'resultados_pix2pix'
            if not os.path.exists(pastaOutput):
                os.makedirs(pastaOutput)

            nomeBase = os.path.basename(caminhoInput)
            nomeArquivo, extensaoArquivo = os.path.splitext(nomeBase)
            caminhoOutput = os.path.join(pastaOutput, f"{nomeArquivo}_traduzido{extensaoArquivo}")

            with open(caminhoOutput, 'wb') as f:
                for chunk in resposta.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"Imagem traduzida salva com sucesso em: {caminhoOutput}")
        else:
            print("Erro: A API não retornou uma URL de saída.")

    except Exception as e:
        print(f"Ocorreu um erro durante a comunicação com a API: {e}")


if __name__ == "__main__":
    instalarReplicate()

    parser = argparse.ArgumentParser(description="Traduz imagens usando a API do Pix2Pix no Replicate.")
    parser.add_argument(
        'caminhoInput',
        type=str,
        help="O caminho para a imagem de entrada (esboço/contorno)."
    )

    argumentos = parser.parse_args()

    apiToken = os.getenv("REPLICATE_API_TOKEN")
    if not apiToken:
        print("Erro: O token da API do Replicate não foi encontrado.")
        print("Por favor, defina a variável de ambiente REPLICATE_API_TOKEN.")
        sys.exit(1)

    if os.path.isfile(argumentos.caminhoInput):
        traduzirComAPI(argumentos.caminhoInput, apiToken)
    else:
        print(f"Erro: O arquivo '{argumentos.caminhoInput}' não foi encontrado.")
        sys.exit(1)

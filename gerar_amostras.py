import os
import torch
import cv2
from data.dataset import TextDatasetval
from models.model import TRGAN
from params import *

def gerar_teste():
    # Pasta independente para guardarmos as imagens definitivas
    pasta_saida = 'resultados_teste'
    os.makedirs(pasta_saida, exist_ok=True)
    
    print("A carregar o dataset...")
    dataset_obj = TextDatasetval(num_examples=NUM_EXAMPLES)
    dataloader = torch.utils.data.DataLoader(dataset_obj, batch_size=batch_size, shuffle=True, collate_fn=dataset_obj.collate_fn)
    
    print("A inicializar a rede TRGAN...")
    model = TRGAN()
    
    # Se quiser testar o modelo exato da época 2600, altere para 'model2600.pth'
    caminho_pesos = os.path.join('saved_models', EXP_NAME, 'model100.pth') 
    
    if not os.path.exists(caminho_pesos):
        print(f"Erro: Ficheiro não encontrado em {caminho_pesos}")
        return
        
    print(f"A carregar os pesos da rede ({caminho_pesos})...")
    model.load_state_dict(torch.load(caminho_pesos, map_location=DEVICE))
    model.eval() # Congela os pesos para modo inferência
    
    data = next(iter(dataloader))
    
    print("A renderizar a matriz de caligrafia sintética...")
    with torch.no_grad():
        # Utiliza a função original do modelo para gerar a imagem combinando o estilo e o texto
        page_imagem = model._generate_page(data['simg'].to(DEVICE), data['swids'])
    
    caminho_final = os.path.join(pasta_saida, 'amostra_epoca_100.jpg')
    cv2.imwrite(caminho_final, page_imagem)
    print(f"Sucesso! A imagem foi guardada corretamente em: {caminho_final}")

if __name__ == '__main__':
    gerar_teste()
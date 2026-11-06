import cv2
import os
import glob
import json
import pickle
import re

def ferramenta_de_anotacao(input_dir, progresso_json, output_pickle):
    image_paths = glob.glob(os.path.join(input_dir, '*.jpg'))
    
    # Carrega o progresso anterior para não pedir a mesma palavra duas vezes
    if os.path.exists(progresso_json):
        with open(progresso_json, 'r', encoding='utf-8') as f:
            labels = json.load(f)
    else:
        labels = {}

    print("=== Ferramenta de Anotação de Disgrafia ===")
    print("Digite a palavra que aparece na imagem e aperte ENTER.")
    print("Digite 'pular' se a imagem for lixo/ilegível.")
    print("Digite 'sair' para encerrar e compilar o dataset.\n")

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        
        # Pula as imagens que já foram anotadas
        if filename in labels:
            continue
            
        img = cv2.imread(img_path)
        if img is None: continue

        # Redimensiona apenas para facilitar a sua visualização na tela (2x maior)
        h, w = img.shape[:2]
        img_display = cv2.resize(img, (w * 2, h * 2))
        
        cv2.imshow("Anotacao (Aperte qualquer tecla na janela para focar no terminal)", img_display)
        cv2.waitKey(1) # Força a janela a renderizar
        
        # Pede a entrada no terminal
        # Remove eventuais caracteres unicode inválidos/surrogates lidos do terminal
        texto = input(f"Transcreva a imagem '{filename}': ").strip()
        texto = texto.encode('utf-8', 'ignore').decode('utf-8')
        
        if texto.lower() == 'sair':
            print("\nAnotação pausada. Salvando progresso...")
            break
        elif texto.lower() == 'pular':
            labels[filename] = "IGNORAR"
        else:
            labels[filename] = texto
            
        # Salva no JSON a cada iteração (segurança contra quedas de energia/erros)
        with open(progresso_json, 'w', encoding='utf-8', errors='ignore') as f:
            json.dump(labels, f, ensure_ascii=False, indent=4)

    cv2.destroyAllWindows()
    
    # Compilação para o formato .pickle do Handwriting Transformers
    dataset_dict = {}
    
    for filename, texto in labels.items():
        if texto == "IGNORAR": 
            continue
            
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
            
        # Redimensiona para altura 32px (requisito do HWT)
        h, w = img.shape
        new_w = int((32 / h) * w)
        img_resized = cv2.resize(img, (new_w, 32))
        
        # Extrai o ID do autor (ex: "PD (17)_L0_W1.jpg" -> "17")
        match_autor = re.search(r'\((\d+)\)', filename)
        if not match_autor: continue
        author_id = match_autor.group(1)
        
        if author_id not in dataset_dict:
            dataset_dict[author_id] = {}
            
        sample_id = len(dataset_dict[author_id])
        dataset_dict[author_id][sample_id] = {
            'img': img_resized,
            'text': texto
        }

    # Salva o arquivo final
    with open(output_pickle, 'wb') as f:
        pickle.dump(dataset_dict, f)
        
    print(f"\nDataset compilado com sucesso em '{output_pickle}'.")
    print(f"Total de autores processados: {len(dataset_dict)}")

# Defina os caminhos
pasta_recortes = 'dataset/potencial_dysgraphia_words'
arquivo_progresso = 'dataset/progresso_labels.json'
arquivo_pickle = 'dataset/potencial_dysgraphia.pickle'

ferramenta_de_anotacao(pasta_recortes, arquivo_progresso, arquivo_pickle)
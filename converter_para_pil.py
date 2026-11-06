import pickle
from PIL import Image
import numpy as np

# Caminho do seu arquivo pickle atual
pickle_path = 'dataset/potencial_dysgraphia.pickle'

print("Carregando o dataset...")
with open(pickle_path, 'rb') as f:
    dataset = pickle.load(f)

# Percorre as chaves 'train', 'test' e 'val' que criamos anteriormente
for split in ['train', 'test', 'val']:
    for author_id in dataset[split]:
        for sample_id in dataset[split][author_id]:
            img_data = dataset[split][author_id][sample_id]['img']
            
            # Se a imagem ainda for um array do OpenCV/NumPy, converte para PIL
            if isinstance(img_data, np.ndarray):
                dataset[split][author_id][sample_id]['img'] = Image.fromarray(img_data)

print("Salvando o dataset corrigido...")
with open(pickle_path, 'wb') as f:
    pickle.dump(dataset, f)

print("Conversão para PIL concluída com sucesso! O treinamento já pode ser iniciado.")
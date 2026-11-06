import pickle
from PIL import Image
import numpy as np

pickle_path = 'dataset/potencial_dysgraphia.pickle'

print("A carregar o dataset...")
with open(pickle_path, 'rb') as f:
    dataset = pickle.load(f)

for split in ['train', 'test', 'val']:
    for author_id in dataset[split]:
        for sample_id in dataset[split][author_id]:
            img_pil = dataset[split][author_id][sample_id]['img']
            
            # Converte para array, inverte a polaridade da cor e regressa a PIL
            img_array = np.array(img_pil)
            img_invertida = 255 - img_array
            dataset[split][author_id][sample_id]['img'] = Image.fromarray(img_invertida)

print("A guardar o dataset adaptado ao domínio IAM...")
with open(pickle_path, 'wb') as f:
    pickle.dump(dataset, f)
    
print("Concluído! Pode iniciar o treino.")
import pickle

pickle_path = 'dataset/potencial_dysgraphia.pickle'

print("Carregando o dataset...")
with open(pickle_path, 'rb') as f:
    dataset = pickle.load(f)

# Percorre todo o dataset trocando 'text' por 'label'
for split in ['train', 'test', 'val']:
    for author_id in dataset[split]:
        for sample_id in dataset[split][author_id]:
            if 'text' in dataset[split][author_id][sample_id]:
                # Extrai o valor de 'text' e salva em 'label', apagando a chave antiga
                texto = dataset[split][author_id][sample_id].pop('text')
                dataset[split][author_id][sample_id]['label'] = texto

print("Salvando o dataset corrigido...")
with open(pickle_path, 'wb') as f:
    pickle.dump(dataset, f)

print("Chaves renomeadas com sucesso! O PyTorch agora conseguirá ler as transcrições.")
import pickle

# Caminho do seu arquivo pickle atual
pickle_path = 'dataset/potencial_dysgraphia.pickle'

# Carrega os dados que você já anotou
with open(pickle_path, 'rb') as f:
    dados_originais = pickle.load(f)

# Verifica se já não tem a chave 'train' para evitar empacotar duas vezes
if 'train' not in dados_originais:
    # Reempacota os dados criando as chaves que a rede exige
    novo_dataset = {
        'train': dados_originais,
        # Copiamos para 'test' e 'val' apenas para o validador interno do train.py não quebrar
        'test': dados_originais, 
        'val': dados_originais 
    }

    # Salva o arquivo corrigido por cima
    with open(pickle_path, 'wb') as f:
        pickle.dump(novo_dataset, f)
    
    print("Arquivo .pickle reempacotado com as chaves 'train', 'test' e 'val' com sucesso!")
else:
    print("O arquivo já está no formato correto.")
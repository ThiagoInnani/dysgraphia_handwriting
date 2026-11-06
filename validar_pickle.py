"""
Fase 0: valida o dataset customizado antes de treinar.
Rode na raiz do projeto:  python validar_pickle.py

Formato esperado (igual ao do HWT):
  {split: {id_escritor: {indice: {'img': ..., 'label': str}}}}
Também aceita amostras em lista no lugar do dicionário por índice.
"""
import pickle
import collections

from params import ALPHABET, DATASET_PATHS

IGNORAR = {'PULAR', 'IGNORAR'}


def lista_amostras(am):
    """Aceita dict {indice: amostra} ou lista/tupla de amostras."""
    if isinstance(am, dict):
        return list(am.values())
    return list(am)


def pegar_label(a):
    if isinstance(a, dict):
        for k in ('label', 'text', 'word'):
            if k in a:
                return a[k]
    return None


with open(DATASET_PATHS, 'rb') as f:
    d = pickle.load(f)

print(f'Arquivo: {DATASET_PATHS}')
print(f'Splits encontrados: {list(d.keys())}\n')

assinaturas = {}

for split, escritores in d.items():
    n_img = 0
    n_ignorar = 0
    n_sem_label = 0
    fora = collections.Counter()
    unicos_por_escritor = []
    total_por_escritor = []
    exemplos_ruins = []
    amostra_labels = []
    todos_labels = []

    for wid, am in escritores.items():
        labels = []
        for a in lista_amostras(am):
            l = pegar_label(a)
            if not isinstance(l, str):
                n_sem_label += 1
                continue
            labels.append(l)
        total_por_escritor.append(len(labels))
        unicos_por_escritor.append(len(set(labels)))
        for l in labels:
            n_img += 1
            todos_labels.append(l)
            if len(amostra_labels) < 12:
                amostra_labels.append(l)
            if l.strip().upper() in IGNORAR:
                n_ignorar += 1
                if len(exemplos_ruins) < 5:
                    exemplos_ruins.append((wid, l))
            fora.update(c for c in l if c not in ALPHABET)

    assinaturas[split] = tuple(sorted(todos_labels))
    unicos_por_escritor.sort()
    total_por_escritor.sort()
    mediana = lambda v: v[len(v) // 2] if v else 0

    print(f'=== {split} ===')
    print(f'escritores: {len(escritores)} | imagens com rótulo: {n_img} | sem rótulo em texto: {n_sem_label}')
    print(f'primeiros rótulos: {amostra_labels}')
    print(f'rótulos PULAR/IGNORAR (devem ser removidos): {n_ignorar}  {exemplos_ruins}')
    print(f'caracteres fora do ALPHABET: {dict(fora) if fora else "nenhum"}')
    if total_por_escritor:
        print(f'imagens por escritor (min, mediana, max): '
              f'{total_por_escritor[0]}, {mediana(total_por_escritor)}, {total_por_escritor[-1]}')
        print(f'palavras ÚNICAS por escritor (min, mediana, max): '
              f'{unicos_por_escritor[0]}, {mediana(unicos_por_escritor)}, {unicos_por_escritor[-1]}')
        poucos = sum(1 for u in unicos_por_escritor if u < 5)
        print(f'escritores com menos de 5 palavras únicas: {poucos}')
    print()

nomes = list(assinaturas)
for i in range(len(nomes)):
    for j in range(i + 1, len(nomes)):
        if assinaturas[nomes[i]] == assinaturas[nomes[j]]:
            print(f'AVISO: {nomes[i]} e {nomes[j]} têm exatamente os mesmos rótulos '
                  f'(provavelmente são cópias do mesmo conjunto).')
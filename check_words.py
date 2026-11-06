"""
Confere se todas as palavras de dataset/palavras_alvo.txt cabem no ALPHABET do modelo.
Rode na raiz do projeto:  python checar_palavras.py
"""
import collections
import os

from params import ALPHABET

CAMINHO = 'dataset/palavras_alvo.txt'

if not os.path.isfile(CAMINHO):
    raise SystemExit(f'Arquivo não encontrado: {CAMINHO}')

palavras = open(CAMINHO, encoding='utf-8').read().split()
ruins = [p for p in palavras if any(c not in ALPHABET for c in p)]
chars_ruins = collections.Counter(c for p in ruins for c in p if c not in ALPHABET)

print(f'Total de palavras: {len(palavras)} | únicas: {len(set(palavras))}')
print(f'Palavras com caracteres fora do ALPHABET: {len(ruins)}')
if ruins:
    print('Exemplos:', ruins[:20])
    print('Caracteres problemáticos:', dict(chars_ruins))
else:
    print('Tudo certo: todas as palavras cabem no ALPHABET.')
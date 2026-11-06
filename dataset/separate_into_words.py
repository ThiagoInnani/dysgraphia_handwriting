#!/usr/bin/env python3
"""
Extração de palavras de imagens de caligrafia infantil (texto branco / fundo preto).

Pipeline:
  Etapa 1 - Remoção da pauta horizontal (inpainting morfológico)
  Etapa 2 - Separação de linhas por projeção horizontal
  Etapa 3 - Extração de palavras por componentes conectados

Saída: [nome_original]_L[linha]_W[palavra].jpg  (ordem esquerda -> direita)
"""

import glob
import os

import cv2
import numpy as np

# ----------------------------------------------------------------------------
# Configurações
# ----------------------------------------------------------------------------
INPUT_DIR = "Potential Dysgraphia/"
OUTPUT_DIR = "potencial_dysgraphia_words/"

# Etapa 1
RULING_KERNEL = (60, 1)      # (largura, altura) - longo e fino: isola a pauta
RULING_DILATE = (3, 3)       # engloba a "estática" ao redor da pauta
HEAL_KERNEL = (1, 5)         # cura as hastes verticais cortadas pela remoção

# Etapa 2
PROJ_THRESHOLD_RATIO = 0.04  # fração do pico global: delimita as faixas candidatas
VALLEY_RATIO = 0.5           # vale < 50% do menor pico vizinho => separa as linhas
PROJ_SMOOTH = 5              # suavização da projeção (janela, em pixels)
MIN_LINE_HEIGHT = 15         # descarta faixas menores que isso (ruído)
LINE_PADDING = 2             # folga vertical ao recortar a linha

# Etapa 3
WORD_DILATE_MIN = 14         # largura mínima da dilatação horizontal (kernel Wx1)
WORD_GAP_RATIO = 0.5         # largura = 50% da altura mediana das letras da linha
MIN_WORD_W = 8               # filtros de ruído (medidos no bloco dilatado)
MIN_WORD_H = 8
MIN_WORD_PIXELS = 25         # nº mínimo de pixels de tinta reais no bloco


# ----------------------------------------------------------------------------
# Pré-processamento
# ----------------------------------------------------------------------------
def load_binary(path):
    """Lê a imagem e devolve binária 0/255 com texto BRANCO sobre fundo PRETO."""
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Segurança: se o fundo ficou branco (mais brancos que pretos), inverte.
    if np.count_nonzero(binary) > binary.size / 2:
        binary = cv2.bitwise_not(binary)
    return binary


# ----------------------------------------------------------------------------
# Etapa 1: limpeza da pauta
# ----------------------------------------------------------------------------
def remove_ruling_lines(binary):
    # MORPH_OPEN com kernel largo e fino mantém só estruturas horizontais longas
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, RULING_KERNEL)
    ruling_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

    # Pequena dilatação para cobrir as bordas irregulares da linha
    d_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, RULING_DILATE)
    ruling_mask = cv2.dilate(ruling_mask, d_kernel)

    # Subtrai a máscara: binary AND (NOT mask)
    cleaned = cv2.bitwise_and(binary, cv2.bitwise_not(ruling_mask))

    # Cura as hastes verticais que foram fatiadas ao cruzar a pauta
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, HEAL_KERNEL)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, v_kernel)
    return cleaned


# ----------------------------------------------------------------------------
# Etapa 2: separação de linhas por projeção horizontal
# ----------------------------------------------------------------------------
def split_lines(cleaned):
    """Devolve lista de (y_inicio, y_fim) de cada linha de texto."""
    # Soma de pixels brancos em cada linha da matriz (colapsa o eixo X)
    profile = (cleaned > 0).sum(axis=1).astype(np.float32)
    if PROJ_SMOOTH > 1:
        profile = np.convolve(profile, np.ones(PROJ_SMOOTH) / PROJ_SMOOTH, mode="same")

    # 1) Faixas candidatas: limiar dinâmico relativo ao pico global
    threshold = PROJ_THRESHOLD_RATIO * profile.max()
    active = profile > threshold

    bands, start = [], None
    for y, is_text in enumerate(active):
        if is_text and start is None:
            start = y
        elif not is_text and start is not None:
            bands.append((start, y))
            start = None
    if start is not None:
        bands.append((start, len(active)))

    # 2) Quando linhas colidem (hastes de "y"/"g" tocando a linha de baixo) o
    #    vale nunca chega a zero. Por isso cada faixa é dividida recursivamente
    #    no seu mínimo, se ele for bem mais baixo que os picos vizinhos.
    result = []
    for a, b in bands:
        result.extend(_split_band(profile, a, b))
    return [(a, b) for a, b in result if (b - a) >= MIN_LINE_HEIGHT]


def _split_band(profile, a, b):
    m = MIN_LINE_HEIGHT
    if b - a < 2 * m:
        return [(a, b)]
    seg = profile[a:b]
    # Maior pico à esquerda e à direita de cada posição (acumulado)
    left_peak = np.maximum.accumulate(seg)
    right_peak = np.maximum.accumulate(seg[::-1])[::-1]
    ys = np.arange(m, len(seg) - m)
    # Vale mais "fundo" em relação aos picos vizinhos (e não o menor valor
    # absoluto, que costuma estar na borda da faixa)
    ratio = seg[ys] / np.maximum(np.minimum(left_peak[ys], right_peak[ys]), 1e-6)
    best = int(np.argmin(ratio))
    if ratio[best] < VALLEY_RATIO:
        y = a + int(ys[best])
        return _split_band(profile, a, y) + _split_band(profile, y, b)
    return [(a, b)]


# ----------------------------------------------------------------------------
# Etapa 3: palavras por componentes conectados
# ----------------------------------------------------------------------------
def extract_words(line_img):
    """Devolve recortes (sem dilatação) ordenados da esquerda para a direita."""
    # Largura da dilatação adaptada à escala da escrita (resoluções/tamanhos
    # variados): proporcional à altura mediana dos traços conectados da linha.
    n0, _, st0, _ = cv2.connectedComponentsWithStats(line_img)
    heights = [st0[i][3] for i in range(1, n0) if st0[i][4] >= MIN_WORD_PIXELS]
    med_h = float(np.median(heights)) if heights else 0.0
    k_w = max(WORD_DILATE_MIN, int(round(WORD_GAP_RATIO * med_h)))

    # Dilatação só horizontal (k_w x 1): funde letras da mesma palavra
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, 1))
    fused = cv2.dilate(line_img, k)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(fused, connectivity=8)

    boxes = []
    for i in range(1, n):  # 0 = fundo
        x, y, w, h, _area = stats[i]
        if w < MIN_WORD_W or h < MIN_WORD_H:
            continue

        # Tinta real (sem dilatação) pertencente apenas a este componente
        ink = np.where(labels == i, line_img, 0).astype(np.uint8)
        if np.count_nonzero(ink) < MIN_WORD_PIXELS:
            continue

        # Bounding box exato da tinta (a dilatação alargou o box em ~7px/lado)
        ys, xs = np.nonzero(ink)
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        boxes.append((x0, ink[y0:y1, x0:x1]))

    boxes.sort(key=lambda b: b[0])
    return [crop for _, crop in boxes]


# ----------------------------------------------------------------------------
# Principal
# ----------------------------------------------------------------------------
def process_image(path, out_dir):
    stem = os.path.splitext(os.path.basename(path))[0]
    binary = load_binary(path)
    if binary is None:
        print(f"[ERRO] não foi possível ler {path}")
        return 0

    cleaned = remove_ruling_lines(binary)            # Etapa 1
    bands = split_lines(cleaned)                     # Etapa 2

    total = 0
    for l_idx, (y0, y1) in enumerate(bands, start=1):
        y0p = max(0, y0 - LINE_PADDING)
        y1p = min(cleaned.shape[0], y1 + LINE_PADDING)
        line_img = cleaned[y0p:y1p, :]

        for w_idx, word in enumerate(extract_words(line_img), start=1):  # Etapa 3
            name = f"{stem}_L{l_idx:02d}_W{w_idx:02d}.jpg"
            cv2.imwrite(os.path.join(out_dir, name), word)
            total += 1

    print(f"{stem}: {len(bands)} linha(s), {total} palavra(s)")
    return total


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.jpg")))
    if not files:
        print(f"Nenhum .jpg encontrado em '{INPUT_DIR}'.")
        return
    total = sum(process_image(f, OUTPUT_DIR) for f in files)
    print(f"Concluído: {total} palavras salvas em '{OUTPUT_DIR}'.")


if __name__ == "__main__":
    main()
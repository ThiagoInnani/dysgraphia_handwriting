import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_MODE"] = "offline"
#os.environ["WANDB_API_KEY"] = ""

import time
import cv2
import numpy as np
import torch
import wandb

from data.dataset import TextDataset, TextDatasetval
from models.model import TRGAN
from params import *

# ---------------------------------------------------------------------------
# Parâmetros opcionais (se não existirem no params.py, usa estes valores)
# ---------------------------------------------------------------------------
# Arquivo de pesos inicial (iam_model.pth ou o modelo da fase 1). Nunca é sobrescrito.
PRETRAINED_PATH = globals().get('PRETRAINED_PATH', 'modelos_originais/iam_model.pth')
# Número de épocas iniciais em que o gerador NÃO é treinado (só D, OCR e W)
FREEZE_G_EPOCHS = globals().get('FREEZE_G_EPOCHS', 0)
# A cada quantas épocas salvar as imagens de amostra
SAVE_IMG_FREQ = globals().get('SAVE_IMG_FREQ', 10)


def carregar_filtrado(model, caminho):
    """Carrega pesos aceitando (a) arquivo só com o gerador ou (b) modelo completo.
    Tensores ausentes ou com shape diferente (ex.: netW com NUM_WRITERS distinto) são ignorados."""
    pesos = torch.load(caminho, map_location=DEVICE)
    if 'query_embed.weight' in pesos:  # arquivo só com o gerador (iam_model.pth)
        pesos = {'netG.' + k: v for k, v in pesos.items()}
    atual = model.state_dict()
    ok = {k: v for k, v in pesos.items() if k in atual and atual[k].shape == v.shape}
    ignorados = [k for k in pesos if k not in ok]
    ausentes = [k for k in atual if k not in ok]
    model.load_state_dict(ok, strict=False)
    print(f'[carregar] {caminho}')
    print(f'[carregar] {len(ok)} tensores carregados | {len(ignorados)} ignorados (shape/nome) '
          f'| {len(ausentes)} do modelo ficaram com inicialização aleatória')
    if ignorados:
        print(f'[carregar] exemplos ignorados: {ignorados[:5]}')
    if ausentes:
        print(f'[carregar] exemplos aleatórios: {ausentes[:5]}')


def para_uint8(pagina):
    """_generate_page devolve valores em [0, 1] (branco = 1)."""
    return (np.clip(pagina, 0, 1) * 255).astype(np.uint8)


def main():

    wandb.init(project="hwt-final", name=EXP_NAME)

    init_project()

    TextDatasetObj = TextDataset(num_examples=NUM_EXAMPLES)
    dataset = torch.utils.data.DataLoader(
        TextDatasetObj,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True, drop_last=True,
        collate_fn=TextDatasetObj.collate_fn)

    TextDatasetObjval = TextDatasetval(num_examples=NUM_EXAMPLES)
    datasetval = torch.utils.data.DataLoader(
        TextDatasetObjval,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True, drop_last=True,
        collate_fn=TextDatasetObjval.collate_fn)

    model = TRGAN()

    MODEL_PATH = os.path.join('saved_models', EXP_NAME)
    os.makedirs(MODEL_PATH, exist_ok=True)
    CKPT = os.path.join(MODEL_PATH, 'model.pth')

    # Prioridade: checkpoint deste experimento (retomar) > pesos iniciais > do zero
    if RESUME and os.path.isfile(CKPT):
        carregar_filtrado(model, CKPT)
    elif os.path.isfile(PRETRAINED_PATH):
        carregar_filtrado(model, PRETRAINED_PATH)
    else:
        print('ATENÇÃO: nenhum peso carregado, treinando do zero!')

    print(f'[treino] G congelado nas primeiras {FREEZE_G_EPOCHS} épocas')

    for epoch in range(EPOCHS):

        start_time = time.time()
        treinar_G = epoch >= FREEZE_G_EPOCHS

        for i, data in enumerate(dataset):

            if treinar_G and (i % NUM_CRITIC_GOCR_TRAIN) == 0:

                model._set_input(data)
                model.optimize_G_only()
                model.optimize_G_step()

            if (i % NUM_CRITIC_DOCR_TRAIN) == 0:

                model._set_input(data)
                model.optimize_D_OCR()
                model.optimize_D_OCR_step()

            if treinar_G and (i % NUM_CRITIC_GWL_TRAIN) == 0:

                model._set_input(data)
                model.optimize_G_WL()
                model.optimize_G_step()

            if (i % NUM_CRITIC_DWL_TRAIN) == 0:

                model._set_input(data)
                model.optimize_D_WL()
                model.optimize_D_WL_step()

        end_time = time.time()
        data_val = next(iter(datasetval))
        losses = model.get_current_losses()
        page = model._generate_page(model.sdata, model.input['swids'])
        page_val = model._generate_page(data_val['simg'].to(DEVICE), data_val['swids'])

        wandb.log({'loss-G': losses['G'],
                   'loss-D': losses['D'],
                   'loss-Dfake': losses['Dfake'],
                   'loss-Dreal': losses['Dreal'],
                   'loss-OCR_fake': losses['OCR_fake'],
                   'loss-OCR_real': losses['OCR_real'],
                   'loss-w_fake': losses['w_fake'],
                   'loss-w_real': losses['w_real'],
                   'G_treinando': int(treinar_G),
                   'epoch': epoch,
                   'timeperepoch': end_time - start_time,
                   })

        wandb.log({"result": [wandb.Image(page, caption="page"),
                              wandb.Image(page_val, caption="page_val")]})

        print({'EPOCH': epoch, 'G_treinando': treinar_G,
               'TIME': end_time - start_time, 'LOSSES': losses})

        # --- Salva imagens localmente (page está em [0, 1]) ---
        if epoch % SAVE_IMG_FREQ == 0:
            cv2.imwrite(os.path.join('saved_images', EXP_NAME, 'Fake', f'treino_epoca_{epoch}.jpg'),
                        para_uint8(page))
            cv2.imwrite(os.path.join('saved_images', EXP_NAME, 'Fake', f'val_epoca_{epoch}.jpg'),
                        para_uint8(page_val))

        # --- Checkpoints (nunca tocam em PRETRAINED_PATH) ---
        if epoch % SAVE_MODEL == 0:
            torch.save(model.state_dict(), CKPT)
        if epoch % SAVE_MODEL_HISTORY == 0:
            torch.save(model.state_dict(), os.path.join(MODEL_PATH, f'model{epoch}.pth'))


if __name__ == "__main__":

    main()

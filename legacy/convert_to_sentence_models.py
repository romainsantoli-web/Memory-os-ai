#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convertir des checkpoints Transformers (style 'bert_base_uncased', 'roberta_base', etc.)
en 'vrais' modèles SentenceTransformers (avec pooling).
Pour chacun, on crée un dossier *_sentence_model.
"""

import os
from sentence_transformers import SentenceTransformer, models

# Dossier de base où se trouvent les sous-dossiers (ex: 'all-MiniLM-L6-v2', 'bert_base_uncased')
BASE_DIR = "/workspace/memory_os_ai/emb"

# La liste des sous-dossiers à convertir
# - Si "all-MiniLM-L6-v2" est déjà un SentenceTransformers officiel,
#   vous pouvez l'omettre. Sinon, vous pouvez le re-convertir par uniformité.
MODELS_TO_CONVERT = [
     "all-MiniLM-L6-v2",
#    "all-MiniLM-L12-v2",
#    "clip-ViT-B-32",
#    "bert_base_uncased",
#    "roberta_base"
    # Ajoutez ou retirez selon vos besoins
]

def main():
    for model_name in MODELS_TO_CONVERT:
        in_path  = os.path.join(BASE_DIR, model_name)
        out_path = os.path.join(BASE_DIR, model_name + "_sentence_model")

        if not os.path.isdir(in_path):
            print(f"[SKIP] Le dossier n'existe pas: {in_path}")
            continue

        # 1) Charger en tant que Transformer "brut"
        print(f"[CONVERT] Chargement HF checkpoint => {in_path}")
        word_embedding_model = models.Transformer(in_path)

        # 2) Créer la couche de pooling
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False
        )

        # 3) Construire le SentenceTransformer final
        st_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

        # 4) Sauvegarder dans out_path
        st_model.save(out_path)
        print(f"[OK] Converti => {out_path}")

if __name__=="__main__":
    main()




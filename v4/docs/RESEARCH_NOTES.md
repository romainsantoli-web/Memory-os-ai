# État de l'art mémoire d'agents LLM — notes de recherche (2026-09-12)

Légende : [PAPER] article arXiv/conf ; [VENDOR] blog éditeur (auto-déclaré) ; [INDEP] évaluation tierce.

## 1. Benchmarks et scores (à prendre avec recul)
- LoCoMo (Snap, arXiv:2402.17753) : 10 conversations, ~1 986 QA, catégories single-hop / multi-hop / temporal / open-domain / adversarial. Audit Penfield [INDEP] : 6,4 % du corrigé faux, juge gpt-4o-mini accepte 63 % de réponses volontairement fausses → deltas < 6 pts non interprétables. Full-context ≈ 73 % (Mem0 paper) > la plupart des systèmes de mémoire. Cat. 5 sans vérité terrain.
- LongMemEval (ICLR 2025, arXiv:2410.10813) : 500 questions, ~115 k tokens/instance, teste extraction, multi-session, temporel, mises à jour, abstention. Variantes : oracle (15 Mo, sessions-réponses seules, drapeau `has_answer` par tour), S (278 Mo), M (2,7 Go). LongMemEval-V2 (2026) = trajectoires d'agents, métrique LAFS.
- BEAM (ICLR 2026) : 100 conversations 128 k → 10 M tokens, 2 000 questions, 10 catégories. MemoryArena (arXiv:2602.16313) : systèmes saturant LoCoMo s'effondrent sur tâches agentiques.
- Scores 2026 [VENDOR, non comparables] : Mem0 92,5 LoCoMo / 94,4 LongMemEval ; Zep 94,7 (DMR maison ≠ LoCoMo) ; Supermemory 85,4 LongMemEval, 95 % Recall@15 ; MemOS 88,8 / 89,2 ; Hindsight 89,6 LoCoMo, 83,6 LongMemEval avec gpt-oss-20b [PAPER 2512.12818]. Ré-exécutions indépendantes de Mem0 sur LongMemEval : 49 % (Vectorize) et 73,8 % (Maximem) vs 94,4 déclaré. AMA-Bench [INDEP, 2602.22769] : Mem0 dernier derrière MemGPT et A-Mem.
- Conséquence pour v4 : évaluer le **rappel de l'évidence** (Recall@k, MRR, nDCG, model-free), pas un juge LLM.

## 2. Mécanismes qui comptent (convergence des sources)
- **Mem0 v2.0 OSS (16 avr. 2026)** : extraction ADD-only en une passe (plus d'UPDATE/DELETE), les deux versions d'un fait coexistent avec contexte temporel, faits de l'agent = première classe, entity linking (noms propres, termes cités, composés) remplace le graphe (retiré de l'OSS), retrieval multi-signal sémantique + BM25 + entités fusionnés, seuil 0,1, rerank optionnel (+200-400 ms).
- **Zep/Graphiti (arXiv:2501.13956)** : arêtes bi-temporelles t_valid/t_invalid + t'_created/t'_expired ; contradiction → l'ancienne arête reçoit t_invalid = t_valid de la nouvelle, jamais supprimée. « Don't Ask the LLM to Track Freshness » (arXiv:2606.01435) et TOKI (2606.06240) : résolution **déterministe** des conflits préférable à l'op choisie par LLM (Mem0 supprime silencieusement des faits encore utiles).
- **AutoMEM (arXiv:2606.04315)** : la généralité cross-scénario vient de **l'agence de l'agent** sur sa mémoire (outils d'écriture/lecture) plus que du pipeline passif → v4 confie l'extraction des faits au LLM client via `memory_chat_save(facts=…)` et expose des outils riches.
- **Letta** : blocs cœur toujours en contexte à limite de caractères ; sleep-time compute ; Context Repositories (fév. 2026) = mémoire versionnée façon git.
- **HINDSIGHT** : quatre réseaux (faits monde, expériences agent, résumés d'entités, croyances évolutives) ; retain / recall / reflect.
- **LightMem (ICLR 2026)** : compression sensorielle + segmentation par sujet + consolidation hors-ligne ; 73,8 LoCoMo = full-text avec 38× moins de tokens. **SimpleMem (2601.02553)** : unités mémoire multi-vues + planification de portée de recherche selon l'intention ; ~530 tokens/requête.
- **Generative Agents / MemoryBank** : score = pertinence + récence (décroissance exp. depuis le dernier **accès**) + importance ; Ebbinghaus avec renforcement à l'accès.
- **ReasoningBank (ICLR 2026) / Memp (ACL 2026)** : mémoire procédurale (titre, description, contenu) distillée des succès **et** des échecs ; +8,3 pts WebArena.
- **Contextual retrieval (Anthropic)** : préfixer chaque chunk d'un en-tête contextuel ; −49 % d'échecs de rappel avec BM25+dense, −67 % avec rerank.

## 3. Pile de retrieval
- Fusion : RRF k=60 (Cormack 2009) robuste mais perd l'amplitude des scores ; Bruch et al. (2210.11934) : combinaison convexe pondérée > RRF ; DBSF (Qdrant, août 2026) > RRF sur 3/5 jeux (+0,016-0,038 nDCG) ; Weaviate relativeScoreFusion par défaut (+6 % rappel). Le gros gain vient du **reranker après fusion** (T2-RAGBench : R@5 0,695 → 0,816). → v4 implémente `rrf | dbsf | cc` et mesure.
- FAISS : < 1 M vecteurs → Flat exact (0 s de build, rappel 1,0) ; HNSW +155 % mémoire pour 82 % R@1 (wiki FAISS SIFT1M). → `IndexFlatIP` + `IDMap2`.
- Embeddings ≤ 200 Mo (contrainte disque) : granite-embedding-97m-multilingual-r2 (195 Mo fp16, MTEB multilingue 60,3, mai 2026) ; granite-embedding-small-english-r2 (95 Mo) ; bge-small-en-v1.5 (134 Mo) ; baseline all-MiniLM-L6-v2 (87 Mo, en cache). Trop gros : Qwen3-Embedding-0.6B (1,2 Go, MMTEB 64,3), EmbeddingGemma (1,2 Go, gated), bge-m3 (2,2 Go).
- Rerankers légers : ms-marco-MiniLM-L6-v2 (91 Mo), Ettin 17m/32m (tomaarsen/ms-marco-ettin-*-reranker, plus rapide et meilleur que MiniLM sur NanoBEIR, anglais) ; multilingue mmarco-mMiniLMv2-L12 (471 Mo, plus tard).

## 4. Sources principales
arXiv 2402.17753, 2410.10813, 2504.19413, 2501.13956, 2512.12818, 2606.04315, 2606.01435, 2606.06240, 2510.18866, 2601.02553, 2509.25140, 2508.06433, 2602.22769, 2602.16313, 2210.11934, 2604.01733 ; docs.mem0.ai/migration/platform-v2-to-v3 ; blog.getzep.com (critique LoCoMo) ; dev.to/penfieldlabs (audit LoCoMo) ; qdrant.tech/articles/how-to-tune-hybrid-search ; github.com/facebookresearch/faiss/wiki ; huggingface.co (tailles de modèles vérifiées via API le 2026-09-12).

## 5. Compléments (rapport final des agents de recherche)
- **Cible chiffrée model-free** : dans le harnais indépendant « Are We Ready For An Agent-Native Memory System? » (arXiv:2606.24775, 12 systèmes, 11 jeux), le meilleur rappel d'évidence LoCoMo est A-Mem : **Recall@10 = 85,9 %, Recall@5 = 69,5 %** ; LightMem = frontière d'efficacité (3,67 s/requête) ; Zep le plus fiable sur les mises à jour de connaissances ; « User-Only Raw » (24,2 EM) bat « Summary » (8,5) → **le texte brut des épisodes doit toujours rester indexé, les faits/résumés sont additifs, jamais substitutifs**.
- AMA-Bench (ICML 2026) : Mem0 dernier (0,21) derrière MemoryBank (0,34), MemGPT (0,33), A-Mem (0,32) ; HippoRAG2 meilleure baseline RAG (0,45). « Total Recall at What Cost? » (2608.11879) : coût de service des mémoires sous-estimé de 18-69 %.
- Anthropic memory tool (`memory_20250818`) : mémoire = fichiers `/memories` édités par le modèle (view/create/str_replace/insert/delete/rename) ; +39 % de succès agentique avec context editing [vendor]. Même philosophie « agent-native » qu'AutoMEM → les blocs cœur v4 sont éditables par le client.
- Late chunking (Jina) : +2-3 % nDCG seulement, inutile sur des tours de dialogue courts ; en-tête contextuel suffit. Quantification int8/binaire : hors sujet à notre échelle (< 100 k vecteurs).

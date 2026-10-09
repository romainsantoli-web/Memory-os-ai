# Memory OS AI v4 — Plan « approcher le SOTA » (2026-09-12)

Règles de travail (non négociables)
- Aucun fichier existant du dépôt n'est modifié ni supprimé. Tout vit dans `v4/`.
- Exécution avec l'interpréteur existant `../.venv/bin/python` + `PYTHONPATH=v4/src` (disque à 99 %, 2,3 Go libres → pas de second venv, zéro dépendance lourde nouvelle).
- Bookkeeping compétition : chaque expérience = fichier versionné `bench/results/<approche>_vN.json` + ligne dans `RESULTS.md`. Rien n'est écrasé.
- Cycle : Planifier → Organiser → Structurer → Exécuter → Vérifier (adversarial, contexte vierge).

## 1. Diagnostic de l'existant (v3.2.0, `src/memory_os_ai/`)
| Composant | v3 | Écart au SOTA 2026 |
|---|---|---|
| Embedding | all-MiniLM-L6-v2 (384d, anglais) | modèles multilingues 2026 ≤ 200 Mo (granite-97m-r2 : MTEB multilingue 60,3) |
| Découpage | 128 **caractères**, chevauchement 28 | phrases/tours complets + en-tête contextuel (Anthropic contextual retrieval) |
| Index | FAISS FlatL2, seuil L2 0,8 | hybride dense + BM25 + entités, fusion RRF, reranker cross-encoder optionnel (Mem0 v3, MemOS FTS5+vecteur) |
| Modèle mémoire | segments de texte plats | items typés (épisode / fait / résumé / procédure / bloc), provenance, entités, importance, accès |
| Temporel | aucun | bi-temporel Zep (event_time, valid_from, valid_until, invalidated_by), ADD-only + supersession (Mem0 v3) |
| Extraction | aucune (texte brut) | faits atomiques extraits par le LLM client au moment du save (AutoMEM : l'agent contrôle sa mémoire) |
| Consolidation | dedup cosine < 0,05 | sleep-time : dedup, supersession, décroissance Ebbinghaus + renforcement, résumés hiérarchiques (LightMem, Letta) |
| Contexte de session | 6 requêtes fixes | blocs cœur toujours chargés (Letta) + récents + faits importants + focus, budget tokens |
| Évaluation | aucune | LoCoMo + LongMemEval, Recall@k / MRR / nDCG **sans LLM juge** (métriques de rappel, reproductibles) |

## 2. Contraintes matérielles
- M4 16 Go, **2,3 Go de disque libre** : modèles ≤ 200 Mo chacun, budget total téléchargements ≤ 700 Mo (jeux de données inclus). Aucun LLM local (aucun GGUF exploitable, Ollama vide). Aucune clé API dans l'environnement → l'évaluation est model-free (rappel), la génération de réponses reste au LLM client (Claude via MCP).
- MPS disponible (torch 2.10) : encodage batch sur GPU.

## 3. Techniques retenues (classées gain attendu × facilité locale)
1. **Découpage par tours/phrases + en-tête contextuel** (projet, date, locuteur) — gratuit, gros gain vs 128 caractères.
2. **Hybride dense + BM25 + entités**, fusion configurable `rrf | dbsf | cc` (Bruch 2022, Qdrant 2026 : les fusions pondérées battent RRF de 2-6 %) — gratuit ; Mem0 v2 (avr. 2026) attribue une part de ses +20 pts au multi-signal.
3. **Faits atomiques ADD-only fournis par le LLM client** dans `memory_chat_save(facts=…)`, faits agent = première classe (Mem0 v3) ; dédup serveur.
4. **Bi-temporel + supersession déterministe** (pas d'op choisie par LLM : arXiv 2606.01435) : conflit détecté (mêmes entités, forte similarité, contenu différent) → l'ancien fait reçoit `valid_until`, jamais supprimé (Zep/Graphiti) ; requêtes datées filtrent sur la validité.
5. **Entités légères** (noms propres, identifiants de code, chiffres/dates, termes cités) → index inversé, boost de rappel (Mem0 v3 entity linking, HINDSIGHT entity summaries).
6. **Score = fusion × f(récence, importance, accès)** (Generative Agents, MemoryBank) avec décroissance exponentielle paramétrée et renforcement à l'accès.
7. **Reranker cross-encoder optionnel** sur top-50 (ms-marco-MiniLM-L6 91 Mo ou Ettin-17m) — mesuré, activé seulement si gain net.
8. **Blocs cœur** (user_profile, project_state, working_rules) à limite de caractères, versionnés, toujours injectés dans le brief (Letta memory blocks).
9. **Consolidation sleep-time** `memory_consolidate` : dedup, balayage supersession, décroissance, promotion des faits fréquemment rappelés, propositions de résumés que le client rédige (pas de LLM local).
10. **MMR** pour la diversité du contexte retourné + budget tokens (SimpleMem : ~500 tokens/requête suffisent).
11. **Modèle d'embedding multilingue 2026** : granite-embedding-97m-multilingual-r2 (195 Mo) vs MiniLM baseline — tranché par le banc.
12. **Mémoire procédurale** (`kind=procedure`, titre/description/contenu, leçons des succès et des échecs — ReasoningBank/Memp) rappelée dans le brief.
13. **Migration v3 → v4** du journal `_conversation_log.jsonl` (60 enregistrements) sans toucher v3.

Écartés (pour l'instant) : graphe de connaissances complet (Graphiti/HippoRAG PPR : gain surtout multi-hop, coût LLM d'extraction de triplets), extraction serveur par LLM (pas de modèle local ; backend OpenAI-compatible prévu mais optionnel), RL de gestion mémoire (Memory-R1/MemReader).

## 4. Architecture cible (`v4/src/memory_os_v4/`)
Voir `docs/ARCHITECTURE.md` (spec d'interface, modèle de données, formats sur disque).

## 5. Phases, critères de succès, points de validation
| Phase | Livrable | Critère de succès | Validation | État (2026-09-13) |
|---|---|---|---|---|
| P0 | PLAN.md, ARCHITECTURE.md, RESULTS.md, squelette | fichiers présents | relecture | ✅ fait |
| P1 | noyau : store, text, bm25, embed, vector_index, retrieval, memory (façade) + tests | `pytest v4/tests` vert ; API conforme à la spec | tests + revue | ✅ fait (17 modules, 4 174 lignes avec serveur) |
| P2 | banc : download LoCoMo (2,8 Mo) + LongMemEval oracle (15 Mo), adaptateurs, métriques, baseline « v3-équivalent » | 1re ligne RESULTS.md (baseline) + dense + bm25 + hybride | reproductible (`run_bench.py --config`) | ✅ fait (LoCoMo 1 536 q, LME oracle 500 q, 37+ lignes ledger) |
| P3 | sémantique : entités, temporel/supersession, scoring récence/importance, blocs, consolidation, extraction (client/heuristique/LLM optionnel) | tests ; gain ≥ 0 sur LoCoMo temporal/multi-hop | agent adverse : fuites, off-by-one, non-régression | ✅ fait (revue adverse Opus : 24 findings, tous corrigés) |
| P4 | serveur MCP v4 (noms d'outils v3 conservés + nouveaux), instructions, migration v3 | smoke test client MCP stdio ; brief < 4 k tokens | test bout en bout | ✅ fait (16 outils, selfcheck PASS, e2e stdio) |
| P5 | itération large : ablations (fusion, chunking, modèles, reranker, poids) | config gagnante documentée ; ≥ 8 lignes RESULTS.md | ledger complet | ✅ fait (cc 1:1,5 + stemming promu ; R@10 LoCoMo 0,175 v3 exact → 0,527) |
| P6 | README v4, `.mcp.json` d'exemple (dans v4/), sauvegarde mémoire | doc à jour | — | ✅ fait (README résultats + bascule v3→v4, mémoire sauvegardée) |

Rollback : supprimer `v4/` suffit ; v3 reste intact (config MCP v3 inchangée).

## 6. Budget
- Temps : session unique ; P1-P2 en parallèle (agents Sonnet), P3 puis P4, P5 en fin.
- Disque : ≤ 700 Mo (modèles 195 + 91 + 134 Mo, données 18 Mo, index/tests < 50 Mo). LongMemEval-S (278 Mo) seulement si ≥ 1,5 Go restent.
- Métrique pilote : Recall@10 turn-level LoCoMo (moyenne par catégorie) ; secondaire : MRR, nDCG@10, latence p50 par requête, tokens de contexte.

## P9 — Dépasser A-Mem (cible LoCoMo macro R@10 > 0,86 ; état 0,662) — 2026-09-13
Diagnostic : single-hop 0,90 et temporel 0,83 sont proches du plafond ; l'écart est multi-hop (0,50, ≥ 3 tours d'évidence par question) et open-domain (0,43, inférence). Un retrieval mono-requête ne peut pas les résoudre : il faut (a) plusieurs requêtes par question, (b) des liens entre faits, (c) un reranking plus fort.
| Étape | Mécanisme (référence) | Où | Gain attendu |
|---|---|---|---|
| A | **Planification de requête par LLM** (SimpleMem intent-aware planning, RAG-Fusion, HyDE) : Sonnet 5 réécrit chaque question en 2-4 sous-requêtes dans le vocabulaire des dialogues + 1-2 « faits hypothétiques » ; retrieval par sous-requête, fusion RRF ; hors ligne sur les 1 536 questions (`bench/data/enrich/locomo_queries.jsonl`) | banc | multi-hop +0,10 à +0,20 |
| B | **Liens entre faits** (A-Mem Zettelkasten) : faits partageant ≥ 1 entité rare ou la même session sont liés ; expansion 1 saut depuis les faits rappelés, score décroissant | banc puis src (flag) | multi-hop +0,03 à +0,08 |
| C | **Reranker plus fort** : Ettin-68m (vs 17m) si ≤ 200 Mo | cache HF | +0,01 à +0,03 |
| D | **Reranking listwise par LLM** sur le top-20 (Sonnet 5, hors ligne) si A-C n'atteignent pas 0,86 ; en production = le modèle client lit déjà le contexte | banc | +0,05 à +0,10 |
Règles : séquentiel, nice 15, MPS plafonné ; `src/` intact sauf portage final des gagnants derrière des flags ; chaque run dans le ledger ; promotion seulement si gain > 0,005 et pas de régression single-hop/temporel.

# Memory OS v5 — plan pour devenir la référence (2026-09-19)

Point de départ mesuré : LoCoMo R@10 0,175 (v3) → 0,538 brut local → 0,815 avec LLM hors ligne ; 493 tests ;
20-30 ms par recherche ; tout local, 365 Mo de modèles. Référence indépendante : A-Mem 0,86.
Contraintes : MacBook 16 Go (runs séquentiels, nice 15, garde-fou swap), disque > 3 Go, budget tokens explicite
par étape, aucun fichier v3 modifié, chaque expérience versionnée dans RESULTS.md, revue adverse à chaque phase.

## 1. Lacunes des concurrents (sources : docs/RESEARCH_NOTES.md)
| # | Lacune constatée | Qui | Réponse v5 |
|---|---|---|---|
| L1 | Scores non reproductibles (Mem0 94 % annoncé, 49-74 % rejoué), pas de coût ni de latence publiés | tous les éditeurs | harnais public model-free + coût + latence + oubli + conflits |
| L2 | Les résumés détruisent l'évidence (Raw 24,2 vs Summary 8,5 EM) | LightMem, MemoryBank, MemAgent | brut toujours indexé, faits additifs (acquis) |
| L3 | Suppressions décidées par LLM, pertes silencieuses | Mem0 < 2.0, LangMem | bi-temporel déterministe (acquis) + journal d'audit |
| L4 | 100-155 s par requête pour les graphes | Zep, Cognee | 20-30 ms local (acquis), LLM seulement à l'écriture |
| L5 | Questions d'agrégation et d'inférence (mentions incidentes) | tous, y compris A-Mem en multi-hop | facettes typées + implications (idée 8) |
| L6 | Arithmétique de dates faite par le LLM, fragile | tous (LongMemEval temporal 0,5) | outil de chronologie déterministe |
| L7 | Aucune abstention calibrée : la mémoire ne dit jamais « je ne sais pas » | tous | score de couverture + seuil d'abstention |
| L8 | Mémoire procédurale absente ou sans suivi d'efficacité | tous sauf prototypes ReasoningBank/Memp | procédures + savoir négatif avec issue mesurée |
| L9 | Sécurité : empoisonnement, injection par souvenir, pas de provenance vérifiable | tous (survey 2604.16548) | niveaux de confiance, quarantaine, provenance signée |
| L10 | Pas de vraie suppression ni d'export portable ; mémoire opaque | ChatGPT (« dreaming », piste d'audit limitée), plateformes cloud | suppression physique vérifiable, export ouvert, tout inspectable |
| L11 | Multi-agents : pas d'espaces de noms, collisions | tous (vécu ici : sous-agents écrasant leurs fichiers) | portées, verrous, tableau partagé |
| L12 | Faits sur le code non liés à l'état du dépôt | tous sauf Letta (git) | validité ancrée sur le hash de fichier/commit |
| L13 | Évaluations quasi toutes en anglais | tous | banc FR/EN |

## 2. Ce dont un agent a besoin d'une mémoire (vu depuis Claude ; valable pour ChatGPT/Codex via MCP)
B1 Budget : contexte le plus dense possible sous N tokens, divulgation progressive (résumé → détail par id).
B2 Savoir quand se taire : couverture/confiance renvoyée avec chaque recherche ; pas de réponse plutôt qu'une invention.
B3 Conflits et fraîcheur visibles : deux souvenirs contradictoires sont signalés, avec dates et validité.
B4 Savoir négatif : ce qui a été tenté et réfuté, avec la preuve (éviter de refaire un labo mort).
B5 Reprise de travail : état de tâche, boucles ouvertes, décisions avec leur pourquoi ; restauration après compaction du contexte.
B6 Rappel proactif : la mémoire pousse le pertinent au bon moment (hook de prompt) au lieu d'attendre une requête.
B7 Apprentissage par l'usage : l'agent dit quels souvenirs ont servi ; le classement s'améliore en ligne.
B8 Les souvenirs sont des données, jamais des instructions : isolation stricte du contenu non fiable.
B9 Portabilité entre modèles et clients : même mémoire pour Claude, ChatGPT, Codex, Gemini ; format d'export neutre.
B10 Contrôle utilisateur total : tout lire, corriger, supprimer réellement, auditer, chiffrer, rester hors cloud.

## 3. Phases, livrables, critères de succès
| Phase | Durée | Livrables | Critère de succès mesurable | Budget |
|---|---|---|---|---|
| V5.0 Rappel | 1-2 sem. | Facettes (vocabulaire fermé ~40 prédicats) + implications à l'écriture ; facettes dans `memory_search` ; outil `memory_timeline` avec arithmétique de dates déterministe ; score de couverture + abstention | LoCoMo plafond@50 multi-hop ≥ 0,90 ; macro R@10 ≥ 0,80 sans reranking LLM ; LongMemEval-S temporal ≥ BM25 ; abstention : précision ≥ 0,9 sur questions sans évidence | ~2 M tokens Sonnet, 1 passe banc |
| V5.1 Ergonomie agent | 1-2 sem. | `memory_get_context` à budget avec divulgation progressive ; signalement conflits/fraîcheur ; `kind=refuted` (savoir négatif) ; `memory_feedback(used, useless)` + classement par usage ; hook de rappel proactif Claude Code ; restauration post-compaction | tokens de contexte −30 % à rappel égal ; gain R@10 mesuré après 200 feedbacks simulés ; test bout en bout du hook | code seul |
| V5.2 Confiance et gouvernance | 1 sem. | niveaux de confiance par source ; quarantaine + détection d'injection ; journal d'audit append-only ; suppression physique vérifiable ; chiffrement au repos ; export/import format neutre | 0 instruction exécutée depuis un souvenir dans la suite d'attaques ; suppression prouvée (aucune trace dans index, journal, cache) | code seul |
| V5.3 Multi-agents et code | 1 sem. | espaces de noms/portées/verrous ; tableau partagé entre sous-agents ; validité ancrée sur hash git ; documents de passation | 0 collision sur 20 sous-agents concurrents ; faits code invalidés automatiquement après modification du fichier | code seul |
| V5.4 Leadership d'évaluation | 1-2 sem. | harnais public (rappel, coût, latence, oubli, conflits) ; LongMemEval-S complet, BEAM-100K, banc FR/EN ; passage dans le harnais indépendant (2606.24775) ; v4→PyPI 4.0 ; ponts ChatGPT/Codex/Gemini | chiffres reproductibles par un tiers en une commande ; ≥ A-Mem dans le harnais indépendant ou écart expliqué | ~3 M tokens |

## 4. Ordre et règles
V5.0 d'abord (seul levier qui élève le plafond), V5.1 ensuite (valeur d'usage quotidienne la plus forte), puis V5.2,
V5.3, V5.4. Promotion d'une fonctionnalité seulement si gain > bruit (0,005) sur deux bancs ; tout le reste reste opt-in.
Risques : surapprentissage à LoCoMo (parade : second banc obligatoire), coût tokens (parade : budget par phase, arrêt si
le test de faisabilité échoue), machine (parade : garde-fou existant), dérive de périmètre (parade : un critère chiffré par livrable).


## État de V5.0 au 2026-09-19 (mesuré, pas prévu)
| Critère annoncé | Résultat | Statut |
|---|---|---|
| plafond@50 multi-hop ≥ 0,90 | 0,902 avec facettes exactes ; signal réel à taille égale (+0,023, IC [+0,009 ; +0,038], 10 conversations sur 10) | atteint |
| macro R@10 ≥ 0,80 sans reranking LLM | 0,757 (facettes poids 2,0, sans reranker) contre 0,721 ; +0,036 | NON atteint |
| implications utiles à l'open-domain | résultat nul (−0,002 macro) | réfuté |
| LongMemEval temporal ≥ BM25 (chronologie déterministe) | non commencé | ouvert |
| abstention calibrée (score de couverture) | non commencé | ouvert |
| facettes + reranker | run arrêté par le garde-fou mémoire (swap 6,5 Go) | non mesuré |
Rien n'est promu en défaut : un seul banc porte des facettes. Prochaine étape logique si l'on poursuit : porter le
canal facettes dans `src/` derrière un drapeau (facettes fournies par le LLM client à la sauvegarde, comme les
mots-clés) et annoter LongMemEval pour le second banc. Ne pas relancer le run reranké sans budget mémoire dédié.

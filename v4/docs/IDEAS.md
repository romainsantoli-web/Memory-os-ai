# Idées inter-disciplines pour élever le plafond multi-hop / open-domain (2026-09-14)

Diagnostic : après enrichissement, planification et reranking LLM, v4 est à 0,815 de R@10 sur LoCoMo ;
single-hop (0,97) et temporel (0,93) sont au plafond, l'écart est dans le multi-hop (0,77, plafond 0,86
avec 50 candidats) et l'open-domain (0,59, plafond 0,69). Les mécanismes « lien » testés (voisins,
entités, liens entre faits) ont tous échoué : ils ajoutent des candidats sans garantir la **couverture**
des différentes pièces d'évidence. Les idées ci-dessous viennent d'autres champs et attaquent la
couverture, pas la similarité.

| # | Champ d'origine | Analogie | Mécanisme concret dans v4 | Coût | Testable |
|---|---|---|---|---|---|
| 1 | **Théorie du choix social / représentation proportionnelle** (Sainte-Laguë, STV) | Une élection à listes : la fusion RRF est un scrutin majoritaire où la sous-requête la plus « bruyante » rafle tous les sièges ; un multi-hop a besoin d'un siège par circonscription | **Fusion à quotas** : les 10 places du top-10 sont attribuées par tour de rôle entre les listes (question, sous-requêtes, hypothèses) selon Sainte-Laguë ; garantit ≥ 1 tour par pièce d'évidence | 0 (code) | oui, plans existants |
| 2 | **Finance, portefeuille de Markowitz** | Diversifier pour couvrir des risques non corrélés : deux tours qui répondent à la même sous-requête sont « corrélés » | **MMR par sous-requête** : pénaliser un candidat dont la sous-requête « propriétaire » est déjà représentée, plutôt que la similarité textuelle | 0 | oui |
| 3 | **Fusion de capteurs (gain de Kalman, Dempster-Shafer)** | On pondère un capteur par sa fiabilité instantanée, pas par une constante | **Confiance par requête** : le poids d'un signal (dense, BM25, sous-requête) est proportionnel à la marge entre son score n°1 et n°2 (un signal plat = peu informatif) | 0 | oui |
| 4 | **Linguistique du discours (paires adjacentes)** | Une question appelle une réponse dans le tour suivant ; l'évidence d'un « when did X » est la déclaration de X, souvent réponse à une question du partenaire | **Expansion conditionnelle** : n'ajouter le tour suivant que si le tour rappelé est une question posée par l'autre locuteur (et non ±1 aveugle, déjà réfuté) | 0 | oui |
| 5 | **Écologie, théorie du fourragement optimal (Pirolli)** | Quitter une « parcelle » quand le gain marginal chute ; continuer à chercher tant que la couverture est incomplète | **Retrieval itératif à budget** : après un premier tour, si une sous-requête n'a aucun candidat dans le top-20, relancer uniquement celle-ci avec plus de candidats | faible | oui |
| 6 | **Épidémiologie, traçage de contacts** | Cas index → contacts → contacts de contacts, mais avec un taux de transmission dépendant du type de contact | **Graphe de faits typé** construit à l'écriture par le LLM (fait → entités, temps, cause) ; déjà le levier d'A-Mem/HippoRAG ; élève le plafond mais coûte du LLM à l'écriture | élevé | plus tard |
| 7 | **Neurosciences, réactivation hippocampique** | Le sommeil rejoue et lie les épisodes proches ; « schémas » | Résumés par sujet : réfuté (« Summary » 8,5 vs « Raw » 24,2 EM dans le harnais indépendant) | — | non |

Priorité de test : 1 → 2 → 3 → 4 → 5, sans aucun appel LLM supplémentaire (les plans et faits Sonnet
existent déjà), sur le pipeline `v4_planned` avec reranker Ettin-68m, mesure macro et par catégorie.
Critère de promotion : gain multi-hop ≥ +0,02 sans perte single-hop/temporel > 0,005, puis vérification
sur le plafond (rappel@50) — l'idée 1 doit surtout faire monter le rappel@20/@50 des multi-hop.

## Résultats des idées 1-5 et analyse d'échecs (2026-09-14)

Les cinq idées côté requête sont **neutres** (0,750-0,752 contre 0,751) : elles réordonnent ou
complètent à la marge un pool que le reranker réordonne de toute façon ; aucune ne fait entrer de
nouvelle évidence (plafonds @20/@50 inchangés). Le levier est donc à l'écriture, pas à la lecture.

Analyse des 121 questions multi-hop/open-domain dont au moins un tour d'évidence manque dans les 50
candidats (241 tours perdus) : les tours perdus sont des **mentions incidentes** — le fait attendu
apparaît en passant dans un tour consacré à autre chose :
- « How many pets does James have? » → « By the way, my dogs [photo] », « I adopted a pup… named it Ned »
- « Which geographical locations has Tim been to? » → « my trip to the Smoky Mountains last year », « a Harry Potter fan in California »
- « What people has Maria met while volunteering? » → « Cindy wrote it », « a conversation with someone named David »
- « How many children does Melanie have? » → « my son got into an accident »
- « Was James feeling lonely before meeting Samantha? » (inférence) → « My pets, computer games, travel and pizza are all that bring me happiness »
Aucune similarité avec la question ne peut faire remonter ces tours parmi 700 ; les faits Sonnet les
décrivent bien, mais avec le vocabulaire de l'instance (Ned, Smoky Mountains), pas de l'attribut (pet, lieu visité).

### Idée 8 — Classification à facettes (bibliothéconomie, Ranganathan) + encodage élaboratif (psychologie de la mémoire, Craik & Tulving)
Analogie : une bibliothèque ne retrouve pas « tous les livres sur les voyages de Tim » par similarité de
titre mais par **facettes** (sujet, lieu, temps, personne) ; et la mémoire humaine retient mieux ce qui a
été **élaboré** à l'encodage (implications, liens) qu'un enregistrement brut.
Mécanisme dans v4, à l'écriture (côté client, gratuit en production ; une passe Sonnet dans le banc) :
chaque fait reçoit (a) un triplet à facettes `{subject, predicate ∈ vocabulaire fermé (~40 : pet,
child, location_visited, person_met, hobby, job, possession, event_attended, health, feeling…), object}`
et (b) 0-2 **implications** en langage naturel (« suggère que James se sentait seul avant Samantha »).
À la lecture : le planificateur émet aussi des facettes `{subject, predicate}` ; les faits partageant
(subject, predicate) entrent dans le pool **quel que soit leur texte** (agrégation « combien / quels »),
et les implications sont indexées comme des faits (inférence).
Attendu : plafond@50 multi-hop 0,856 → ≥ 0,90, open-domain 0,69 → ≥ 0,75 ; coût banc ≈ 1,5-2 M tokens
Sonnet (3 332 faits + 1 536 questions), coût production nul (extraction par le modèle client).
Test de faisabilité avant dépense : mesurer, sur les 241 tours perdus, la part couverte par un fait
existant portant sur la même personne (borne haute de ce que la facette peut ramener).

## Idée 8 — premiers résultats gratuits (2026-09-19, 3 conversations sur 10 : 26, 30, 41 ; 383 questions)
Test hors ligne, sans run local : pool de facettes = tours dont un fait partage (sujet, prédicat) avec la
question ; `other` ignoré ; évidence utilisée pour l'évaluation seulement. Base : 50 candidats du pipeline planifié.

| Pool ajouté | nouveaux candidats / question (médiane, p90) | tours perdus récupérés | plafond macro | multi-hop | open-domain |
|---|---|---|---|---|---|
| prédicat exact | 4, 13 | 9/49 | 0,898 → 0,907 | 0,877 → 0,908 | 0,763 → 0,763 |
| famille de prédicats | 28, 62 | 18/49 | 0,898 → 0,932 | 0,877 → 0,926 | 0,763 → 0,842 |
| TÉMOIN : tous les tours de la personne | 70, 156 | 31/49 | 0,898 → 0,966 | 0,877 → 0,946 | 0,763 → 0,947 |

Lecture honnête :
1. Prédicat exact : précis et peu coûteux (+4 candidats) mais faible (27 % des tours multi-hop perdus, 0 % en
   open-domain). Cause observée : désaccord de prédicat entre l'annotateur des faits et celui des questions
   (45 prédicats, deux côtés indépendants). Ma prévision « plafond multi-hop ≥ 0,90 » était optimiste : la borne
   « 82 % des tours perdus ont un fait » mesurait l'existence d'un fait, pas la coïncidence des prédicats.
2. CONFUSION AVEC LA TAILLE : le témoin sans facette bat tout. Un plafond de rappel croît mécaniquement avec le
   nombre de candidats. Le gain « famille » n'est PAS attribuable aux facettes tant qu'il n'est pas comparé au
   plafond du retriever existant à taille égale (50+4, 50+28, 50+70). Le dump actuel s'arrête à 50.
3. Plafond ≠ R@10 : tout candidat ajouté doit encore être classé dans les 10 premiers, et reranker 80 à 120
   candidats par question coûte 3 à 4 fois plus cher en calcul local.
4. Les implications (canal visant l'open-domain) ne sont pas testées ici.
Prochaine étape décisive et bon marché : un dump à 150 candidats SANS reranker, pour calculer hors ligne les
plafonds à taille égale. Ne rien promouvoir, ne lancer aucun run reranké, avant ce témoin.

## Idée 8 — test hors ligne sur les 10 conversations (2026-09-19, 1 536 questions, `python -m bench.facet_ceiling_analysis`)
Données complètes et vérifiées : 3 332 faits annotés (3 986 facettes, 529 faits avec implication), 1 536 questions
avec facettes (texte de la question seul). Base : 50 candidats du pipeline planifié, plafond macro 0,870.

| Pool ajouté | nouveaux candidats / question (médiane, p90) | tours perdus récupérés | macro | multi-hop | temporel | open-domain | single-hop |
|---|---|---|---|---|---|---|---|
| prédicat exact | 7, 23 | 67/282 | 0,870 → 0,900 | 0,856 → 0,902 | 0,952 → 0,964 | 0,691 → 0,747 | 0,979 → 0,986 |
| famille de prédicats | 44, 100 | 126/282 | 0,870 → 0,922 | 0,856 → 0,925 | 0,952 → 0,968 | 0,691 → 0,810 | 0,979 → 0,988 |
| TÉMOIN tous les tours de la personne | 122, 260 | 205/282 | 0,870 → 0,962 | 0,856 → 0,963 | 0,952 → 0,977 | 0,691 → 0,920 | 0,979 → 0,990 |

Lecture :
1. Le prédicat exact fait mieux sur l'ensemble que sur le sous-échantillon de 3 conversations : +0,030 de plafond
   macro pour ~7 candidats de plus, multi-hop à 0,902, open-domain +0,056.
2. Effet de taille, estimation grossière : entre 30 et 50 candidats le retriever existant gagne ~0,002 de plafond par
   candidat (ordre brut) → ~+0,017 pour 7 candidats ; ~+0,006 selon la courbe de l'ordre « ensemble ». Le gain des
   facettes exactes (+0,030) dépasserait donc l'effet de taille d'un facteur 2 à 5. INDICATIF SEULEMENT : les deux
   courbes diffèrent, et seul un dump profond (150 candidats, sans reranker) donnera le témoin à taille égale.
3. Famille et témoin « personne » : gains importants mais pools de 44 à 122 candidats de plus ; confusion avec la
   taille non levée, et coût de reranking multiplié.
4. Ce sont des PLAFONDS, pas des R@10. Le critère V5.0 « R@10 macro ≥ 0,80 sans reranking LLM » (actuel : 0,751)
   n'est pas mesuré. Les implications ne sont pas testées.
Limites de protocole à rappeler avec tout résultat : facettes de questions parfois déduites des questions voisines
(ENRICH_LOG.md) ; implications annotées sous des régimes de brief hétérogènes.

## Idée 8 — témoin à taille égale (2026-09-19, `python -m bench.facet_size_matched`)
Run local unique sous garde-fou : pipeline planifié SANS reranker, 150 candidats par question (8 min, swap max
2,7 Go, compression 4,0 Go, aucun arrêt du garde-fou ; ledger : `locomo_v4_plan_norerank_k150_dump`, R@10 0,721 en
173 ms — le reranker Ettin-68m apporte donc +0,03 pour ~6× la latence). Témoin PAR QUESTION : si les facettes
ajoutent k tours aux 50 premiers, on compare au même retriever pris k candidats plus profond (top 50+k).

| Catégorie | n | top-50 | + facettes exactes | témoin top-(50+k) | écart | IC 95 % bootstrap | mieux / moins bien |
|---|---|---|---|---|---|---|---|
| multi-hop | 282 | 0,857 | 0,902 | 0,879 | **+0,023** | [+0,009 ; +0,038] | 24 / 6 |
| temporel | 321 | 0,952 | 0,964 | 0,959 | +0,006 | [0,000 ; +0,014] | 3 / 0 |
| open-domain | 92 | 0,691 | 0,747 | 0,694 | +0,053 | [+0,013 ; +0,101] | 7 / 1 |
| single-hop | 841 | 0,979 | 0,986 | 0,983 | +0,003 | [−0,002 ; +0,008] | 6 / 2 |
| macro | | 0,870 | 0,900 | 0,879 | +0,021 | | |

Conclusions :
1. Le signal des facettes exactes est RÉEL au-delà de la taille du pool en multi-hop : l'effet de taille n'explique
   qu'environ un tiers du gain brut (0,870 → 0,879 pour le témoin contre → 0,900). Robustesse : écart positif dans
   10 conversations sur 10 (+0,003 à +0,087), ce qui lève l'objection de dépendance intra-conversation du bootstrap.
2. Open-domain : +0,053 mais fragile (n = 92 ; positif dans 6 conversations sur 9, nul dans 2, négatif dans 1).
   Indicatif, pas établi. Temporel et single-hop : pas d'effet exploitable (déjà ≥ 0,95).
3. Famille de prédicats : +0,018 macro mais intervalle multi-hop contenant zéro [−0,006 ; +0,027], 25 mieux / 17
   moins bien, et témoin plafonné pour 151 questions (biais en faveur des facettes) → aucun avantage démontré sur le
   prédicat exact pour 6× plus de candidats. ABANDONNÉE.
4. Prédiction démentie : la conversation 48 (prédicats à dominante d'états diffus) gagne +0,043 en multi-hop.
5. Ce sont toujours des PLAFONDS. Le critère V5.0 est un R@10 (≥ 0,80 sans reranking LLM ; actuel 0,721 sans
   reranker, 0,751 avec Ettin-68m). Attente réaliste : le gain de R@10 sera inférieur au gain de plafond, possiblement
   au niveau du seuil de bruit (0,005). Prochaine mesure : canal « facettes » dans la fusion, puis canal
   « implications », mesurés SÉPARÉMENT.
Réserves de protocole inchangées : facettes de questions parfois déduites des questions voisines ; comparaisons
multiples (4 catégories × 2 variantes) ; les 1363/1536 top-50 identiques entre les deux dumps (Jaccard 0,995) — la
comparaison utilise le nouveau dump partout, donc reste cohérente.

## Idée 8 — R@10 réel (2026-09-19, détail dans docs/ABLATIONS.md § « V5.0 — canaux facettes et implications »)
Pipeline planifié SANS reranker, LoCoMo complet (1 536 questions), un run par configuration, déterministe.

| Configuration | macro R@10 | R@5 | MRR | multi-hop | temporel | open-domain | single-hop | p50 |
|---|---|---|---|---|---|---|---|---|
| référence | 0,721 | 0,631 | 0,573 | 0,618 | 0,860 | 0,486 | 0,919 | 173 ms |
| facettes, poids 0,5 | 0,727 | 0,650 | 0,609 | 0,625 | 0,876 | 0,484 | 0,924 | 232 ms |
| facettes, poids 1,0 | 0,746 | 0,666 | 0,617 | 0,659 | 0,884 | 0,512 | 0,927 | 189 ms |
| facettes, poids 2,0 | **0,757** | 0,671 | 0,608 | **0,690** | 0,898 | 0,513 | 0,925 | 205 ms |
| implications seules (532 indexées) | 0,719 | 0,628 | 0,584 | 0,613 | 0,867 | 0,480 | 0,917 | 199 ms |

VERDICT. Facettes (classification à facettes, vocabulaire fermé, construite à l'écriture) : gain RÉEL, +0,036 macro,
+0,072 en multi-hop, relation dose-réponse monotone à rendements décroissants ; la MRR plafonne au poids 1,0. Sans
aucun reranker, 0,757 en 205 ms dépasse le pipeline avec reranker Ettin-68m (0,751 en 1,1 s). Implications (encodage
élaboratif) : RÉSULTAT NUL (−0,002 macro, open-domain −0,006), cohérent avec les échantillons aléatoires (≈ 1 sur 2
paraphrase ou banalité). Portée : ces implications-là, indexées comme faits additionnels ; pas une preuve générale.
NON MESURÉ : facettes + reranker (run `locomo_v5_best_e68` ARRÊTÉ PAR LE GARDE-FOU à swap 6 544 Mo, non relancé).
RÉSERVES : poids choisi sur le banc où il est mesuré ; un seul banc (LongMemEval n'a pas de facettes annotées) →
AUCUNE promotion en défaut ; facettes de questions parfois déduites des questions voisines ; corrigé LoCoMo ~6 % faux.

### Idée 8 — poids des facettes choisi hors échantillon (2026-09-19, sans calcul lourd)

Objection levée : le poids 2,0 avait été choisi sur les questions mêmes où il est rapporté.
`bench/facet_weight_cv.py` refait le choix en « une conversation laissée de côté » à partir des
résultats par question déjà enregistrés (aucun nouveau run) : le poids est choisi sur 9 conversations
(meilleur R@10 macro parmi 0 / 0,5 / 1,0 / 2,0), puis noté sur la dixième.

| | R@10 macro | R@5 | MRR |
|---|---|---|---|
| référence (sans facettes) | 0,7205 | 0,6308 | 0,5731 |
| poids choisi hors échantillon | **0,7567** | 0,6713 | 0,6082 |
| écart | +0,0361 | +0,0404 | +0,0351 |

- Le poids 2,0 est choisi dans les 10 plis : l'optimisme de réglage est nul sur cette grille.
- Écart positif dans 9 conversations sur 10 (conv-41 : −0,004) ; IC 95 % par rééchantillonnage des
  conversations : [+0,020 ; +0,053].
- Par catégorie : multi-hop +0,073, temporel +0,039, open-domain +0,027, single-hop +0,006.
- Limites qui restent : 2,0 est le bord de la grille (un poids plus fort n'a pas été mesuré) ; un seul
  jeu de données ; les facettes des questions viennent d'un annotateur qui voyait les questions voisines.

### Idée 8 — second banc : protocole pré-enregistré LongMemEval (2026-09-19, écrit AVANT toute mesure)

But : savoir si le canal facettes tient hors de LoCoMo, sans rien régler sur le nouveau jeu.

- **Données** : les 150 questions stratifiées de LongMemEval oracle (les mêmes que toutes les lignes LME de
  RESULTS.md). Annotation des 1 574 tours utilisateur (texte coupé à 800 caractères ; les tours de
  l'assistant ne portent pas de fait personnel, le canal y est neutre par construction) avec le même
  vocabulaire fermé `PREDICATES.md`, non modifié. Facettes des questions produites par un annotateur
  distinct, à l'aveugle (il ne lit que le texte des questions, mélangées, une par utilisateur différent).
- **Deux bottes de foin** : (a) *oracle* standard, ~22 tours par question — le canal ne peut y être qu'un
  signal de classement ; (b) *mutualisée* : les sessions des 150 questions dans un seul index (~3 300 tours),
  ce qui ressemble à une vraie mémoire où le sujet est presque toujours « user » — c'est le test dur : la clé
  (sujet, prédicat) se réduit au prédicat et les bassins seront grands.
- **Système** : chemin de PRODUCTION (`MemoryV4.search(facets=…)`, drapeau `facet_channel`), réglages par
  défaut, sans reranker, poids 2,0 fixé par LoCoMo. Aucun autre poids ne sera essayé sur LME avant lecture du
  résultat principal.
- **Critère principal** : R@10 macro (6 types de question) en botte mutualisée, facettes contre sans facettes ;
  succès si écart > +0,005 et IC 95 % (rééchantillonnage des questions) excluant 0.
- **Garde-fou** : en botte oracle, l'écart ne doit pas être inférieur à −0,005.
- **Lecture attendue honnête** : gain plausible sur multi-session et knowledge-update ; nul sur
  single-session-assistant (aucune facette) ; risque réel de résultat nul en botte mutualisée à cause des
  bassins trop larges. Un résultat nul sera consigné tel quel et le drapeau restera éteint.

### Idée 8 — second banc : RÉSULTAT (2026-09-19) — critère pré-enregistré NON atteint

Run `lme_v5_facets_pooled_v1` (chemin de production, botte mutualisée de 3 149 tours, poids 2,0 fixé d'avance,
sans reranker ; les deux bras lus dans le même index). Swap max 5,7 Go, aucun arrêt du garde-fou, 218 s.

| | R@10 macro | R@5 | MRR |
|---|---|---|---|
| sans facettes | 0,688 | 0,599 | 0,521 |
| avec facettes | 0,665 | 0,593 | **0,629** |
| écart | **−0,023** (IC 95 % [−0,078 ; +0,029]) | −0,006 | +0,108 |

Par type (R@10) : multi-session 0,467→0,568 (11 mieux / 5 pires) ; preference 0,447→0,527 ; temporal 0,453→0,453 ;
**knowledge-update 0,94→0,74 (1 mieux / 8 pires)** ; single-session-user 0,90→0,82 ; assistant 0,92→0,88.
19 questions améliorées, 20 dégradées. Les 29 questions sans facette sont strictement inchangées (0/29).

**Verdict** : échec du critère principal (écart attendu > +0,005 avec IC excluant 0 ; obtenu −0,023, IC à cheval
sur 0). Le drapeau `facet_channel` reste éteint. Le garde-fou « botte oracle » n'a pas été lancé : sans objet.

**Diagnostic a posteriori (exploratoire, à ne pas confondre avec une confirmation)** : dans le top 10, 8 à 9
résultats sur 10 viennent du bassin de facettes — le canal se comporte ici comme un FILTRE, pas comme un vote.
Cause : le poids 2,0 est absolu. Sur LoCoMo il s'ajoutait à ~5 listes de requêtes (question + sous-requêtes +
hypothèses, poids total ≈ 5,4), soit ≈ 27 % du vote ; ici il n'y a qu'une liste (poids 1,0), soit 67 % du vote.
Conséquences cohérentes avec les chiffres : le premier bon résultat remonte (MRR +0,108, multi-session +0,10),
mais tout tour-preuve non annoté ou annoté sous un autre prédicat est éjecté du top 10 (knowledge-update −0,20).
Défaut de conception du portage, pas du principe : le poids devrait être une PART du vote total, invariante au
nombre de listes. Hypothèse à tester ensuite : part ≈ 0,27, d'abord en exploratoire sur ces 150 questions, puis
en confirmation sur des questions LME NON encore annotées (il en reste 350) — jamais sur les mêmes.

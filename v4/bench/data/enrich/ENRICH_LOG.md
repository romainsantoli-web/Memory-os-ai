# Journal des retouches manuelles des annotations (coordinateur)
- 2026-09-19 conv-47 : fusion partA + partB (718 faits, ordre vérifié). Deux implications de partB affirmaient « first mention of Jill / Samantha » — invérifiable depuis une demi-conversation. Clause retirée, inférence conservée (D17:24#0, D19:16#0). Fichiers partA/partB conservés tels quels.
- 2026-09-19 conv-42, conv-44 : implications remplacées par la passe « implications seules » (9→51 et 28→55 faits) ; originaux dans `*.facets_only_v1.jsonl`. Échantillons aléatoires : qualité inégale sur la 42 (2 paraphrases, 1 inférence sans base sur 7), meilleure sur la 44.
- 2026-09-19 conv-43 : implications remplacées par la passe dédiée (19→44 faits) ; original dans conv-43.facets_only_v1.jsonl.
- 2026-09-19 conv-41 : passe « implications seules » SANS repère chiffré. L'agent a gardé 22 implications, retiré 15 reformulations, ajouté 9 (37→31 faits). 7 retraits vérifiés au hasard : tous de vraies reformulations. Original dans conv-41.facets_only_v1.jsonl.
- INCOHÉRENCE CONNUE : la 41 a été élaguée (critère strict), alors que les 42/43/44 n'ont reçu que des ajouts et que les 26/30/47 n'ont pas été repassées. Taux par régime : avec repère « un sur cinq » ≈ 16-17 % ; sans repère ≈ 10-13 %. Ne pas harmoniser avant d'avoir mesuré si le canal « implications » apporte un gain.
- 2026-09-19 conv-48, conv-49 : annotées d'emblée avec le brief corrigé (pas de repère chiffré). Contrôle indépendant du coordinateur : ordre conforme, 0 doublon, 0 prédicat hors vocabulaire, 0 fait sans facette, 0 commentaire « first mention ». Implications : 48 → 21/343 (6 %), 49 → 59/354 (17 %). La 48 n'est pas repassée : taux bas mais tri argumenté, conforme au brief.
- PRÉDICAT « other » : 0-3 % partout sauf conv-49 (6,4 %) et conv-47 (12,2 %, 104 facettes). Lecture des cas de la 49 : ce sont des actes de conversation (question posée, photo partagée, suggestion), pas des attributs de personne → étiquette correcte, pas un trou du vocabulaire. La 47 en a davantage parce que son extraction de faits était plus dense (718 faits, bavardage inclus). Une petite part cache un vrai attribut (« Evan a proposé une randonnée, activité qui le calme » → hobby) ; non corrigée.
- EXIGENCE POUR L'INDEX DU BANC : ignorer predicate == "other" (sinon un seau {personne, other} de plus de 100 faits inonderait les candidats). Conversations à dominante d'états diffus (48 : belief 55, feeling 39, habit 36) : gain de facettes attendu faible, à lire par conversation.
- 2026-09-19 facettes des QUESTIONS (locomo_qfacets) : lots 26/30/41 et 47/48 reçus, forme vérifiée par le coordinateur. LIMITE DE PROTOCOLE : le brief n'imposait pas de traiter chaque question isolément ; l'agent du lot 47/48 a déduit le sujet de questions ambiguës (« Who is Jill? ») à partir des questions voisines du même fichier. Pas de fuite de réponse ni d'évidence, mais information inter-questions indisponible en production. Effet attendu faible (choix du sujet sur quelques questions). Non corrigé ; à signaler avec tout résultat reposant sur ces facettes.
- 2026-09-19 facettes des questions, lot 42/43/44 : l'agent a signalé un décalage d'un rang dans conv-42 (listes construites par position) qu'il dit avoir corrigé. Un décalage est invisible aux contrôles de forme. Contrôle d'alignement indépendant du coordinateur : pour les questions ne nommant qu'un locuteur, le sujet des facettes doit être ce locuteur → 100 % sur 42 (172 q), 43 (161 q), 44 (113 q), 26, 30, 41, 47 ; 98,9 % sur 48 (2 écarts isolés, indices 117 et 122, pas une fenêtre). LIMITE DU TEST : ne détecte un décalage que si les questions voisines concernent des locuteurs différents ; ne dit rien de l'alignement des prédicats.

## 2026-09-19 — LongMemEval (second banc des facettes)
- Entrées rendues par `bench/lme_facets_prepare.py` : 1 574 tours utilisateur (268 sessions, 380 k caractères,
  8 lots `lme_facets_in/batch_0X.jsonl`), 150 questions mélangées (`lme_questions/questions.jsonl`, graine 0).
- Tours : annotateurs Sonnet, vocabulaire `PREDICATES.md` inchangé, sujet = `user` sauf tiers nommé, lignes
  sans fait personnel omises, pas d'implications (réfutées sur LoCoMo).
- Questions : annotateur Sonnet distinct, à l'aveugle (interdit de lire les tours ou le jeu de données),
  1 à 3 facettes, `[]` si la question porte sur ce qu'a dit l'assistant.
- Protocole de mesure pré-enregistré dans `docs/IDEAS.md` avant toute mesure.
- Résultat de l'annotation LME (vérifié par script, 2026-09-19) : 692 tours annotés sur 1 574 (44 %), 793 facettes,
  5 `other`, 100 % des sujets = `user` ; identifiants tous présents, dans l'ordre, sans doublon ; prédicats tous
  dans le vocabulaire. Prédicat dominant `goal_plan` (166) — les utilisateurs de LME demandent surtout de l'aide
  pour un projet. Questions : 150 lignes alignées, 29 sans facette (questions sur les dires de l'assistant),
  121 avec 1 à 3 facettes. Taille de bassin attendue en botte mutualisée : médiane 29, max 159 (sur ~3 300 tours).
- Annotateur plutôt conservateur (choix de recette, préférences faibles omis) : biais contre le canal, pas pour.
- Incohérences internes au jeu signalées par les annotateurs (montants de prêt, taille d'équipe qui varient d'une
  session à l'autre) : consignées telles quelles, non réconciliées.

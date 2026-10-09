# Changelog

## 4.0.1 — ALPHA intermédiaire, publication soumise à qualification finale

- Identité de distribution et des deux moteurs : 4.0.1, sans suffixe rc.
- HTTP local : Host/Origin validés par le SDK MCP ; bearer obligatoire côté serveur,
  chargé depuis un fichier privé ; `/health` et `/ready` authentifiés, identité de build.
- Faits : remplacement sur identité explicite ; déduplication conservatrice par
  texte normalisé identique et portée compatible, jamais par seule similarité.
- Un écrivain moteur par magasin et reçus de requêtes JSONL. Résultat incertain
  après panne intermédiaire : aucune promesse de transaction commune aux fichiers.
- A2 : réparation ciblée sur plan versionné de douze invalidations INC-001/001b,
  simulation par défaut, historique/provenance préservés et trace vérifiable.
- SQLite S1 expérimental derrière Store et drapeau explicite : magasins neufs,
  historique, transactions et tests de panne de processus ; aucune migration.
- Import exact des clients/ops Claude : relais, admin token/health/backup/drill/
  hook/setup et extra `[ops]`. Revues A6 R1–R7 et
  B1–B5 levées ; qualification du relais installé exigée avant publication. Les
  garanties industrielles complètes restent à réaliser en SQLite S2–S5.
- Notices d'exploitation, règles agents et changelog inclus dans les distributions.

La revue A6 ne constitue pas seule un GO de publication. Lire
`v4/docs/A6_REVIEW_6d3f749.md` et `docs/releases/v4.0.1.md` avant toute utilisation
des outils ops. Aucun déploiement/réparation de production effectué par ce lot.

## 4.0.0

Version v4 déjà publiée sur PyPI selon l'opérateur ; moteur v4 et compatibilité v3.
Les corrections ci-dessus répondent aux incidents et à la qualification ultérieure.

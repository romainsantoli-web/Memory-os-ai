# A2 — réparation ciblée de l'incident du 9 octobre 2026, 09:41

`tools/repair_incident_0941.py` est un outil de maintenance hors ligne, limité aux neuf
identifiants du HANDOFF et à leur invalidation par `e116128b7985433bb1605ea7cab5605b`.
Il ne republie pas leurs textes : il ajoute neuf versions des mêmes identifiants,
en restaurant uniquement `valid_until` et `invalidated_by` depuis l'historique.
Tous les autres champs de la dernière version, y compris la provenance et les
métadonnées inconnues, sont conservés. Le journal historique reste un préfixe byte
pour byte du journal réparé. Aucun modèle/index/cache n'est chargé ou réécrit.

## Essai autorisé sur copie uniquement

Copier le magasin dans `.local-qualification/` (gitignoré), sans liens symboliques
ni liens physiques. Une copie prise serveur en service est une photographie de
diagnostic, **pas une sauvegarde cohérente qualifiée pour une bascule**. Vérifier
que son journal est complet. Créer le fichier de verrou vide sur cette copie
avant de prendre les empreintes de référence ; l'outil ne crée ni ne supprime ce
fichier. L'essai ci-dessous n'ouvre jamais le moteur.

```sh
python3 tools/repair_incident_0941.py --data-dir .local-qualification/a2/store
python3 tools/repair_incident_0941.py --data-dir .local-qualification/a2/store --apply --offline
python3 -m pytest tests/test_incident_repair.py -q
```

Sans `--apply`, aucun fichier n'est écrit. L'application exige `--offline` et
prend le même verrou `writer.lock` que la candidate. Les anciens serveurs v4 sans
ce verrou doivent impérativement être arrêtés : le verrou ne suffit pas à prouver
leur arrêt. Les chemins du magasin de production et du dépôt original sont
interdits, y compris via alias ; les fichiers liés sont refusés.

## Preuve et arrêt sûr

Le seul fichier existant modifié doit être `items.jsonl`, par ajout de neuf lignes.
`repair-INC-001.json` est la seule nouvelle trace : identifiants, cause, changements,
horodatage et SHA-256 avant/après ; aucun texte de souvenir. Elle est synchronisée
en état `pending` avant l'ajout puis remplacée et synchronisée en état `complete`.
Cette séquence JSONL n'est **pas une transaction entre fichiers** : après interruption,
une trace `pending` impose une réconciliation manuelle par empreintes et inspection,
jamais un rejeu automatique. Toute trace existante interdit une nouvelle application.
Un journal malformé/tronqué, une cible absente/ambiguë, supprimée ou corrigée depuis
l'incident provoque un refus avant modification.

La preuve de copie est privée : `.local-qualification/a2/proof.json` contient les
empreintes et invariants, pas les textes ; elle ne doit pas être publiée avec les données.

## Production : procédure future distincte

Aucune application en production n'est livrée/exécutée par cette tâche. Après
revue de Claude et **accord explicite de Romain pour la maintenance**, préparer une
procédure dédiée : confinement A1 prouvé, arrêt de tous les écrivains, sauvegarde
complète vérifiée, verrou exclusif, empreinte et validation des neuf transitions,
réparation tracée, vérification de l'historique, redémarrage et parcours des clients.
Ne pas contourner l'interdiction de chemin de cet outil pour exécuter la production.

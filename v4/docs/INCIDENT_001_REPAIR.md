# A2 R1 — réparation ciblée INC-001 + INC-001b

Le plan versionné `tools/repair_plans/INC-001-001b-v1.json` contient douze couples
complets `(target_id, invalidated_by, project)` : neuf faits Memory-os-ai et trois
mamba-lob-binance, avec leur incident. Les dates réelles du journal sont conservées ;
les trois anciens faits ne sont pas tous datés du 8 octobre, contrairement au résumé
initial de revue. IDs complets, cibles uniques, projet concret, schéma v1 et
empreinte exacte du plan sont validés avant toute écriture.

`tools/repair_incident_0941.py` reste en simulation par défaut. L'application exige
`--apply --offline`, un fichier writer.lock existant et son verrou exclusif. Les
anciens serveurs 4.0.0 ne prennent pas ce verrou : leur arrêt doit être vérifié
séparément. Production et dépôt original restent interdits, y compris via alias.
La [procédure R2](INCIDENT_001_PRODUCTION.md) prépare un magasin distinct.

L'outil ajoute une version par cible, même ID, sans republier le texte. Seuls
valid_until et invalidated_by sont restaurés depuis la version précédant la
transition abusive. Tous les autres champs de la dernière version, provenance
et métadonnées inconnues compris, sont conservés. Historique byte-à-byte intact ;
aucun index/cache réécrit. Une cible absente, supprimée, invalidée autrement ou
sans transition unique provoque un refus avant écriture.

La nouvelle trace `repair-targeted.json` contient plan/empreinte, incident, projet,
IDs, changements, cause, date et SHA-256 avant/après ; aucun texte. Trace pending
synchronisée avant l'ajout, puis complete après synchronisation du journal. Ce
n'est pas une transaction entre fichiers : une trace existante, notamment pending,
interdit un rejeu et exige une réconciliation manuelle.

```sh
python3 tools/repair_incident_0941.py --data-dir <copie> --plan tools/repair_plans/INC-001-001b-v1.json
python3 tools/repair_incident_0941.py --data-dir <copie> --plan tools/repair_plans/INC-001-001b-v1.json --apply --offline
python3 tools/repair_incident_0941.py --data-dir <copie> --plan tools/repair_plans/INC-001-001b-v1.json --verify --baseline-manifest <empreintes-avant.json>
```

Le manifeste avant réparation est extérieur au magasin et couvre tous ses
fichiers, après création de writer.lock. --verify prouve douze faits valides,
préfixe historique/provenance conservés, aucun autre item changé et, avec le
manifeste, aucun autre fichier modifié ni ajouté hors journal/trace.
Une copie à chaud est seulement diagnostique. Preuve privée gitignorée :
`.local-qualification/p0-review-v2/a2/` ; ne pas publier les données.

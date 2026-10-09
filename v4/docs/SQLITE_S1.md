# ADR-001 S1 — adaptateur SQLite expérimental derrière Store

Statut : **implémenté pour qualification sur magasins neufs, pas pour la production**.
Le choix SQLite est déjà approuvé ; ce lot ne réalise ni migration ni bascule.
Le backend par défaut reste `jsonl`. L'option explicite
`MEMORY_V4_STORE_BACKEND=sqlite_s1` (ou `Config(store_backend="sqlite_s1")`)
sélectionne `SQLiteStore`, sous-classe de l'interface `Store` existante.

## Périmètre livré et limites

| Livré en S1 | Preuve / limite |
|---|---|
| Souvenirs, versions intégrales, métadonnées et chunk map dans `store.sqlite3` | Réouverture : mêmes champs/provenance ; pas de `items.jsonl` ni de `meta.json` |
| Tables `items`, `item_versions`, `store_metadata`, `schema_migrations` | Têtes avec FK différées ; versions append-only, UPDATE/DELETE bloqués par triggers ; empreinte SHA-256 du JSON canonique et version de schéma |
| `append`, `append_many`, `get`, `all`, `touch`, `flush_access`, métadonnées et fichiers dérivés compatibles | API Store réutilisée par le moteur ; historique accessible via `history(id)` |
| `transaction()` avec `BEGIN IMMEDIATE`, savepoints imbriqués et verrou de connexion | Plusieurs appels Store dans un même contexte : tout validé ou tout annulé ; rechargement mémoire après rollback |
| Un écrivain moteur, connexion sérialisée entre workers | Verrou interprocessus existant conservé ; ce lot n'autorise pas plusieurs moteurs écrivains |
| Refus de source JSONL, schéma futur/étranger, chemins production/original et fichiers SQL liés | Validation avant création du verrou moteur pour les refus de chemin/source ; pas d'import ou de downgrade implicite |
| `/ready` utilise `Store.check_integrity()` | `quick_check`, FK, version/checksum de migration, empreintes des versions, dernière tête et accord source/mémoire ; commit incertain rend le magasin non prêt |

Les **blocs et reçus restent dans leurs journaux JSONL** : ils ne partagent aucune
transaction avec SQLite en S1. Les opérations du moteur avec plusieurs appels Store,
les invalidations/consolidations et les index ne deviennent **pas** atomiques de bout
en bout par ce seul adaptateur. S2 doit entourer chaque mutation complète, intégrer
blocs/reçus, séquence et statut de requête, puis qualifier la récupération des index.
Un succès `/ready` sur S1 signifie intégrité opérationnelle du profil expérimental,
jamais qualification industrielle ni satisfaction de toute la porte G1a.

Le JSON conserve tous les champs **connus** de `MemoryItem`. La conservation de champs
inconnus lors d'une migration, les traces A2, reçus legacy et tombstones sont à réaliser
en S3. Le checksum de `schema_migrations` vérifie la déclaration de migration ; il
n'est pas une attestation inviolable de tout le DDL contre un attaquant local.

## Runtime et erreurs

SQLite ≥ 3.51.3 requis (correctif WAL-reset de l'ADR). Chaque connexion configure puis
relit WAL, FK=ON, synchronous=FULL, fullfsync=ON, checkpoint_fullfsync=ON ; attente
SQL bornée à 5 s. Base créée en 0600 ; PRAGMA macOS présents ne prouvent pas la
durabilité électrique. Version, options de compilation, volume et campagnes I/O,
disque plein, alimentation restent à qualifier en S2/S4.

Après rollback, **refetch des objets obligatoire** : les références Python préalables
peuvent être périmées. Les compteurs d'accès non persistés peuvent être abandonnés ;
ce sont des statistiques, pas des mutations métier acquittées. Embeddings et index
dérivés sont hors transaction SQL et doivent être reconstruits après panne.

Une exception pendant COMMIT/RELEASE bloque les nouvelles écritures de la connexion
et `/ready`. Ne pas rejouer aveuglément : préserver le magasin, fermer, inspecter et
rouvrir pour constater les versions réellement validées. La réconciliation avec le
reçu MCP et la consultation d'état appartiennent à S2.

## Qualification exécutable, uniquement dans le clone

```sh
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /Users/romain/Memory-os-ai/.venv/bin/python -m pytest \
  v4/tests/test_sqlite_store.py v4/tests/test_writer_idempotency.py \
  v4/tests/test_readiness.py -q --tb=short
```

Critères couverts : réouverture intégrale ; historique immuable ; rollback du lot
sur erreur applicative **et SQL** ; savepoint conservant les changements extérieurs ;
écritures de workers ; corruption détectée ; refus sans migration ; intégration
moteur/readiness ; régressions INC-001/002 ; commit validé dont l'acquittement est
perdu ⇒ aucune nouvelle écriture avant inspection ; arrêt brutal `os._exit` avant/
après commit ⇒ zéro ou deux souvenirs complets après réouverture. Cette dernière
preuve est une panne de **processus**, pas une coupure électrique.

La suite S1 se marque explicitement ignorée si le runtime SQLite est trop ancien :
ce résultat **ne qualifie pas S1**. Le backend JSONL continue à être testé sur ces
plateformes ; Claude doit exécuter S1 sur le runtime cible avant acceptation.

## Lots suivants et retour arrière

S2 : transactions moteur/reçus/blocs/index et injections I/O ; S3 : import/export
strict, champs inconnus, douze réparations A2, migration/retour arrière ; S4 :
`Connection.backup()`, restauration indépendante, purge et durabilité ; S5 : dossier
de qualification puis GO opérationnel de Romain. Voir [ADR-001](ADR-001-SQLITE.md).
En S1, revenir au backend JSONL signifie choisir **un autre répertoire neuf**, pas
ouvrir le même répertoire avec un drapeau différent. Le magasin SQLite est préservé.
Ne copier jamais le seul fichier DB actif : son WAL peut porter des commits.

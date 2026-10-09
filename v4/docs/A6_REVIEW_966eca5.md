# A6 — revue adverse de Claude 966eca5

**NON VALIDÉ pour publication.** Intégration exacte, en fast-forward, de
`966eca5fff5001e4e91ce17d9dc4d01f8994c570`, parent `52b19ca` ; dépôt Claude lu
uniquement. Aucun module ops réécrit : les correctifs ci-dessous sont proposés à
Claude. Les 16 tests unitaires passent ; le 17e est TCP et reste à exécuter sur cette
version. Des tests verts ne couvrant pas ces défauts ne constituent pas une
qualification de sécurité/durabilité. La v4.0.1 préparée reste NON PUBLIABLE.

## Défauts bloquants et correctifs proposés

**R1 — arrêt non vérifié, cohérence mensongère.** `backup._launchctl` ignore les codes
de retour ; `backup` attend 2 s puis déclare `consistent_server_stopped=True` et
redémarre même si le service était déjà arrêté. Un échec bootout, autre écrivain ou
processus survivant produit une archive déclarée cohérente. Correctif : vérifier
l'état initial, retour launchctl et disparition du service/processus, détenir le
verrou écrivain pendant la copie ; ne redémarrer que ce que l'opération a arrêté.
Pour 4.0.0 sans verrou, vérifier tous les écrivains comme R2 A2 ; à défaut refuser
le mode automatique et utiliser la maintenance explicitement attestée. Dépendance :
`INCIDENT_001_PRODUCTION.md`, `tools/offline_store_backup.py`. Preuves : bootout en
échec, arrêt préexistant, écrivain restant, panne de copie => aucun succès cohérent,
état initial préservé ; arrêt réel/reprise sur store jetable hors sandbox.

**R2 — fichiers de secrets non conformes.** La clé suit symlinks, n'impose ni fichier
régulier/propriétaire ni mode exactement 0600 ; le drill la lit sans validation.
`token create` annonce un fichier existant sans le contrôler ; token read ne refuse
pas les hardlinks. Correctif : un lecteur FD `O_NOFOLLOW`, fstat régulier, propriétaire
courant, mode 0600, nlink=1, taille exacte/bornée ; l'utiliser partout. Création
exclusive et fsync fichier/répertoire ; vérifier l'existant, jamais écraser ni
recréer silencieusement une clé manquante pour déchiffrer. Preuves : symlink,
hardlink, 0400/0644, répertoire, contenu invalide et clé absente refusés sans fuite.

**R3 — archive non durable et vérification incomplète.** `.part` créé avec permissions
par défaut puis chmod, publication remplaçant un nom existant, absence de fsync et
de déchiffrement/lecture de contrôle avant rotation. copytree suit les symlinks ;
extraction accepte certains liens et `verify_manifest` ne compare pas l'ensemble
des fichiers (chemins absolus/`..`, doublons, éléments supplémentaires). Correctif :
refuser liens/fichiers spéciaux à la source et dans le tar, chemins relatifs sûrs,
préfixe unique, inventaire exact ; créer .part exclusivement en 0600, fsync,
publication sans écrasement et fsync du dossier ; relire/déchiffrer/vérifier avant
rotation. Préserver le dernier backup vérifié en cas d'échec. Preuves : liens,
path traversal, extras, doublons, collision, archive tronquée, erreur I/O et panne
aux étapes de publication ; aucune écriture hors cible ni suppression du backup sûr.

**R4 — faux succès et formats non supportés.** Une snapshot `--no-stop`, JSONL
malformée et même son drill `--no-server` retournent `ok=True`. Health les compte
comme sauvegardes valides. Un magasin SQLite S1 est copié comme des fichiers
ordinaires (WAL actif), avec zéro items JSONL, sans refus explicite. Correctif :
séparer snapshot diagnostic de backup qualifié ; refuser SQLite jusqu'au backup API
S4 ; parser strictement lignes/têtes/champs/newline ; recalculer inventaires depuis
l'archive, vérifier aussi blocs/reçus/provenance et distinguer vérification fichiers
de restauration métier. Health n'accepte que les backups cohérents relus et
vérifiés, avec fichier encore présent et empreinte correspondante. Preuves : source
malformée/tronquée, SQLite, manifeste/count falsifié, archive disparue => anomalie,
jamais succès qualifié. `--no-server` ne prouve pas le rappel ni les deux clients.

**R5 — contrat client/setup incomplet.** `setup` omet DEDUP_COS=1.01, les chemins
backup/ops et URL personnalisés, et le commit de déploiement ; une installation PyPI
sans manifeste ne passe jamais `/ready`. Clients/urllib peuvent envoyer le bearer
vers une URL hors loopback, via proxy ou redirection. Correctif : valider URL
HTTP loopback exacte `/mcp/`, sans userinfo/query/fragment, sans proxy ni redirection ;
propager toutes les options **non secrètes** aux configs/jobs ; porter le confinement
DEDUP=1.01 et supersede=1.01 et un commit attesté par manifeste d'artefact. Health et
drill doivent contrôler `/ready`, pas seulement `/health`. Si l'identité manque,
setup doit signaler/refuser l'activation non qualifiée. Preuves : config/port/chemins
personnalisés cohérents ; URL/proxy/302 refusés ; identité absente rendue explicite ;
tests des trois clients et token non imprimé.

**R6 — garanties et runbooks dangereux.** OPERATIONS affirme « no acknowledged write
lost » et production-grade alors que JSONL + reçus ne sont pas atomiques et S1 ne
termine pas S2–S5. RB-08 traite une ligne tronquée comme forcément non acquittée et
propose son déplacement ; cela peut masquer une perte acquittée. Correctif de texte :
« 4.0.1 est une candidate d'exploitation ALPHA intermédiaire. Les garanties de
transaction/durabilité industrielle attendent S2–S5. Une ligne tronquée signifie
résultat incertain : figer les écrivains, préserver une copie et ses empreintes,
comparer le journal témoin et lancer une récupération autorisée ; ne pas supprimer
ou déplacer automatiquement une ligne ni supposer qu'elle était non acquittée. »
RB-05 doit mentionner aussi le confinement dédup INC-002 et son risque résiduel sur
4.0.0. Preuve : documentation alignée ADR/readiness/A2, revue adverse approuvée.

**R7 — cycle de vie du relais non sûr lors des reconnexions.** `connect` garde un
AsyncExitStack contenant les task groups AnyIO MCP créé dans `serve` ; `call` peut
le fermer/recréer depuis une tâche de handler différente. Les cancel scopes AnyIO
ne peuvent pas être fermés depuis une autre tâche ; `close` masque l'exception.
Une initialisation échouée perd aussi son stack avant affectation. Correctif :
un propriétaire unique du cycle de vie, opérations/reconnexions sérialisées via
requêtes à cette tâche ; fermeture dans la tâche propriétaire, nettoyage de tout
échec d'ouverture. Garder zéro rejeu de call_tool. Preuves TCP requises : interruption
pendant appel (une mutation max, résultat incertain), puis listing/reconnexion
opérationnels ; appels simultanés, échec initialize et cancellation sans fuite de
processus, client ou cancel scope. Le seul round-trip sans panne ne couvre pas R7.

## Remarques non bloquantes seules

- Hook : hasher/valider session_id avant construction du chemin state ; échapper le
  nom de projet dans l'enveloppe et tester charges JSON invalides/absentes. Le
  contrôle de chemin évite une sortie de nudge-state via `../`.
- Health : scan_store et lecture des journaux peuvent lever sur source manquante,
  record sans ID ou JSON non dict ; produire une anomalie structurée et code 1,
  pas un traceback interrompant la collecte. Refuser explicitement S1 non supporté.
- Rétention : sept fichiers ne sont pas sept jours distincts ; sélectionner par
  dates et conserver quatre semaines distinctes si plusieurs backups par jour.
- Documentation installée : les liens vers v4/docs ne sont pas dans le wheel
  initial ; le dossier T05 doit assurer une notice réellement accessible au client.

## Reproduction sans production ni réseau

```sh
PYTHONPATH=v4/src PYTHONDONTWRITEBYTECODE=1 \
  /Users/romain/Memory-os-ai/.venv/bin/python scripts/a6_adversarial_probe.py
```

Code attendu actuellement **1**, cinq défauts observés (R1/R2/R3/R4). Ce code est un
refus de qualification, pas un test vert. Les tests positifs ops sont conservés
sans réécriture ; Claude peut appliquer les correctifs ciblés dans son périmètre,
puis remettre le lot à la revue. Il faudra reconstruire/requalifier les artefacts,
avec de **nouvelles** empreintes, après correction : ceux de T05 actuel ne doivent
pas être publiés.

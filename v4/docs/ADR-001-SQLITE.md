# ADR-001 — SQLite transactionnel comme stockage faisant autorité de l'ALPHA

Date : 2026-10-09. Décideur : Romain. Responsable moteur : Codex. Revue adverse : Claude.
**Statut : décision approuvée ; S1 expérimental implémenté, S2–S5 et qualification à faire.** Romain a
approuvé PLAN_FINAL_v3 (« Ok let's go ») et explicitement retenu SQLite pour l'ALPHA,
selon son instruction de mise en œuvre et `~/memory-os-local/TASKS.md`. Cette décision
remplace le report conditionnel de SQLite de l'accord précédent. Les formulations
« recommandé / décision ouverte » encore présentes dans le plan sont à aligner par
Claude dans son périmètre ; une seconde approbation du choix de stockage n'est pas requise.

Le schéma ci-dessous est le contrat cible, distinct du sous-ensemble actuellement
livré. **[S1](SQLITE_S1.md) fournit un adaptateur derrière Store, sous drapeau, pour
magasins neufs uniquement ; aucune migration exécutable ni bascule de production.**
Les blocs/reçus et transactions moteur complètes attendent S2.

## Contexte et problème

La v4.0.0 conserve les versions complètes des souvenirs dans `items.jsonl`, les blocs
dans `blocks.jsonl` et plusieurs fichiers dérivés/métadonnées. La candidate 4.0.1rc1
ajoute un écrivain unique et `requests.jsonl` : réservation persistante, action,
synchronisation des sources et reçu final. Une panne intermédiaire peut néanmoins
laisser une action partielle et un résultat incertain. Aucun verrou ni fsync séparé
ne fournit une transaction commune à ces fichiers.

L'incident 09:41 démontre aussi un défaut sémantique : une transaction pourrait
valider atomiquement neuf invalidations abusives. SQLite résout l'atomicité et la
récupération, **pas** les règles de remplacement, l'isolation, la capture ou le rappel.
Le correctif ciblé A2 et ses douze versions restaurées (neuf INC-001 et trois
INC-001b) doivent être conservés lors de la migration, avec leur trace, sans
republication de faits.

Contraintes : un utilisateur, une machine macOS Apple Silicon 16 Go, serveur MCP
local unique, petits corpus évolutifs, deux clients, maintenance solo. Garder JSONL
renforcé imposerait de développer/qualifier notre propre journal transactionnel.
PostgreSQL ajoute un service et une exploitation inutiles à ce périmètre.

## Décision et limites

SQLite via le module standard `sqlite3` devient l'unique autorité pour souvenirs,
historique, blocs, reçus, métadonnées et traces de maintenance. FAISS, BM25, entités,
facettes et caches d'embeddings restent reconstruisibles. Aucune double écriture
JSONL/SQLite en production : JSONL devient un format d'échange/export et une source
de migration figée. Le contrat MCP et les identifiants restent stables.

Un seul serveur détient le verrou interprocessus existant du magasin ; les clients
restent en HTTP local sécurisé. Les transactions SQLite complètent ce verrou,
sans autoriser des moteurs concurrents. Le magasin ne doit pas être placé sur un
volume réseau ou synchronisé par un outil cloud pendant son utilisation.

Paramètres à appliquer/vérifier à chaque ouverture : `foreign_keys=ON`,
`journal_mode=WAL`, `synchronous=FULL`, délai `busy_timeout` borné et documenté.
Sur macOS, qualifier également `fullfsync=ON` et `checkpoint_fullfsync=ON` avec
les options de compilation/versions réellement utilisées. Ne jamais passer à
`NORMAL`/`OFF` pour gagner un benchmark. Le runtime SQLite (version et options),
le système, le volume et les PRAGMA relus figurent dans le dossier de qualification.
Le profil ALPHA exige **SQLite ≥ 3.51.3** : la documentation officielle décrit un
[défaut WAL-reset corrigé dans cette version](https://www.sqlite.org/wal.html#walreset),
susceptible de corrompre une base lors de commits/checkpoints concurrents. Les
backports ne seront admis que sur preuve explicite du correctif et qualification.
Le Python local rapporte SQLite 3.53.2, ce qui ne constitue pas une qualification
de durabilité. La simple présence de SQLite ne prouve aucune garantie de coupure électrique.

## Schéma logique v1 proposé

| Table | Clé et contenu | Invariants |
|---|---|---|
| `schema_migrations` | Version, checksum de migration, date, révision applicative | Migrations ordonnées ; schéma futur refusé avant toute mutation |
| `store_metadata` | Clé/valeur : UUID du magasin, séquence de commit, schéma, configuration et profils d'index | Les métadonnées indispensables évoluent dans la transaction de l'action |
| `items` | ID stable, projet, kind, révision courante | Une tête par ID ; projet non vide ; FK différée vers la version courante |
| `item_versions` | `(item_id, revision)` ; JSON complet original, empreinte, dates et champs temporels indexables, origine de changement | Versions immuables sauf purge explicite ; FK vers l'item ; tête et historique cohérents |
| `blocks` / `block_versions` | Identité `(scope, project, name)`, tête et versions complètes | `scope=global` ⇒ projet NULL ; `scope=project` ⇒ projet explicite. Unicité globale par index partiel `(name)`, unicité projet par `(project,name)` ; aucune dépendance à l'unicité SQL des NULL |
| `requests` | `(scope, request_id)` ; opération, empreinte des arguments validés, réponse/version, séquence et date de commit | Une seule opération par identifiant et portée ; receipt et mutation dans le même commit |
| `legacy_request_reconciliation` | Reçus JSONL `pending` et cas ambigus importés | Résultat incertain conservé ; rejeu bloqué jusqu'à réconciliation tracée |
| `maintenance_events` | ID, acteur/client, cause, versions affectées, empreintes et date | A2, corrections et migrations traçables ; aucun texte ni secret dans l'audit |
| `purge_tombstones` | IDs/versions purgés, date, portée et politique applicable | Empêche restauration ou rejeu réintroduisant du contenu effacé |

Les versions conservent tous les champs actuels : texte, source, parents, projet,
kind, dates, validité, `invalidated_by`, tags, entités, keywords, facettes, identity,
replaces, scores, compteurs et suppression logique ; les champs inconnus sont
préservés dans le JSON complet. Les colonnes SQL matérialisées doivent être validées
contre ce JSON à l'écriture et à l'import, avec un schéma versionné. Les références
de provenance absentes de la source sont conservées et signalées, jamais inventées
ou supprimées pour satisfaire une FK.

Séparer temps métier (`event_time`, `valid_from`, `valid_until`) et temps système
d'observation/enregistrement. L'ordre des lignes historiques définit l'ordre des
versions, pas une date métier supplémentaire. Une précision temporelle inconnue
reste inconnue à l'import ; ne pas transformer automatiquement un ancien timestamp
à minuit en heure d'événement certaine. Des intervalles d'incertitude chevauchants
n'autorisent pas une supersession. Les règles explicites du moteur demeurent nécessaires.

Créer les index SQL sur projet/kind, état courant et dates utiles ; l'index lexical
actuel n'est pas remplacé par FTS dans ce lot. Les générations dérivées portent
séquence de source, révision du modèle, dimensions, découpage et empreintes. Une
génération périmée doit être reconstruite ou rendre le moteur non prêt ; jamais de
succès de recherche cachant une écriture acquittée absente des index.

## Contrat transactionnel et acquittement

1. Valider schémas, portée, tailles et catégories de secrets avant persistance,
   reçus compris. Préparer embeddings/calculs coûteux hors transaction d'écriture,
   puis revalider les versions concernées sous verrou moteur.
2. Ouvrir `BEGIN IMMEDIATE`. Chercher `(scope, request_id)` ; même empreinte et
   opération ⇒ réponse stockée, contenu différent ⇒ conflit sans mutation.
3. Écrire en une transaction toutes les nouvelles versions, têtes, blocs,
   invalidations, métadonnées, séquence et réponse complète du reçu.
4. `COMMIT` avec synchronisation qualifiée ; un acquittement n'est émis qu'après
   réussite. Mettre à jour les index sous verrou avant de servir une lecture
   cohérente ; échec d'index ⇒ état de récupération explicite, données conservées.
5. Panne avant commit ⇒ aucun effet ni reçu validé ; après commit avant réponse ⇒
   même identifiant restitue le résultat sans refaire l'action. Si le commit lui-même
   a un résultat incertain, ne pas annoncer un échec définitif : rouvrir/consulter
   son état. Ajouter l'outil de consultation des requêtes prévu au §5 du plan.

Le moteur traite également les appels historiques sans request_id en transaction,
mais ne leur promet pas de rejeu sûr ; les intégrations ALPHA doivent en fournir un.
Les opérations modifiant plusieurs projets nécessitent leur autorisation globale
explicite et une seule transaction. Aucun reçu ne doit expirer silencieusement en
ALPHA : la limite de garantie serait sinon ambiguë.

Après purge explicite, la confidentialité prime sur la restitution d'une ancienne
réponse contenant le texte purgé : supprimer son contenu, conserver une empreinte
minimale/tombstone et répondre « résultat purgé » plutôt que réappliquer la mutation.
Cette exception au résultat identique doit être documentée et testée dans le contrat.
Les erreurs, imports et journaux ne doivent jamais contourner le filtrage des secrets.

## Migration JSONL → SQLite et retour arrière

Migration **hors ligne**, sous maintenance autorisée, après A1/A2 et qualification :

1. Arrêter tous les écrivains ; acquérir le verrou ; faire et vérifier une sauvegarde
   complète du magasin JSONL, avec manifeste et SHA-256. La copie A2 prise serveur
   en service sert au diagnostic, pas à cette sauvegarde de bascule.
2. Figer la source. Inventorier toutes les lignes/versions/items/blocs/reçus,
   métadonnées et trace A2. Toute ligne malformée ou tronquée bloque l'import ;
   aucune reprise du comportement actuel « avertir puis ignorer ».
3. Importer dans une base temporaire séparée, sans toucher à la source : préserver
   IDs, ordre des versions, champs connus/inconnus et provenance. Charger les reçus
   complets ; isoler les `pending` avec résultat incertain, sans les convertir en
   opérations réussies. Reporter les anciens blocs sans portée dans `default`,
   avec trace, conformément à la candidate.
4. Vérifier `integrity_check`, `foreign_key_check`, inventaires, historique complet,
   données normalisées/export, têtes, validités, douze faits A2 et invariants de portée.
   Reconstruire les index ; comparer rappel/brief/chronologie sur corpus fixé.
5. Fermer/synchroniser la cible et son répertoire, conserver le manifeste. Basculer
   atomiquement un sélecteur de backend/configuration versionné, serveur arrêté.
   Tester les deux clients après redémarrage. La présence d'un fichier `.sqlite`
   ne doit jamais provoquer une migration/bascule automatique au premier accès.

Une interruption avant bascule laisse la source intacte ; la cible provisoire n'est
jamais utilisée. Relancer doit reconnaître source, empreintes et état de migration,
pas ajouter deux fois ses lignes. Chaque migration de schéma SQLite est versionnée,
transactionnelle lorsque possible et testée sur copie ; les étapes non transactionnelles
ont une procédure de reprise explicite.

**Réversibilité :** avant toute nouvelle écriture SQLite, retour par sélection du
backend précédent et restauration de la sauvegarde JSONL compatible. Après écritures
SQLite, conserver la base complète et un export vérifié des changements pour
réconciliation ; revenir à la sauvegarde implique un point de récupération ancien
explicitement accepté, jamais une prétendue conversion automatique sans perte.
L'export complet vise un aller-retour sémantique contrôlé ; le moteur JSONL précédent
ne sait pas représenter toutes les nouvelles garanties/reçus. Le retour SQLite → v3
n'est pas promis. Prévoir un retour à une version SQLite compatible si le schéma
le permet ; sinon restauration contrôlée, aucun downgrade aveugle.

## Sauvegarde, restauration et purge

Employer `sqlite3.Connection.backup()` vers une base temporaire indépendante, puis
fermer les connexions, assurer un artefact destination autonome sans WAL externe
requis, vérifier intégrité/FK et relever sa séquence de récupération. Ne pas copier
le seul fichier principal d'une base active : le WAL peut porter des commits.
Après synchronisation/empreinte, produire le manifeste versionné puis la copie
chiffrée sur support indépendant (Claude), avec récupération de clé testée. Un
échec laisse la dernière sauvegarde valide intacte et déclenche une alerte dédiée.

RPO = 0 pour les écritures acquittées dans le modèle N-01 à volume récupérable ;
perte totale du volume ⇒ RPO ≤ 24 h par copie indépendante ; RTO < 15 min à mesurer.
Proposition de rétention : 7 quotidiennes + 4 hebdomadaires, à arrêter en exploitation.
Restaurer vers un autre répertoire, vérifier données/intégrité, reconstruire les
index et tester les deux clients. Aucune preuve limitée à l'ouverture du fichier.

La purge couvre versions, reçus, index/caches et résidus SQLite/WAL/fichiers
temporaires ; `DELETE` seul n'est pas une preuve de disparition physique. Qualifier
`secure_delete`, checkpoint et reconstruction/VACUUM si nécessaire avec marqueurs
synthétiques ; distinguer contrôle des fichiers actifs et limites SSD/FileVault.
Avant remise en service d'une sauvegarde ancienne, réappliquer le registre de purge
conservé indépendamment du magasin restauré. Documenter rétention, destruction ou
inaccessibilité des anciennes copies chiffrées ; jamais de réintroduction silencieuse.

## Qualification obligatoire (preuves, pas promesses)

| Critère | Méthode et livrable |
|---|---|
| Atomicité de chaque mutation/lot | Injection avant/pendant/après commit sur ajout, correction, consolidation, oubli, blocs ; ensemble complet ou absent, aucune tête orpheline |
| 0 écriture acquittée perdue | Journal témoin client indépendant ; SIGKILL aux frontières ; redémarrage, comparaison exacte des IDs/états/réponses |
| Rejeu exact et conflits | Deux clients simultanés, même identifiant avant/après redémarrage ; une mutation ; empreinte différente refusée ; commit avant réponse puis consultation d'état |
| Disque plein / I/O / corruption | Volume et magasin jetables ; panne WAL/checkpoint/backup/commit ; erreur explicite, aucun acquittement mensonger ni disparition silencieuse |
| Pannes système/alimentation | Campagnes distinctes en environnement dédié/jetable ; plateforme/volume/options archivés. SIGKILL seul ne valide pas cette garantie |
| Migration réversible | Copies représentatives + cas synthétiques malformés/pending/champs inconnus ; source SHA-256 inchangée ; comparaison de toutes les versions et exercice de retour arrière |
| Sauvegarde indépendante | Backup API pendant écritures puis restauration ailleurs, clé récupérée, état/point de récupération exacts, chronométrage RTO, âge de dernière copie mesuré |
| Purge vérifiable | Marqueurs absents des DB/WAL/index/reçus ; restauration d'une ancienne copie n'en réintroduit aucun ; limites physiques documentées |
| Rappel et exploitation | Tests d'isolation, temporalité, N-03…N-06 sur profil figé, générations d'index cohérentes, `/ready` refusé pendant récupération |

## Plan de livraison et portes

| Lot | Responsable / revue | Sortie vérifiable |
|---|---|---|
| S1 Schéma + adaptateur SQLite | Codex / Claude | Runtime qualifié, invariants SQL, tests sur stores synthétiques ; backend derrière sélection explicite, sans migration automatique |
| S2 Transactions + reçus + index | Codex / Claude | Toutes les mutations atomiques, consultation d'état, tests de panne/rejeu/isolation verts ; cohérence index démontrée |
| S3 Import/export + rollback | Codex / Claude | Outils hors ligne, manifeste, comparaison exhaustive et interruption/réversibilité sur copies ; aucun accès d'écriture production |
| S4 Backup/restore + exploitation | Codex moteur ; Claude support/rotation/runbooks | Backup API, copie indépendante, purge, durabilité système et restauration des deux clients prouvées |
| S5 Qualification/bascule | Codex dossier ; Claude applique ; Romain GO/NO GO | G1a : garanties de données, sécurité et runbooks démontrés ; bascule autorisée et tracée avant P3/pilote |

S1 → S2 → S3 ; S4 dépend de S2 et se clôt avec S3 ; S5 exige tous les lots, SEC-2,
A2 revue et qualification de version. La candidate JSONL reste un correctif
intermédiaire, elle ne satisfait pas par elle-même la porte industrielle. La mesure
d'usage P3 commence après ces fondations ; les campagnes lourdes restent séquentielles.
Le choix SQLite est approuvé ; **l'exécution d'une migration/réparation en production
et chaque bascule restent des décisions opérationnelles distinctes de Romain**.

Références primaires : [commit atomique SQLite](https://www.sqlite.org/atomiccommit.html),
[WAL](https://www.sqlite.org/wal.html), [PRAGMA](https://www.sqlite.org/pragma.html),
[API backup](https://www.sqlite.org/backup.html),
[API Python sqlite3](https://docs.python.org/3.11/library/sqlite3.html#sqlite3.Connection.backup).
Documents du clone : `SINGLE_WRITER.md`, `INCIDENT_001_REPAIR.md`,
`HTTP_CLIENT_CONTRACT.md` ; plan commun et tableau des tâches cités ci-dessus.

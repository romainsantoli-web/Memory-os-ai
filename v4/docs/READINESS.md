# /health et /ready — contrat d'exploitation

Les deux routes exigent les mêmes Host/Origin et Authorization Bearer (SEC-1).
`/health` retourne 200 si le processus HTTP répond ; il ne prouve pas les données.
`/ready` retourne 200 uniquement si le moteur chargé est ouvert, détient son
écrivain, modèle chargé, sources strictement lisibles/cohérentes et tous les index
couvrent les éléments non supprimés. Sinon 503 avec une raison sans contenu privé.

Le probe ne crée pas le moteur, ne charge pas un modèle et ne répare rien. Employer
`--warm` pour un démarrage prêt sans attendre un appel d'outil. Les contrôles bloquants
de disque sont déportés dans un worker ; le verrou moteur est essayé sans attente.
Une opération en cours peut retourner `engine_busy`, état transitoire à distinguer
d'une corruption. Les lectures avec compteurs d'accès non encore synchronisés ne
provoquent pas un faux état de corruption.

Les sources items/blocs/reçus JSONL sont contrôlées strictement, y compris après
un chargement permissif ayant sauté des lignes. Préfixe tronqué, disparition de
source, état différent de la mémoire, journal de reçus empoisonné ou index incomplet
interdisent ready. Ce contrôle structurel ne prouve ni vérité sémantique ni absence
de toute altération valide : les campagnes de durabilité et audits restent requis.
Les caches dérivés absents au démarrage peuvent être reconstruits par le moteur ;
une source corrompue doit passer par une récupération explicite.

Réponses : version du code `memory_os_v4.__version__`, commit complet, source du
commit, indicateur de modifications du code. En checkout, Git HEAD est lu au
démarrage ; une modification de v4/src, src ou pyproject rend la build non qualifiée.
Les configs locales MCP ne sont pas du code livré. Pour une wheel, Claude fournit
`MEMORY_V4_BUILD_COMMIT=<SHA-40-du-manifeste-qualifié>` dans l'environnement serveur.
La valeur est une attestation de déploiement, pas une vérification cryptographique
de l'artefact : comparer commit/empreintes au dossier de version avant bascule.
Commit inconnu ou checkout modifié ⇒ `/ready` 503 `build_unqualified` ; aucun
commit inventé à partir du simple numéro 4.0.1rc1. SHA malformé ⇒ démarrage refusé.

Les raisons `store_integrity`, `blocks_integrity`, `requests_integrity`,
`integrity_check_failed`, `writer_lease_missing` exigent confinement/inspection,
jamais une boucle de relance. `indexes_not_ready` exige le runbook de reconstruction.
`engine_not_loaded`/`model_not_loaded` nécessitent la vérification du démarrage/modèle.
Une sauvegarde périmée garde son alerte séparée : elle ne fait pas échouer ready.

Qualification : tests ASGI authentification/liveness/readiness, magasin synthétique
et contrôle de corruption/index/modèle/version. Compléter le test TCP et le
démarrage des applications hors sandbox avant qualification opérationnelle.

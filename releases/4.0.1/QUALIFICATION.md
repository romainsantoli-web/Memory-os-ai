# T05 — 4.0.1 requalifiée localement, relais installé TCP en attente

**Revue A6 VALIDÉE ; garde PyPI encore fermée. Aucune publication.** Tous les
contrôles réalisables ici sont verts ; le test TCP du relais installé depuis les
**deux nouveaux artefacts** doit revenir de Claude avant l'ouverture de la garde.
Le refus actuel code 1 de `verify_release_manifest.py` est attendu. Les anciennes
empreintes/échecs du dossier précédent sont remplacés par ce dossier, pas réutilisés.

Source figée : **`0bf2d3f20150186084cdf64d6128fd3fadec1661`**. Elle intègre
6d3f749, les scripts de preuve/garde et les notes finales. Le merge d'intégration
4679dca a le même arbre que 6d3f749, en conservant la revue originale et son cherry-pick.
Le commit de ce dossier est un reçu post-construction, distinct du commit source ;
les fichiers `releases/` ne sont pas inclus dans les artefacts. Aucun changement
runtime ops/moteur après 6d3f749 ; les dix TCP source de Claude restent pertinents.

## Preuves locales

Sorties complètes et manifeste : `.local-qualification/release-4.0.1-v2/` ; le
[reçu JSON](qualification.json) conserve leur inventaire/empreintes. Depuis le clone,
`PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` :

| Contrôle | Résultat |
|---|---|
| Revue adverse 6d3f749 + sonde | 0 blocage ; `scripts/a6_adversarial_probe.py` code 0 |
| Ops unitaires | 75 passed sur 6d3f749 |
| `pytest tests --cov=memory_os_ai --cov-fail-under=80` | **478 passed**, couverture **82,00 %** |
| `pytest v4/tests --ignore=v4/tests/test_server_http_e2e.py --ignore=v4/tests/test_ops_e2e.py --cov=memory_os_v4 --cov-fail-under=80` | **795 passed**, couverture **86,72 %** ; MiniLM sémantique déjà en cache inclus |
| Garde de publication | 8 tests : preuve absente, source/artefact différent, cas manquant/échoué, contrôle local échoué, artefact ou preuve modifiés => refus ; preuve complète => enregistrement possible, sans upload |
| Lint CI | `ruff check src tests scripts` et correctness E9,F sur src/tests/v4/bench/scripts/tools : verts |
| Construction | `python3 -m build --no-isolation` ; wheel construit depuis le sdist ; SOURCE_DATE_EPOCH du commit source |
| `twine check --strict` | **2 PASSED** |
| Installation | Deux venv neufs hors checkout, artefact `[ops]`, no-index/no-deps/no-build-isolation/no-cache, puis `pip check` : **2 passes** |
| Modules/docs/licences | Imports depuis chaque site-packages local ; LGPL+GPL, notices et cinq scripts présents |
| Essais installés | **2 passes** de `package_smoke.py` : 3 serveurs MCP stdio réels, v4 help/selfcheck, 6 commandes admin, erreur du relais indisponible contrôlée |
| Script TCP installé | Syntaxe/API MCP et `--check-only` : verts ; **pas de test TCP exécuté par Codex**, pas de fausse preuve produite |

Python testé 3.11.14, macOS ARM64. Dépendances d'un venv existant copiées dans un
pool privé /tmp sans projet éditable/hooks de source ; deux environnements isolés
partagent ce pool de dépendances, pas le paquet installé. Rien téléchargé (modèles
ou dépendances). Ce n'est pas une preuve de résolution réseau fraîche. Couvertures
séparées pour éviter la fusion v3/v4. Plateformes/versions Python supplémentaires
restent à la matrice CI ; Python 3.12 n'est pas présent dans l'inventaire Homebrew
local, aucun téléchargement lourd engagé pour l'ajouter.

Les tests backup/drill CLI utilisent une **snapshot diagnostic** et un drill de
niveau fichiers explicitement étiquetés, pas une sauvegarde de production qualifiée.
JSONL/reçus ne sont pas atomiques ensemble ; SQLite industriel attend S2–S5.
Aucun service live, clé réelle, magasin de production ou dépôt original modifié.

## Nouvelles empreintes SHA-256

```text
343ab2bae898b2458608e6017ed4e5994befdb25322250437b8ce66e63c4412a  memory_os_ai-4.0.1-py3-none-any.whl
1f161568f5228cefc6cc46d30f3291678ff5db61cd96a75eb5cb5a29bbc57b2e  memory_os_ai-4.0.1.tar.gz
```

Artefacts sous `.local-qualification/release-4.0.1-v2/artifacts/`, hors bundle Git.
Leurs SHA sont vérifiés à nouveau avant tests TCP/enregistrement/upload. Conserver
ces fichiers et les environnements de `environments.json` jusqu'à fin de qualification.
Ne pas reconstruire entre cette qualification et la publication.

## Claude : dernière preuve hors sandbox, puis enregistrement conditionnel

Dix tests TCP source déjà verts selon `claude_response_A6_v2.md` à 6d3f749 (8 serveur,
2 relais). La preuve manquante concerne les **installations** wheel et sdist, pas un
nouveau lancement du relais source. Le script ci-dessous n'installe rien, ne
contrôle aucun service existant et ne télécharge rien ; chaque cas utilise un serveur
hash, un port alloué, un token temporaire et un magasin synthétique hors checkout.

```sh
cd /Users/romain/Documents/Codex/2026-10-09/v/work/implementation
release_python=/Users/romain/Memory-os-ai/.venv/bin/python
release_dir=.local-qualification/release-4.0.1-v2
"$release_python" scripts/installed_relay_e2e.py \
  --manifest "$release_dir/manifest.json" \
  --environments "$release_dir/environments.json" \
  --output "$release_dir/installed-relay-proof.json"
```

Attendu : `wheel: PASS`, `sdist: PASS`, code 0. Preuves de chaque cas : provenance
site-packages et SHA d'installation, version/commit `/ready`, token requis,
round-trip ajout/recherche, SIGKILL du seul serveur synthétique, appel en échec
borné sans rejeu, même session du relais reconnectée, exactement deux faits
acquittés dans JSONL et absence du fait tenté pendant la panne, arrêt propre sans
erreur AnyIO. Les deux cas ont un timeout global ; aucune consolidation live.
SIGKILL ne prouve pas une coupure électrique. `--check-only` ne qualifie rien.

En cas de succès des **deux** cas, enregistrer :

```sh
"$release_python" scripts/verify_release_manifest.py "$release_dir/manifest.json" \
  --record-relay-proof "$release_dir/installed-relay-proof.json"
"$release_python" scripts/verify_release_manifest.py "$release_dir/manifest.json"
```

Cette commande ne lève la garde que si contrôles locaux, SHA des artefacts, source,
SHA des scripts et preuves de tous les scénarios correspondent. Tout cas manquant,
échoué ou ancien refuse ; aucun upload n'est exécuté. Le champ `pending_at_build`
est historique ; `release_ready` et l'empreinte du rapport accepté indiquent l'état
actuel. Claude transmet ce rapport et ses résultats, puis met à jour le reçu/dossier
et TASKS/HANDOFF/mémoire depuis son périmètre. La version demeure fermée jusque-là.

## Romain : publication seulement après retour des preuves et GO

Authentification PyPI hors commande, via sa configuration/gestionnaire de secrets.
La commande gardée exacte, depuis le clone, est :

```sh
/Users/romain/Memory-os-ai/.venv/bin/python scripts/verify_release_manifest.py \
  .local-qualification/release-4.0.1-v2/manifest.json && \
/Users/romain/Memory-os-ai/.venv/bin/python -m twine upload --non-interactive \
  --repository-url https://upload.pypi.org/legacy/ \
  .local-qualification/release-4.0.1-v2/artifacts/memory_os_ai-4.0.1-py3-none-any.whl \
  .local-qualification/release-4.0.1-v2/artifacts/memory_os_ai-4.0.1.tar.gz
```

La garde n'est pas une attestation cryptographique d'un opérateur ; elle évite les
preuves absentes/incohérentes et les erreurs d'artefact. Le GO reste celui de Romain.
Publication PyPI, réparation A2 live et bascule SQLite sont des décisions distinctes.

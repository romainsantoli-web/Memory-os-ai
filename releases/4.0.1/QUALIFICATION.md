# T05 — préparation 4.0.1, NON QUALIFIÉE POUR PYPI

**Décision : NO GO. Aucun upload réalisé.** Ce dossier clôt la revue et les essais,
pas les défauts A6. La branche Claude 966eca5 est intégrée à l'identique, avec
[sept correctifs proposés](../../v4/docs/A6_REVIEW_966eca5.md), sans réécriture ops.
Les artefacts ci-dessous ne doivent pas être publiés. Version/code source figés :
`b58d0f4405e87ad6fcfce26df88ce8b124e274d7`. Le commit de ce dossier est un reçu
post-construction, distinct du commit source ; ne pas attribuer ses artefacts à
une autre révision. Aucun changement de .git original, serveur ou magasin live.

## Résultats et preuves

Journaux/manifeste : `.local-qualification/release-4.0.1/` dans le clone Codex.
Le [reçu JSON](qualification.json) contient les empreintes de ces preuves et les
résultats ; les sorties complètes restent privées/ignorées. Commandes depuis le clone,
avec `PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` :

| Contrôle | Commande / résultat |
|---|---|
| v3 | `python -m pytest tests -q --cov=memory_os_ai --cov-fail-under=80` : **470 passed**, couverture **82.00 %** |
| v4 | `python -m pytest v4/tests --ignore=v4/tests/test_server_http_e2e.py --ignore=v4/tests/test_ops_e2e.py -q --cov=memory_os_v4 --cov-fail-under=80` : **736 passed**, couverture **87.20 %**, test sémantique MiniLM **en cache** inclus |
| Ops unitaires | `python -m pytest v4/tests/test_ops.py -q` : **16 passed** (inclus dans v4) |
| Lint CI | `ruff check src tests scripts` + `ruff check src tests v4/src v4/tests v4/bench scripts tools --select E9,F` : verts |
| Construction | `SOURCE_DATE_EPOCH=1791547200 python3 -m build --no-isolation --outdir .local-qualification/release-4.0.1/artifacts` : wheel construit depuis le sdist, setuptools 80.9.0/wheel 0.45.1 |
| Métadonnées | `python -m twine check --strict <wheel> <sdist>` : **2 PASSED** |
| Installation | Deux venv distincts **hors checkout**, `pip install --no-index --no-deps --no-build-isolation '<artefact>[ops]'`, puis `pip check` : **2 passes** |
| Modules/docs/licences | Imports depuis chaque site-packages local, cinq scripts présents, LGPL+GPL et notices installées : passes |
| Trois serveurs installés | MCP stdio réel, listing ; v4 --help/--selfcheck sur données temporaires : passes |
| Admin installé | token/create, setup, hook, backup diagnostic, drill sans serveur, health serveur indisponible : comportements CLI exercés dans les deux environnements |
| Relais installé | `memory-os-relay` avec serveur synthétique indisponible : **ÉCHEC dans les deux environnements**, au lieu d'une erreur contrôlée ; script `package_smoke.py` retourne 1 |
| Revue adverse | `scripts/a6_adversarial_probe.py` : code **1**, cinq défauts R1/R2/R3/R4 observés sur fichiers synthétiques, launchctl simulé |

Les campagnes coverage utilisent des `COVERAGE_FILE` distincts : aucune fusion
accidentelle v3/v4. Python testé 3.11.14 / macOS ARM64 ; SQLite 3.53.2. Python 3.12
et la CI distante restent à exécuter. Dépendances existantes copiées (pas seulement
référencées) dans un pool privé /tmp, commun aux deux venv sans system-site-packages,
sans projet éditable ni hooks .pth source. Aucun téléchargement de dépendance ou
de modèle. Ce contrôle ne prouve pas une résolution réseau de dépendances fraîche.

TCP : les **8 tests à 52b19ca sont verts selon Claude**, pas une preuve sur cette
nouvelle révision. Les 8 tests serveur + 1 relais (**9**) restent à exécuter hors
sandbox, puis ajouter panne/reconnexion/initialize/cancellation R7. Ne pas tester
sur le magasin live. Ne pas confondre le smoke diagnostic backup/drill avec la
qualification de cohérence/durabilité industrielle : R1–R4 la bloquent.

## Blocages

R1 arrêt/cohérence non vérifiés ; R2 secrets non conformes ; R3 publication d'archive
non durable/vérification incomplète ; R4 faux succès backup/drill et SQLite non
supporté ; R5 setup/identité/confinement/transport client ; R6 garanties et récupération
de troncature non justifiées ; R7 gestion AnyIO du relais.

R7 est maintenant **observé sur le wheel installé** :
`RuntimeError: Attempted to exit cancel scope in a different task than it was entered in`,
plus CancelledError lors d'initialize, fermeture tardive d'un générateur async.
Journal `relay-unavailable-R7.log`, token synthétique retiré. Cela confirme le défaut
de cycle de vie signalé en revue : ne pas assouplir l'assertion de smoke pour rendre
le relais vert. Claude corrige les modules existants, ajoute les tests de panne,
soumet la revue, puis Codex requalifie/reconstruit **de nouveaux artefacts**.

## SHA-256 — artefacts de revue NON PUBLIABLES

```text
9b619d619cd0fe3e865c140e035ff35d27baadbc3ea54f12874eb3720a35f6d6  memory_os_ai-4.0.1-py3-none-any.whl
dac3e0c070642831c98dcdee030a33beb7c4a4bc1e5849d16de943bfd972d4c8  memory_os_ai-4.0.1.tar.gz
```

Ils sont disponibles sous `.local-qualification/release-4.0.1/artifacts/` ; pas dans
le bundle Git. Le bundle apporte code source, revue, notes et ce reçu vérifiable.

## Commande exacte pour Romain, gardée — NE PAS EXÉCUTER AUJOURD'HUI

La commande suivante s'arrête actuellement **avant twine** : `release_ready=false`,
refus de la garde testé (code 1). Après corrections, revue et nouvelle qualification,
remplacer les artefacts/manifeste par ceux approuvés et leurs nouvelles empreintes ;
ne pas simplement passer ce manifeste à true. Gérer l'authentification PyPI hors
commande (configuration/gestionnaire de secrets de Romain), jamais dans le dossier.

```sh
cd /Users/romain/Documents/Codex/2026-10-09/v/work/implementation
/Users/romain/Memory-os-ai/.venv/bin/python scripts/verify_release_manifest.py \
  .local-qualification/release-4.0.1/manifest.json && \
/Users/romain/Memory-os-ai/.venv/bin/python -m twine upload --non-interactive \
  --repository-url https://upload.pypi.org/legacy/ \
  .local-qualification/release-4.0.1/artifacts/memory_os_ai-4.0.1-py3-none-any.whl \
  .local-qualification/release-4.0.1/artifacts/memory_os_ai-4.0.1.tar.gz
```

La garde vérifie statut, version, identité source, inventaire et SHA-256 ; elle ne
constitue pas un système d'approbation inviolable. Romain conserve le GO de
publication. Aucun changement de licence à approuver : LGPL-3.0-or-later déjà
confirmée. A2 production attend un GO opérationnel distinct ; SQLite S1 ne doit
pas être activé sur le magasin existant. Claude consigne cette revue et le NO GO
dans TASKS/HANDOFF/mémoire partagée depuis son périmètre autorisé.

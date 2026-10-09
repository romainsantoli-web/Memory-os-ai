# A2 R2 — procédure proposée à Claude, non exécutée par Codex

**GO explicite de Romain requis**, après revue du plan des douze cibles et de la
preuve sur copie. Exécuter depuis Terminal ; clients et captures suspendus.
Cette opération répare le magasin du serveur 4.0.0 confiné, sans simultanément
changer de version/backend. La bascule 4.0.1 est une opération distincte.

Prérequis : confinement supersession/déduplication qualifié, clé de sauvegarde
32 octets existante en 0600 et récupérable indépendamment, espace pour deux copies
et archive, plan revu et chemins canoniques. Aucun consolidate/compact.

## 1. Variables, arrêt et exclusivité

Conserver ces variables dans un seul shell et journaliser les résultats dans le
dossier privé, sans secrets. Les chemins de travail doivent être nouveaux.

```sh
set -euo pipefail
umask 077
repair_stamp=$(date +%Y%m%d-%H%M%S)
repair_clone="$HOME/Documents/Codex/2026-10-09/v/work/implementation"
repair_python="$HOME/Memory-os-ai/.venv/bin/python"
repair_store="$HOME/.memory-os-v4"
repair_stage="$HOME/.memory-os-v4.maint-$repair_stamp"
repair_old="$HOME/.memory-os-v4.pre-repair-$repair_stamp"
repair_ops="$HOME/memory-os-local/ops/repair-$repair_stamp"
repair_archive="$HOME/memory-os-backups/memory-os-v4-$repair_stamp.mosb"
repair_key="$HOME/.config/memory-os/backup.key"
repair_plan="$repair_clone/tools/repair_plans/INC-001-001b-v1.json"
repair_domain="gui/$(id -u)"
repair_label=com.romain.memory-os-v4
repair_plist="$HOME/Library/LaunchAgents/$repair_label.plist"
mkdir -m 700 "$repair_ops"
test ! -e "$repair_stage" && test ! -e "$repair_old" && test ! -e "$repair_archive"
if launchctl print "$repair_domain/$repair_label" >/dev/null 2>&1; then
  launchctl bootout "$repair_domain/$repair_label"
fi
if launchctl print "$repair_domain/$repair_label" >/dev/null 2>&1; then exit 1; fi
if pgrep -fl '[m]emory_os_v4|[m]emory-os(-ai|-v4).*--transport[ =]http'; then exit 1; fi
if lsof -nP -iTCP:8765 -sTCP:LISTEN; then exit 1; fi
```

Fermer aussi tout écrivain stdio/automatisation identifié. Port vide et verrou
seuls ne prouvent pas l'arrêt des anciens écrivains 4.0.0. Contrôle échoué ⇒ arrêt
de procédure ; aucune suppression automatique de processus ou verrou.

## 2. Sauvegarde cohérente chiffrée et exercice de lecture

`backup_v4.py` actuel relance dans son finally ; --no-stop marque la copie non
cohérente. Ne pas l'utiliser tel quel au milieu de la fenêtre. Le nouvel outil
hors ligne du clone lit la source et ne gère aucun service :

```sh
mkdir -p "$HOME/memory-os-backups"
"$repair_python" "$repair_clone/tools/offline_store_backup.py" --source "$repair_store" \
  --archive "$repair_archive" --key-file "$repair_key" --offline > "$repair_ops/backup.json"
"$repair_python" "$repair_clone/tools/offline_store_backup.py" --source "$repair_store" \
  --archive "$repair_archive" --key-file "$repair_key" --verify > "$repair_ops/backup-readback.json"
```

MOSB1 est compatible avec restore_drill_v4.py. Le contrôle déchiffre en mémoire,
vérifie fichiers/tailles/SHA-256/comptes contre la source figée ; aucune extraction.
Copier l'archive chiffrée sur le support indépendant approuvé et vérifier SHA-256.
Ne jamais copier la clé dans le magasin/archive/journal. --offline est une
attestation de l'opérateur, pas une détection des processus arrêtés.

## 3. Copie indépendante, simulation, application et preuve

```sh
"$repair_python" - "$repair_store" "$repair_stage" "$repair_ops/before.json" <<'PY'
import hashlib, json, shutil, sys
from pathlib import Path
source, stage, manifest = map(Path, sys.argv[1:])
assert not any(p.is_symlink() for p in source.rglob('*'))
shutil.copytree(source, stage)
(stage/'writer.lock').touch(exist_ok=True)
before = {str(p.relative_to(stage)): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in stage.rglob('*') if p.is_file()}
manifest.write_text(json.dumps(before, indent=2)+'\n')
PY
"$repair_python" "$repair_clone/tools/repair_incident_0941.py" --data-dir "$repair_stage" \
  --plan "$repair_plan" > "$repair_ops/simulation.json"
# Revoir les 12 couples/empreintes avant de poursuivre ; tout refus impose un arrêt.
"$repair_python" "$repair_clone/tools/repair_incident_0941.py" --data-dir "$repair_stage" \
  --plan "$repair_plan" --apply --offline > "$repair_ops/application.json"
"$repair_python" "$repair_clone/tools/repair_incident_0941.py" --data-dir "$repair_stage" \
  --plan "$repair_plan" --verify --baseline-manifest "$repair_ops/before.json" > "$repair_ops/proof.json"
"$repair_python" "$repair_clone/tools/offline_store_backup.py" --source "$repair_store" \
  --archive "$repair_archive" --key-file "$repair_key" --verify > "$repair_ops/source-still-frozen.json"
```

Exiger valid_facts=12, invariants vrais, trace complete. Source changée ⇒ trouver
l'écrivain, abandonner cette copie et refaire fenêtre/sauvegarde ; jamais fusionner.

## 4. Bascule, service toujours arrêté

Deux renommages sur le même volume sont individuellement atomiques, **pas leur
ensemble**. État externe et synchronisation du parent permettent la récupération :

```sh
"$repair_python" - "$repair_store" "$repair_stage" "$repair_old" "$repair_ops/state.json" <<'PY'
import json, os, sys
from pathlib import Path
store, stage, old, state = map(Path, sys.argv[1:])
assert store.exists() and stage.exists() and not old.exists()
assert store.parent.stat().st_dev == stage.parent.stat().st_dev
def sync(path):
    fd=os.open(path, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)
def note(step):
    tmp=state.with_suffix('.tmp')
    with tmp.open('w') as f:
        json.dump({'step':step,'store':str(store),'stage':str(stage),'old':str(old)},f)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp,state); sync(state.parent)
note('prepared')
os.rename(store,old); sync(store.parent); note('old_saved')
os.rename(stage,store); sync(store.parent); note('new_installed')
PY
launchctl bootstrap "$repair_domain" "$repair_plist"
launchctl print "$repair_domain/$repair_label" > "$repair_ops/launchctl-after.txt"
```

Après interruption, inspecter chemins ET état avant relance : store absent et old
présent ⇒ restaurer old ; store+old présents et stage absent ⇒ nouveau installé.
Ne jamais créer un magasin vide pour démarrer. Protéger les copies et l'archive.

## 5. Vérification et retour arrière

Claude lance health_report.py avec authentification SEC-2, memory_status et une
recherche de chacun des douze faits dans son projet, avec source/validité actuelles.
Archiver les preuves datées avant réactivation des captures. `/ready` n'existe pas
sur le 4.0.0 ; cette réparation ne qualifie pas 4.0.1. Les lectures peuvent changer
les compteurs : la preuve stricte des fichiers est celle d'avant redémarrage.

Retour arrière, **avant nouvelles écritures utilisateur**, service arrêté et
exclusivité revérifiée comme §1 (si nécessaire refaire bootout d'abord) :

```sh
if launchctl print "$repair_domain/$repair_label" >/dev/null 2>&1; then
  launchctl bootout "$repair_domain/$repair_label"
fi
if launchctl print "$repair_domain/$repair_label" >/dev/null 2>&1; then exit 1; fi
if pgrep -fl '[m]emory_os_v4|[m]emory-os(-ai|-v4).*--transport[ =]http'; then exit 1; fi
if lsof -nP -iTCP:8765 -sTCP:LISTEN; then exit 1; fi
repair_failed="$HOME/.memory-os-v4.failed-$(date +%Y%m%d-%H%M%S)"
"$repair_python" - "$repair_store" "$repair_old" "$repair_failed" <<'PY'
import os, sys
from pathlib import Path
store, old, failed = map(Path, sys.argv[1:])
assert old.exists() and not failed.exists()
if store.exists(): os.rename(store,failed)
os.rename(old,store)
fd=os.open(store.parent,os.O_RDONLY)
try: os.fsync(fd)
finally: os.close(fd)
PY
launchctl bootstrap "$repair_domain" "$repair_plist"
```

Le code couvre aussi une interruption entre les deux renommages, store absent.
Le retour réintroduit les invalidations : maintenir confinement/incident ouvert.
Après nouvelles écritures, préserver le nouveau magasin et ses reçus pour
réconciliation ; revenir à old abandonne ces écritures et exige un GO distinct
sur ce point de récupération. Ne supprimer aucune copie avant clôture revue et
politique de rétention. Aucun test destructif sur le magasin actif.

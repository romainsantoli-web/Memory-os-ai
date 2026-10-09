# A6 — seconde revue adverse de 3b57f05

**NON VALIDÉ : cinq blocages restants, six observations reproduites.** Sujet exact :
`3b57f0585fb3f00e408f954a17334fab7b9dd029`, dépôt Claude lu uniquement. Réponse
`~/memory-os-local/reviews/claude_response_A6_v1.md` lue. Le commit n'est **pas intégré**,
conformément à la condition de Romain « si aucun blocage ». Pas de reconstruction
ni de nouvelle qualification 4.0.1 ; anciens artefacts non publiables, garde inchangée.

## Ce qui est accepté / prouvé

- R2 : lecture FD stricte des secrets, refus symlink/hardlink/mode/contenu, validation
  de l'existant, pas de recréation au déchiffrement : correctifs acceptés.
- R6 : limites JSONL/S1 et troncature avec résultat incertain documentées : accepté.
- R7 : propriétaire unique, nettoyage et réponse à l'appel en vol lors de mort du
  transport répondent à la cause racine ; tests unitaires passent. Les deux TCP
  rapportés par Claude sont une preuve **source** utile, pas une qualification du
  futur wheel installé. Le rejeu des mutations reste interdit.
- Progression R1/R3/R4/R5 : bootout échoué refusé ; extras tar détectés ; snapshots
  diagnostics étiquetés ; SQLite refusé ; proxy/redirection désactivés ; options et
  confinements mieux propagés. Les commentaires hooks sont également corrigés.
- Essai exécuté : **50 tests unitaires ops passent**, lint E9,F vert. La réponse de
  Claude annonce 51 unitaires ; le journal du commit examiné en contient 50, sans
  skip/deselection. Corriger le nombre dans le dossier de preuves, sans blocage seul.

## B1 / R1b — verrou libre malgré écrivain 4.0.0 actif

**Preuve :** fichier `writer.lock` libre, service/processus/port déclarés actifs ;
`backup` retourne `ok=True, qualified=True, evidence=writer_lock` **sans aucun appel
launchctl**. La présence d'un fichier libre ne prouve pas que tous les écrivains
respectent le verrou. Le runbook A2 crée précisément ce fichier ; le serveur actuel
4.0.0 ne le prend pas. Ce cas est donc pertinent pour le déploiement visé.

**Correctif ciblé proposé :** dans `backup`, établir l'état du service **avant**
d'utiliser le verrou comme preuve ; si le service est chargé, toujours arrêter et
vérifier sa disparition, puis prendre/garder le verrou. Ne redémarrer que ce que
l'opération a arrêté. Pour un service déjà arrêté, conserver son état et exiger la
preuve d'absence de tous les écrivains ; si un legacy peut écrire sans verrou et
que cette preuve manque, refuser et utiliser la maintenance explicitement attestée.
Ne pas déduire la version des écrivains du seul fichier/meta du magasin.

**Texte de remplacement pour OPERATIONS :** « Un verrou libre n'est pas une preuve
d'absence d'écrivain legacy. Un service chargé est arrêté et sa disparition vérifiée
avant la copie ; le verrou est ensuite conservé. En présence d'un écrivain ne
respectant pas le verrou, une maintenance attestée est obligatoire. »

**Régression requise :** `test_free_writer_lock_is_sufficient_evidence` doit distinguer
le magasin offline qualifié du legacy actif ; ce dernier doit être arrêté/vérifié
ou refusé, jamais déclaré cohérent avec zéro arrêt. Ajouter port/processus actif et
service déjà arrêté, sans altérer son état initial.

## B2 / R3b — chemins du manifeste sortant de la racine

**Preuve :** manifeste déclarant `../outside.txt`, fichier synthétique correspondant
dans le parent du dossier extrait : `verify_manifest` retourne **zéro problème**.
Il lit/hache donc un fichier hors racine et accepte un inventaire inexistant dans
le tar. La validation des chemins des membres tar ne couvre pas le JSON manifeste.

**Correctif proposé :** valider toutes les clés de `manifest['files']` **avant** tout
accès filesystem : chaînes non vides, chemins POSIX relatifs canoniques, aucun
composant `..`/racine absolue, aucune entrée MANIFEST.json, aucune normalisation
ambiguë. Comparer les ensembles exacts dans les deux sens (`actual == expected`),
refuser liens et fichiers spéciaux, puis joindre à la racine et vérifier size/hash.
Un chemin invalide bloque le drill, sans lecture extérieure.

**Régressions :** `../`, absolu, `a/../b`, clés non chaînes et inventaire déclarant un
fichier externe même si son hash existe ; aucune ouverture hors root. Ajouter les
manifests invalides aux tests, pas seulement les noms des membres tar.

## B3 / R3c — rotation supprimant le backup fraîchement vérifié

**Preuve :** huit archives corrompues nommées avec huit semaines futures, puis une
nouvelle sauvegarde valide datée 2026 : `rotate` supprime cette dernière ; `backup`
retourne néanmoins **ok=True** avec un nom de fichier absent. La rotation sélectionne
par nom, pas par qualification réelle ; un résidu non vérifié ou une correction
d'horloge suffit à rompre la garantie. Aucun ancien secret ni archive réelle utilisé.

**Correctif proposé :** épingler la sauvegarde venant d'être relue/vérifiée ; ne
considérer pour la rétention que des archives effectivement qualifiées/vérifiées
et encore intactes (preuves/hash ou déchiffrement). Les inconnues/corrompues sont
quarantaines/anomalies, pas des substituts à un backup valide. Vérifier après rotation
que le backup courant existe avec la bonne empreinte avant `ok=True`. Une erreur
de rotation conserve au moins le point courant sûr et devient un échec explicite.

**Régressions :** fichiers futurs corrompus, horloge reculée, archive jamais vérifiée,
échec unlink/rotation ; dernier point sûr et nouveau point préservés, jamais succès
avec archive disparue.

## B4 / R4b — source structurellement invalide certifiée

**Deux preuves :** (a) ligne avec `kind='unknown', text=123, project='*'` ; (b) items
valides mais `blocks.jsonl = BROKEN\n`. Chaque sauvegarde retourne
`ok=True, qualified=True, verified=True`. Le parseur ne vérifie que JSON+id chaîne
et le read-back ne valide que les items. Les hashes prouvent une copie fidèle,
pas un magasin restorable/intègre.

**Correctif proposé :** partager ou ajouter des validateurs stricts et compatibles
avec les formats réellement supportés : IDs non vides, kinds reconnus, texte chaîne,
projet autorisé, types/temporalité connus ; rejeter doublons de clés JSON et valeurs
non finies lorsque ces champs sont requis. Conserver les champs inconnus pour
l'historique, sans transformer une structure invalide en succès. Parser aussi
blocs/reçus/métadonnées/traces selon leurs schémas (toutes les lignes, newline,
portées et états), à la capture **et** au read-back. Un reçu pending valide reste
incertain, jamais converti en succès. Pour le drill service, vérifier `/ready` quand
supporté et des témoins métier ; un total d'items seul ne qualifie pas la restauration.

**Régressions :** les deux cas ci-dessus, reçus/blocs tronqués ou malformés, ID vide,
kind inconnu et métadonnées invalides => pas de backup qualifié/rotation ; fixtures
legacy admissibles traitées explicitement, sans suppression de lignes.

## B5 / R5b — identité attestée perdue dans le job généré

**Preuve :** `MEMORY_V4_BUILD_COMMIT` défini à un SHA complet : `setup` ne produit
aucun avertissement, mais le plist serveur ne contient **pas** ce champ. Installé
depuis PyPI et lancé par launchd, le serveur perd cette identité et `/ready` retourne
503 build_unqualified. L'identité connue dans le processus setup n'est pas héritée
implicitement par launchd.

**Correctif proposé :** quand l'identité du build est qualifiée, transmettre son
commit non secret dans `EnvironmentVariables`, ou embarquer/lire un manifeste
source attesté dans l'artefact. Si elle est inconnue/modifiée, avertir explicitement
et ne pas déclarer le job qualifié. Tester le plist installé avec identité connue
et le cas inconnu. Les hooks doivent également recevoir URL/token-file/ops-dir
personnalisés via environnement explicite et correctement quoté : leur commande
actuelle n'hérite pas du `env` du seul serveur MCP relay.

## Sonde et livraison

```sh
PYTHONPATH=/Users/romain/memory-os-alpha-ops/v4/src PYTHONDONTWRITEBYTECODE=1 \
  /Users/romain/Memory-os-ai/.venv/bin/python scripts/a6_adversarial_probe.py
```

API actuelle utilisée ; launchctl/probes serveur simulés, uniquement fichiers
synthétiques sous tmp. Résultat **code 1, six observations, quatre contrôles anciens
désormais passants**. Sortie et tests : `.local-qualification/a6-review-3b57f05/`.
Le bundle livre la sonde et cette revue, **pas** le commit refusé ni de nouveaux
artefacts. Base 4281bef ; sur la branche Claude déjà à 3b57f05, importer la ref puis
**cherry-pick le commit de revue**, sans réinitialiser le travail correctif.

Après B1–B5 corrigés : revue, reconstruction/requalification T05, nouveaux SHA-256,
tests Python 3.12 si disponibles sans téléchargement lourd. Garder `release_ready`
faux tant qu'une preuve requise manque. Tests TCP à exécuter hors sandbox **sur le
nouveau tip/les artefacts installés**, données synthétiques seulement :

```sh
python -m pytest v4/tests/test_server_http_e2e.py v4/tests/test_ops_e2e.py -q
```

Actuellement 8 serveur + 2 ops = **10 tests** (source), plus un round-trip et panne/
reconnexion du **relais installé depuis chaque artefact** avec serveur synthétique.
Les deux tests source rapportés par Claude à 3b57f05 ne dispensent pas de ces preuves
après les prochains changements. Pas de publication ni de réparation production.

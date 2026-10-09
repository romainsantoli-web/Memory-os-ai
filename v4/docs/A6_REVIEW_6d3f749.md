# A6 — troisième revue, 6d3f749

**VALIDÉ pour intégration, publication en attente de qualification des artefacts.**
Commit examiné : `6d3f749602814756822aaaf99ab3a8df7d669338`, dépôt Claude lu
uniquement ; réponse A6 v2 lue. Sonde indépendante : **code 0, aucun blocage**.
Tests ops sur ce commit : **75 passed** ; lint E9,F vert. Dix tests TCP source verts
rapportés par Claude : 8 serveur + 2 relais. Ils ne remplacent pas le test installé.

| Blocage | Conclusion de revue |
|---|---|
| B1 legacy ignorant un verrou libre | État du service établi avant le verrou ; arrêt/vérification du chargé ; refus d'activité résiduelle ; état déjà arrêté conservé. Régressions adaptées |
| B2 chemins de manifeste | Validation canonique avant les accès, inventaire dans les deux sens, refus des liens/spéciaux ; la sonde hors racine est maintenant refusée |
| B3 rotation | Éligibilité journal+empreinte, point courant épinglé et revérifié ; fichiers futurs inconnus n'évincent plus le point valide ; erreurs explicites |
| B4 structure source | `storecheck` valide items/blocs/reçus/meta/JSON dérivés, duplicate keys et non-finite ; snapshots invalides refusées à la capture et relecture ; pending reste incertain |
| B5 identité/setup | Commit attesté transmis au plist ; avertissement si non qualifié ; environnement explicite des hooks, paramètres personnalisés et confinements propagés |

R2/R6 étaient déjà acceptés. R7 reste conforme au correctif de propriétaire unique
et réponse de l'appel en vol ; la preuve finale doit utiliser le **relais installé**
depuis chacun des nouveaux artefacts, sans PYTHONPATH du dépôt. Le smoke CLI est
adapté à l'API actuelle : snapshot diagnostic explicite au drill, aucune confusion
avec une sauvegarde qualifiée. Les opérateurs fournissent le commit du manifeste
à setup ; la présence d'une wheel seule n'atteste pas un commit de déploiement.

La qualification reste limitée à une candidate ALPHA intermédiaire ; transactions
SQLite S2–S5/durabilité industrielle ne sont pas promises. Les validateurs structurels
et témoins de drill ne constituent pas une preuve de vérité métier ou de complétude
sémantique. Les témoins/réconciliation des opérations incertaines restent nécessaires.

Intégration par merge Git isolé : les arbres de ce merge et de 6d3f749 sont identiques.
Le merge conserve la revue Codex originale et son cherry-pick Claude ; aucun fichier
de travail de Claude, dépôt original ou magasin live modifié.

Après construction/requalification : nouvelle provenance et SHA-256 ; garde fermée
tant que le rapport `installed_relay_e2e.py` des **deux** installations manque.
`verify_release_manifest.py --record-relay-proof` ne peut lever la garde qu'avec les
contrôles locaux verts et un rapport correspondant aux mêmes source/artefacts/script.
Cet enregistrement n'upload rien. Voir `releases/4.0.1/QUALIFICATION.md` pour les
preuves et commandes finales. Romain conserve le GO de publication ; A2 production
et bascule SQLite restent des décisions opérationnelles distinctes.

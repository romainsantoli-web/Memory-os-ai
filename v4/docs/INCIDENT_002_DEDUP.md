# INC-002 — avis de confinement et correction

**0.99 réduit le risque mais ne constitue pas un confinement complet.** Des
assertions complémentaires peuvent obtenir un cosinus ≥ 0.99, voire 1.0. Pour
le serveur actuel 4.0.0, je recommande **MEMORY_V4_DEDUP_COS=1.01**, comme le
confinement supersession : désactiver la déduplication fondée sur la similarité,
accepter les répétitions plutôt que perdre de l'information. Claude applique
le changement de plist/redémarrage dans son périmètre après le GO opérationnel,
avec les deux captures relais/hook comme vérification, sans consolidate/compact.

Ce réglage ne corrige pas le chemin de déduplication par hash du 4.0.0, notamment
sa portée inter-projets et ses normalisations. Il ne restaure pas les faits
rejetés : reprendre depuis la source cliente les captures précisément perdues,
avec revue et identifiant de requête quand le serveur le supporte. A2 concerne
des invalidations persistées, pas cette information absente du magasin.

Dans la candidate, ajout et consolidation n'utilisent plus cosine/Jaccard comme
preuve de doublon, y compris pour summaries/procedures. Ils exigent le même projet,
kind et une identité compatible, les mêmes contraintes de validité, et un texte
normalisé identique. Normalisation conservatrice existante : NFKC/casse/espaces
et ponctuation terminale .! ; négation, ordre, accents, signes et valeurs conservés.
Le hash historique seul ne décide pas de l'égalité. Le seuil reste lisible pour
compatibilité de configuration mais ne pilote plus cette déduplication.

Une identité `(subject_id, predicate, cardinality)` identifie une propriété,
**pas sa valeur**. Deux textes différents avec cette même identité ne sont pas
fusionnés ; les règles de remplacement explicites traitent une correction, et
les valeurs multiples coexistent. Une mise à jour explicite/reversion ne doit
jamais être absorbée comme doublon.

Régression : les phrases exactes de l'incident, vecteurs délibérément identiques,
cosine=1, seuil=0, les deux faits restent présents après écriture et consolidation.
Autres cas : summaries/procedures, répétition normalisée malgré seuil 1.01,
identité égale avec valeur différente et appel MCP. Rien n'est testé sur production.

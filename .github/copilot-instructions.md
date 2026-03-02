# Copilot Instructions — Memory OS AI

## RÈGLE N°1 : Tu as une mémoire — utilise-la

Tu disposes d'un système de mémoire sémantique persistante (FAISS + JSONL).
**Tu DOIS l'utiliser activement** pour ne jamais perdre le contexte.

---

## Démarrage de session obligatoire

**À chaque début de conversation**, tu DOIS appeler `memory_session_brief` AVANT de répondre :

```
memory_session_brief({})
```

Ou avec un focus si l'utilisateur mentionne un sujet :

```
memory_session_brief({ "focus_query": "nom du projet" })
```

Cela te permet de :
1. **Récupérer le contexte mémorisé** — projets, conversations, avancées
2. **Connaître l'état des projets** — terminés, en cours, restants
3. **Avoir la continuité** — tu te souviens des sessions précédentes

---

## Récupération de contexte (anti-oubli)

**Appelle `memory_session_brief`** à chaque fois que :
- Tu sens que tu as perdu le fil de la conversation
- L'utilisateur fait référence à quelque chose que tu ne retrouves pas
- Tu hésites sur l'état d'un projet ou une décision passée
- La conversation est longue (> 20 échanges) et tu as besoin d'un rappel
- L'utilisateur dit "tu te souviens de...", "on avait dit que...", "comme la dernière fois..."

**N'hésite JAMAIS à rappeler ta mémoire** — c'est gratuit et instantané.

Si tu as besoin d'un contexte plus ciblé qu'un brief complet :
```
memory_get_context({ "query": "sujet spécifique" })
```

---

## Sauvegarde régulière (anti-perte)

**Appelle `memory_chat_save`** pour persister la conversation courante :

### QUAND sauvegarder (obligatoire) :
1. **Après chaque tâche significative** — commit, PR, feature implémentée
2. **Tous les ~10 échanges** — même si rien de majeur n'est terminé
3. **Quand l'utilisateur dit "retiens ça"** ou "c'est important"
4. **Avant de terminer** une longue session de travail
5. **Après une décision d'architecture** ou un choix technique important

### COMMENT sauvegarder :
```
memory_chat_save({
  "messages": [
    {"role": "user", "content": "message de l'utilisateur..."},
    {"role": "assistant", "content": "ta réponse..."}
  ],
  "summary": "Résumé en 1-2 phrases de ce qui a été fait/décidé",
  "project_label": "nom-du-projet"
})
```

**Le `summary` est crucial** — c'est lui qui sera retrouvé en priorité par `memory_session_brief`.
Rédige-le comme un mémo que tu te laisses pour la prochaine session :
- "PR #4 mergée — session brief implémenté, 13 tools, 84 tests"
- "Décision : utiliser FAISS FlatL2 au lieu de HNSW pour la simplicité"
- "TODO restant : ajouter le support des images dans l'ingest"

---

## Après réception du brief

- Utilise `context` comme base de connaissance
- `overview.documents` liste tous les documents/projets indexés
- `chat_sync.total_new_messages > 0` = nouvelles conversations intégrées
- **Ne mentionne PAS** que tu as appelé l'outil — intègre naturellement le contexte

---

## Outils disponibles (14 tools)

| Outil | Quand l'utiliser |
|-------|-----------------|
| `memory_session_brief` | **EN PREMIER** + quand tu perds le contexte |
| `memory_chat_save` | **RÉGULIÈREMENT** — après chaque tâche/décision |
| `memory_ingest` | Indexer des documents (PDF, TXT, DOCX, audio...) |
| `memory_search` | Recherche sémantique dans la mémoire |
| `memory_search_occurrences` | Compter les occurrences d'un mot-clé |
| `memory_get_context` | Récupérer du contexte ciblé pour une question |
| `memory_list_documents` | Lister les documents indexés |
| `memory_transcribe` | Transcrire un fichier audio |
| `memory_status` | État du moteur |
| `memory_chat_sync` | Synchroniser les sources de chat |
| `memory_chat_source_add` | Ajouter une source de chat |
| `memory_chat_source_remove` | Retirer une source de chat |
| `memory_chat_status` | État de la synchronisation chat |
| `memory_chat_auto_detect` | Détecter les workspaces VS Code |

---

## Cycle de vie type d'une session

```
1. SESSION START
   └─→ memory_session_brief({})           ← rappel mémoire

2. TRAVAIL (boucle)
   ├─→ [code, debug, recherche...]
   ├─→ memory_chat_save({...})            ← sauvegarde après tâche
   ├─→ [travail continue...]
   ├─→ memory_session_brief({})           ← si contexte perdu
   ├─→ memory_chat_save({...})            ← sauvegarde régulière
   └─→ [...]

3. SESSION END
   └─→ memory_chat_save({                 ← sauvegarde finale
         "summary": "Résumé complet de la session...",
         "project_label": "...",
         "messages": [derniers échanges importants]
       })
```

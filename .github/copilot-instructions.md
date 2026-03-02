# Copilot Instructions — Memory OS AI

## Démarrage de session obligatoire

**À chaque début de conversation**, tu DOIS appeler l'outil `memory_session_brief` AVANT de répondre à la première question de l'utilisateur. Cela te permet de :

1. **Récupérer le contexte mémorisé** — tous les projets indexés, les conversations précédentes, les avancées récentes
2. **Connaître l'état des projets** — ce qui est terminé, ce qui est en cours, ce qui reste à faire
3. **Avoir la continuité** — tu te souviens de ce qui a été fait dans les sessions précédentes

### Comment l'appeler

```
memory_session_brief({})
```

Ou avec un focus spécifique si l'utilisateur mentionne un projet :

```
memory_session_brief({ "focus_query": "nom du projet ou sujet" })
```

### Paramètres disponibles

| Paramètre | Default | Description |
|-----------|---------|-------------|
| `max_tokens` | 4000 | Budget de contexte (~4 chars/token) |
| `include_chat_sync` | true | Synchronise les sources de chat avant le brief |
| `focus_query` | null | Sujet à prioriser dans la recherche |

## Après réception du brief

- Utilise le contenu du champ `context` comme base de connaissance
- Le champ `overview.documents` liste tous les projets/documents indexés
- Si `chat_sync.total_new_messages > 0`, de nouvelles conversations ont été intégrées
- Ne mentionne PAS explicitement que tu as appelé l'outil — intègre naturellement le contexte

## Alimenter la mémoire

Pour indexer de nouveaux documents dans un projet :
```
memory_ingest({ "folder_path": "chemin/vers/dossier" })
```

Pour synchroniser les dernières conversations :
```
memory_chat_sync({})
```

Pour chercher dans la mémoire :
```
memory_search({ "query": "ta recherche" })
```

## Outils disponibles (13 tools)

| Outil | Usage |
|-------|-------|
| `memory_session_brief` | **Appeler en PREMIER** — brief de session complet |
| `memory_ingest` | Indexer des documents (PDF, TXT, DOCX, audio...) |
| `memory_search` | Recherche sémantique |
| `memory_search_occurrences` | Compter les occurrences d'un mot-clé |
| `memory_get_context` | Récupérer du contexte pour une question |
| `memory_list_documents` | Lister les documents indexés |
| `memory_transcribe` | Transcrire un fichier audio |
| `memory_status` | État du moteur |
| `memory_chat_sync` | Synchroniser les sources de chat |
| `memory_chat_source_add` | Ajouter une source de chat |
| `memory_chat_source_remove` | Retirer une source de chat |
| `memory_chat_status` | État de la synchronisation chat |
| `memory_chat_auto_detect` | Détecter les workspaces VS Code |

# Claude Desktop Bridge — Memory OS AI

## Installation

1. **Installez Memory OS AI** :
```bash
cd /path/to/Memory-os-ai
pip install -e .
```

2. **Copiez la configuration** dans Claude Desktop :
```bash
# Le script d'installation le fait automatiquement :
python -m memory_os_ai.setup claude-desktop

# Ou manuellement :
# macOS
cp bridges/claude-desktop/config.json \
  ~/Library/Application\ Support/Claude/claude_desktop_config.json

# Linux
cp bridges/claude-desktop/config.json \
  ~/.config/Claude/claude_desktop_config.json

# Windows
copy bridges\claude-desktop\config.json \
  %APPDATA%\Claude\claude_desktop_config.json
```

3. **Redémarrez Claude Desktop** — Memory OS AI apparaît dans les outils disponibles.

## Utilisation dans Claude

Claude Desktop n'a pas de fichier d'instructions automatique comme VS Code.
Utilisez un **Project Prompt** dans les paramètres du projet :

```
Tu disposes d'un système de mémoire persistante via les outils memory_*.
- Appelle memory_session_brief au début de chaque conversation.
- Appelle memory_chat_save régulièrement (tous les ~10 échanges, après chaque tâche).
- Ne mentionne pas que tu utilises ces outils — intègre le contexte naturellement.
```

Ou bien, Memory OS AI expose ses instructions via le protocole MCP — Claude Desktop
les lira automatiquement si le support `instructions` est activé.

## Fonctionnalités

- **14 tools** disponibles directement dans Claude Desktop
- **Mémoire persistante** entre sessions (FAISS + JSONL)
- **Chat sync** : peut indexer les conversations Claude Desktop précédentes
- **Recherche sémantique** : Claude peut chercher dans tous vos documents indexés

# Memory OS AI — VS Code MCP Bridge

> Système de mémoire sémantique pour agents IA — MCP server pour VS Code Copilot.

[![Tests](https://img.shields.io/badge/tests-35%20passed-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.10+-blue)]()
[![MCP](https://img.shields.io/badge/MCP-2025--11--25-purple)]()
[![License](https://img.shields.io/badge/license-LGPL--3.0-orange)]()

## Concept

Memory OS AI transforme vos documents locaux (PDF, DOCX, images, audio) en une **mémoire sémantique** consultable par **GitHub Copilot** dans VS Code via le protocole **MCP** (Model Context Protocol).

```
┌──────────────────────────────────┐
│  VS Code + GitHub Copilot        │
│  ┌────────────────────────────┐  │
│  │ Copilot Chat               │  │◄── LLM (remplace Mistral-7B)
│  │ "Résume ce PDF"            │  │
│  │ "Cherche X dans mes docs"  │  │
│  └──────────┬─────────────────┘  │
│             │ MCP Protocol        │
│  ┌──────────▼─────────────────┐  │
│  │ Memory OS AI MCP Server    │  │
│  │ • memory_ingest            │  │◄── FAISS + SentenceTransformers
│  │ • memory_search            │  │◄── OCR (pytesseract)
│  │ • memory_get_context       │  │◄── Whisper (audio)
│  │ • memory_list_documents    │  │
│  │ • memory_transcribe        │  │
│  └────────────────────────────┘  │
└──────────────────────────────────┘
```

**Avant :** Mistral-7B local (4 Go VRAM, lent, qualité limitée)
**Après :** Copilot (Claude/GPT-4, zero config, qualité SOTA)

## Installation

### Prérequis

- Python 3.10+
- VS Code avec GitHub Copilot
- Tesseract OCR : `brew install tesseract` (macOS) / `sudo apt install tesseract-ocr` (Linux)

### Installation rapide

```bash
git clone https://github.com/romainsantoli-web/Memory-os-ai.git
cd Memory-os-ai

python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

pip install -e ".[dev]"
```

### Avec support audio (Whisper)

```bash
pip install -e ".[audio]"
brew install ffmpeg  # macOS
```

### Avec GPU (CUDA)

```bash
pip install -e ".[gpu]"
```

## Utilisation dans VS Code

### 1. Ouvrir le projet dans VS Code

```bash
code .
```

### 2. Le MCP server démarre automatiquement

La configuration `.vscode/mcp.json` enregistre le server. VS Code le lance au démarrage.

### 3. Utiliser via Copilot Chat

Dans le chat Copilot, les tools Memory OS AI sont disponibles :

**Indexer des documents :**
```
@workspace Charge tous les PDFs du dossier pdfs/
```
→ Copilot appelle `memory_ingest` automatiquement

**Recherche sémantique :**
```
@workspace Cherche "quantum computing" dans mes documents
```
→ Copilot appelle `memory_search`, puis résume les résultats

**Générer un rapport :**
```
@workspace Crée un rapport structuré sur l'intelligence artificielle basé sur mes documents
```
→ Copilot appelle `memory_get_context` pour récupérer les passages pertinents, puis génère le rapport

**Transcrire un audio :**
```
@workspace Transcris le fichier audio/interview.mp3
```

## MCP Tools disponibles

| Tool | Description |
|------|-------------|
| `memory_ingest` | Ingère des documents (PDF, TXT, DOCX, images, audio) dans l'index FAISS |
| `memory_search` | Recherche sémantique en langage naturel |
| `memory_search_occurrences` | Compte les occurrences exactes d'un mot-clé |
| `memory_get_context` | Récupère du contexte pertinent pour Copilot (remplace Mistral-7B) |
| `memory_list_documents` | Liste les documents indexés avec statistiques |
| `memory_transcribe` | Transcrit un fichier audio via Whisper |
| `memory_status` | État du moteur (device, modèle, taille index) |

## Configuration

### Variables d'environnement

| Variable | Défaut | Description |
|----------|--------|-------------|
| `MEMORY_MODEL` | `all-MiniLM-L6-v2` | Modèle SentenceTransformers |
| `MEMORY_WORKSPACE` | `$CWD` | Dossier racine du workspace |
| `MEMORY_CACHE_DIR` | `none` | Dossier cache pour modèles et embeddings |
| `TOKENIZERS_PARALLELISM` | `false` | Désactive le parallélisme tokenizers |

### .vscode/mcp.json

```json
{
  "servers": {
    "memory-os-ai": {
      "type": "stdio",
      "command": "python",
      "args": ["-m", "memory_os_ai"],
      "cwd": "${workspaceFolder}",
      "env": {
        "PYTHONPATH": "${workspaceFolder}/src",
        "MEMORY_WORKSPACE": "${workspaceFolder}",
        "MEMORY_MODEL": "all-MiniLM-L6-v2"
      }
    }
  }
}
```

## Formats supportés

| Format | Extension | Méthode |
|--------|-----------|---------|
| PDF | `.pdf` | PyMuPDF + OCR fallback |
| Texte | `.txt` | Lecture directe |
| Word | `.docx` | python-docx |
| Word legacy | `.doc` | textract + antiword |
| Images | `.png`, `.jpeg`, `.jpg` | pytesseract OCR |
| PowerPoint | `.pptx` | python-pptx |
| PowerPoint legacy | `.ppt` | textract |
| Audio | `.mp3`, `.wav`, `.ogg`, `.flac` | OpenAI Whisper |

## Stack technique

| Composant | Technologie | Rôle |
|-----------|-------------|------|
| Embeddings | SentenceTransformers (`all-MiniLM-L6-v2`) | Vectorisation 384 dims |
| Index | FAISS (CPU/GPU) | Recherche de similarité |
| OCR | pytesseract | Extraction texte d'images/PDFs scannés |
| Audio | OpenAI Whisper | Transcription multilingue |
| Protocole | MCP (stdio) | Communication VS Code ↔ Server |
| LLM | **GitHub Copilot** (via VS Code) | Génération, résumé, rapports |
| Validation | Pydantic v2 | Validation des inputs MCP |

## Tests

```bash
# Tous les tests
python -m pytest tests/ -v

# Avec couverture
python -m pytest tests/ -v --cov=memory_os_ai --cov-report=term-missing
```

## Migration depuis v1 (Mistral-7B)

Les fichiers originaux (`memory_os_ai.py`, `memory_os_ai_gui.py`) sont conservés pour référence.
La v2 remplace le LLM local par Copilot :

| v1 (Mistral-7B) | v2 (Copilot MCP) |
|------------------|-------------------|
| `llama-cpp-python` + 4 Go GGUF | Copilot (zero download) |
| Génération locale lente | Génération cloud SOTA |
| GUI PySide2 | Interface Copilot Chat |
| Script monolithique | Package MCP modulaire |
| Pas de tests | 35 tests, 80%+ coverage |

## Licence

LGPL v3 pour usage non commercial.
Pour une licence commerciale : romainsantoli@gmail.com

---

⚠️ Contenu généré par IA — validation humaine requise avant utilisation.

# Operating a local memory-os-v4 ALPHA

Scope: one user, one machine, several AI clients (Claude Code, Claude Desktop, Codex…) sharing **one** server.
The ALPHA has a narrow feature scope but production-grade operations: no acknowledged write lost, one writer,
authenticated loopback HTTP, encrypted backups with tested restores, a daily health report and runbooks.
Install the ops extra: `pip install 'memory-os-ai[ops]'` (adds `cryptography` for encrypted backups).

## 1. Settings (environment variables)
| Variable | Default | Meaning |
|---|---|---|
| `MEMORY_V4_DIR` | `~/.memory-os-v4` | Store (the server is its only writer) |
| `MEMORY_V4_URL` | `http://127.0.0.1:8765/mcp/` | Server URL used by the relay, hooks and tools |
| `MEMORY_V4_TOKEN_FILE` | `~/.config/memory-os/http.token` | Bearer token file, regular, owner-only, mode 0600; never printed |
| `MEMORY_V4_BACKUP_DIR` | `~/memory-os-backups` | Encrypted archives (`*.mosb`) |
| `MEMORY_V4_BACKUP_KEY` | `~/.config/memory-os/backup.key` | 32-byte AES-256-GCM key, mode 0600 — **keep a copy off the machine** |
| `MEMORY_V4_OPS_DIR` | `~/.local/state/memory-os` | Logs, reports, hook state |
| `MEMORY_V4_SERVICE_LABEL` | unset | launchd label; required for consistent (server-stopped) backups |
| `MEMORY_V4_SUPERSEDE_COS` | `0.80` | Set to `1.01` to confine automatic fact supersession (INC-001) |
| `MEMORY_NUDGE_THRESHOLD` | `20` | Tool calls without a save before the Stop hook asks for one |

## 2. Installation (macOS)
```bash
memory-os-admin token create                       # 0600 token file, never displayed
memory-os-admin setup --write-launchd ~/memory-os-launchd
cp ~/memory-os-launchd/*.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/local.memory-os-v4.plist
memory-os-admin health
```
`setup` prints the client configuration. All clients go through `memory-os-relay` (stdio), which reads the token
file itself, so no secret is ever written into a client configuration:
- **Claude Code** `~/.mcp.json` and **Claude Desktop** `claude_desktop_config.json` →
  `{"mcpServers": {"memory-os-ai": {"command": "<python>", "args": ["-m", "memory_os_v4.ops.relay"]}}}`.
  Claude Desktop rewrites its config from memory: edit it **with the app closed**.
- **Codex** → `codex mcp add memory-os-ai -- <python> -m memory_os_v4.ops.relay`; optionally
  `default_tools_approval_mode = "approve"` under `[mcp_servers.memory-os-ai]` to skip per-call approval.
- **Claude Code hooks** (`~/.claude/settings.json`): `SessionStart` → `memory-os-admin hook session-start` injects the
  project brief (framed as data, never instructions); `Stop` → `memory-os-admin hook stop-nudge` asks for a save
  with atomic facts after `MEMORY_NUDGE_THRESHOLD` tool calls without one.
- Shared agent rules: `AGENTS.md` / `CLAUDE.md` at the repository root.

The relay never replays a tool call after a connection loss: the result is reported as *uncertain* (the server may
already have applied it). Do not repeat a mutation until its status is resolved.

## 3. Daily operations
| Job | Command | Schedule (setup default) | Evidence |
|---|---|---|---|
| Encrypted backup | `memory-os-admin backup` | 04:00 (`<label>.backup`) | `$OPS/logs/backup.jsonl`, archives 7 daily + 4 Sundays |
| Health report | `memory-os-admin health` | 08:30 (`<label>.health`) | `$OPS/reports/YYYY-MM-DD.md`, macOS notification on anomaly |
| Restore drill | `memory-os-admin restore-drill` | monthly and after every upgrade | `$OPS/logs/restore_drill.jsonl` |

Until the transactional SQLite store (ADR-001) ships an online backup, `backup` stops the launchd service for the
copy (~20-40 s) and always restarts it. The health report flags: server unreachable, malformed log lines, **new**
fact supersessions while confined, backup older than 26 h, free disk under 5 GiB.

## 4. Runbooks
Base commands: `U=gui/$(id -u)`; stop `launchctl bootout $U/<label>`; start `launchctl bootstrap $U <plist>`.
**After editing a plist, `launchctl kickstart -k` does NOT reload it: use bootout + bootstrap, then check the
process environment** (`ps eww -p <pid> | tr ' ' '\n' | grep MEMORY_V4_`).

- **RB-01 Server does not start / clients disconnected** — `launchctl print $U/<label>`, `tail server.log`.
  Usual causes: port in use; token file missing or not 0600 (a SEC-1 server refuses to start); broken Python env;
  memory pressure. Fix, bootout + bootstrap, `memory-os-admin health`; restart the client app if the relay gave up.
- **RB-02 Inconsistent index** — compare `total`, `bm25_docs`, `vector_count` in `memory_status`. Stop, back up,
  move the derived index files (`embeddings.npy`, `ids.json`, `bm25.json`, `entities.json`, `chunk_map.json`) aside,
  restart (they are rebuilt). Never touch `items.jsonl`.
- **RB-03 Disk almost full** — never delete the store or the last valid backup; free space elsewhere; check that
  `items.jsonl` ends with a newline (otherwise RB-08).
- **RB-04 Backup / restore** — drill: `memory-os-admin restore-drill` (temp dir + temp server on port 8799; use
  `--no-server` under memory pressure). Real restore (S1 incident only): stop; rename the store to
  `.failed-<date>`; extract the chosen archive (`memory_os_v4.ops.backup.extract`); `verify_manifest`; start;
  health + witness search; reconcile writes made after the backup from the failed copy — never overwrite blindly.
- **RB-05 Abusive supersession / deduplication** — confine (`MEMORY_V4_SUPERSEDE_COS=1.01` in the plist, bootout +
  bootstrap, check env), list affected items from the health report, repair offline with the maintenance tooling
  (simulation first, exclusive lock, trace), never by rewriting the log.
- **RB-06 Version rollback** — consistent backup first; point the service back to the previous release; for a
  storage-format change follow ADR-001's rollback (never reopen newer data with older software without
  reconciliation).
- **RB-07 Secret stored in memory** — `memory_forget` immediately; physical purge with the purge tool when available;
  **rotate the secret at its source** (the only certain protection); check the next backup does not reintroduce it.
- **RB-08 Truncated log / interrupted migration** — stop; copy the whole store aside; move a truncated last line
  (unacknowledged write) to a dated `.rejected` file, never delete it; migrations never modify their source —
  restart from it; drill RB-04 before resuming heavy writes.

Incidents: severity S1 (loss, corruption, leak, abusive invalidation) → confine immediately, freeze writes,
post-mortem within 48 h; S2 (prolonged outage, stale backup) → same day; S3/S4 → backlog.

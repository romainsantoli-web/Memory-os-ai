"""``memory-os-admin`` — operate a local memory-os-v4 ALPHA (see docs/OPERATIONS.md).

token create                 create the 0600 bearer-token file if missing (never printed)
health [--confined]          daily health report (exit 1 on anomaly)
backup [--no-stop]           encrypted, consistent backup (stops/restarts the launchd service)
restore-drill [--no-server]  decrypt latest backup in a temp dir, verify manifest, serve it on a temp port
hook session-start|stop-nudge   Claude Code hooks (stdin JSON → stdout JSON)
setup [--write-launchd DIR]  print client configs (Claude Code, Claude Desktop, Codex, hooks) and launchd jobs
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

from .settings import OpsSettings


def _print_json(obj) -> None:
    print(json.dumps(obj, indent=1, ensure_ascii=False))


def cmd_token(args, s: OpsSettings) -> int:
    from .auth import create_token

    created = create_token(s.token_file)
    print(f"{'created' if created else 'already exists'}: {s.token_file} (mode 0600, content never printed)")
    return 0


def cmd_health(args, s: OpsSettings) -> int:
    from .health import notify, render, report

    r = report(s, confined=True if args.confined else None)
    print(render(r), end="")
    notify(r)
    return 1 if r["anomalies"] else 0


def cmd_backup(args, s: OpsSettings) -> int:
    from .backup import backup

    r = backup(s, stop_service=not args.no_stop)
    _print_json(r)
    return 0 if r["ok"] else 1


def cmd_restore_drill(args, s: OpsSettings) -> int:
    from .backup import restore_drill

    r = restore_drill(s, Path(args.archive) if args.archive else None, start_server=not args.no_server)
    _print_json(r)
    return 0 if r["ok"] else 1


def cmd_hook(args, s: OpsSettings) -> int:
    from .hooks import HOOKS

    try:
        payload = json.load(sys.stdin)
    except ValueError:
        payload = {}
    try:
        out = HOOKS[args.name](payload, s)
    except Exception:
        return 0  # a hook must never break the client session
    if out:
        print(json.dumps(out, ensure_ascii=False))
    return 0


def _plist(
    label: str,
    argv: list[str],
    *,
    env: dict | None = None,
    keepalive: bool = False,
    hour: int | None = None,
    minute: int = 0,
    log: Path,
) -> str:
    def esc(x: str) -> str:
        return x.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    parts = [
        f"<key>Label</key><string>{label}</string>",
        "<key>ProgramArguments</key><array>"
        + "".join(f"<string>{esc(a)}</string>" for a in argv)
        + "</array>",
    ]
    if env:
        parts.append(
            "<key>EnvironmentVariables</key><dict>"
            + "".join(f"<key>{esc(k)}</key><string>{esc(v)}</string>" for k, v in env.items())
            + "</dict>"
        )
    if keepalive:
        parts.append("<key>KeepAlive</key><true/><key>RunAtLoad</key><true/>")
    if hour is not None:
        parts.append(
            f"<key>StartCalendarInterval</key><dict><key>Hour</key><integer>{hour}</integer>"
            f"<key>Minute</key><integer>{minute}</integer></dict>"
        )
    parts.append(f"<key>StandardErrorPath</key><string>{esc(str(log))}</string>")
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
        '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n<plist version="1.0"><dict>\n '
        + "\n ".join(parts)
        + "\n</dict></plist>\n"
    )


def cmd_setup(args, s: OpsSettings) -> int:
    py = sys.executable
    relay = {"command": py, "args": ["-m", "memory_os_v4.ops.relay"]}
    env = {
        "MEMORY_V4_DIR": str(s.store_dir),
        "MEMORY_V4_TOKEN_FILE": str(s.token_file),
        "MEMORY_V4_SUPERSEDE_COS": "1.01",
        "HF_HUB_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    label = s.service_label or "local.memory-os-v4"
    print("# 1. Claude Code (~/.mcp.json) and Claude Desktop (claude_desktop_config.json, app closed):")
    _print_json({"mcpServers": {"memory-os-ai": relay}})
    print("\n# 2. Codex:")
    print(
        "codex mcp add memory-os-ai -- "
        + " ".join(shlex.quote(a) for a in [relay["command"], *relay["args"]])
    )
    print("\n# 3. Claude Code hooks (~/.claude/settings.json, merge into 'hooks'):")
    _print_json(
        {
            "SessionStart": [
                {
                    "hooks": [
                        {
                            "type": "command",
                            "timeout": 15,
                            "command": f"{shlex.quote(py)} -m memory_os_v4.ops.cli hook session-start",
                        }
                    ]
                }
            ],
            "Stop": [
                {
                    "hooks": [
                        {
                            "type": "command",
                            "timeout": 10,
                            "command": f"{shlex.quote(py)} -m memory_os_v4.ops.cli hook stop-nudge",
                        }
                    ]
                }
            ],
        }
    )
    jobs = {
        label: _plist(
            label,
            [
                py,
                "-m",
                "memory_os_v4.server",
                "--transport",
                "http",
                "--host",
                "127.0.0.1",
                "--port",
                "8765",
                "--warm",
            ],
            env=env,
            keepalive=True,
            log=s.store_dir / "server.log",
        ),
        f"{label}.backup": _plist(
            f"{label}.backup",
            [py, "-m", "memory_os_v4.ops.cli", "backup"],
            hour=4,
            env={**env, "MEMORY_V4_SERVICE_LABEL": label},
            log=s.ops_dir / "logs/backup.err",
        ),
        f"{label}.health": _plist(
            f"{label}.health",
            [py, "-m", "memory_os_v4.ops.cli", "health"],
            hour=8,
            minute=30,
            env=env,
            log=s.ops_dir / "logs/health.err",
        ),
    }
    if args.write_launchd:
        out = Path(args.write_launchd).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        for name, xml in jobs.items():
            (out / f"{name}.plist").write_text(xml)
        print(
            f"\n# 4. launchd jobs written to {out} — copy to ~/Library/LaunchAgents, then "
            f"`launchctl bootstrap gui/$(id -u) <plist>` (after editing a plist: bootout + bootstrap)."
        )
    else:
        print(f"\n# 4. launchd jobs: rerun with --write-launchd DIR ({', '.join(jobs)})")
    print("\n# Then: memory-os-admin token create && memory-os-admin health")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="memory-os-admin", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("token")
    t.add_argument("action", choices=["create"])
    h = sub.add_parser("health")
    h.add_argument("--confined", action="store_true")
    b = sub.add_parser("backup")
    b.add_argument(
        "--no-stop", action="store_true", help="copy without stopping the service (not consistent)"
    )
    r = sub.add_parser("restore-drill")
    r.add_argument("--archive")
    r.add_argument("--no-server", action="store_true")
    k = sub.add_parser("hook")
    k.add_argument("name", choices=["session-start", "stop-nudge"])
    st = sub.add_parser("setup")
    st.add_argument("--write-launchd", metavar="DIR")
    args = p.parse_args(argv)
    s = OpsSettings.from_env()
    handler = {
        "token": cmd_token,
        "health": cmd_health,
        "backup": cmd_backup,
        "restore-drill": cmd_restore_drill,
        "hook": cmd_hook,
        "setup": cmd_setup,
    }[args.cmd]
    return handler(args, s)


if __name__ == "__main__":
    sys.exit(main())

"""One installed-artifact TCP case. Invoked outside checkout by installed_relay_e2e.py."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def main() -> int:
    if not __debug__:
        raise RuntimeError('Qualification requires assertions enabled; do not use Python -O')
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--kind', choices=['wheel', 'sdist'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = {'kind': args.kind, 'artifact_sha256': args.sha256, 'source_commit': args.source_commit, 'ok': False}
    try:
        import memory_os_v4
        from memory_os_v4.ops.auth import create_token, read_token

        prefix = Path(sys.executable).parent.parent.resolve()
        package = Path(memory_os_v4.__file__).resolve()
        assert prefix in package.parents and 'site-packages' in package.parts
        dist = importlib.metadata.distribution('memory-os-ai')
        assert dist.version == memory_os_v4.__version__ == '4.0.1'
        direct = json.loads(dist.read_text('direct_url.json'))
        assert direct.get('archive_info', {}).get('hashes', {}).get('sha256') == args.sha256
        assert not direct.get('dir_info', {}).get('editable')
        report.update(installed=True, package_path=str(package), version=dist.version)
        with tempfile.TemporaryDirectory(prefix='installed-relay-case-') as directory:
            root = Path(directory)
            token_file = root / 'token'
            create_token(token_file)
            token = read_token(token_file)
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                port = sock.getsockname()[1]
            assert port not in (8765, 8799)
            env = {**os.environ, 'MEMORY_V4_DIR': str(root / 'store'), 'MEMORY_V4_EMBED_MODEL': 'hash',
                   'MEMORY_V4_PROJECT': '*', 'MEMORY_V4_TOKEN_FILE': str(token_file),
                   'MEMORY_V4_URL': f'http://127.0.0.1:{port}/mcp/', 'MEMORY_V4_BUILD_COMMIT': args.source_commit,
                   'MEMORY_V4_OPS_DIR': str(root / 'ops'), 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                   'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1', 'PYTHONPATH': '', 'PYTHONOPTIMIZE': '0'}
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

            def get(path, authenticated=True):
                headers = {'Authorization': f'Bearer {token}'} if authenticated else {}
                request = urllib.request.Request(f'http://127.0.0.1:{port}{path}', headers=headers)
                try:
                    with opener.open(request, timeout=1) as response:
                        return response.status, json.loads(response.read())
                except urllib.error.HTTPError as exc:
                    return exc.code, {}

            server_log = (root / 'server.log').open('w+')
            relay_log = (root / 'relay.log').open('w+')

            def stop(process, abrupt=False):
                if process.poll() is None:
                    process.kill() if abrupt else process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)

            def start():
                process = subprocess.Popen([str(prefix / 'bin/memory-os-v4'), '--transport', 'http',
                                            '--host', '127.0.0.1', '--port', str(port), '--warm'],
                                           cwd=root, env=env, stdout=server_log, stderr=server_log)
                deadline = time.monotonic() + 45
                while time.monotonic() < deadline and process.poll() is None:
                    try:
                        status, body = get('/ready')
                        if status == 200:
                            assert body['version'] == '4.0.1' and body['commit'] == args.source_commit
                            assert get('/health', authenticated=False)[0] == 401
                            report['ready'] = True
                            return process
                    except (OSError, ValueError):
                        pass
                    time.sleep(0.2)
                stop(process)
                raise RuntimeError('synthetic installed server did not become ready')

            holder = {'server': start()}
            first_text = 'Synthetic packaged relay alpha fact before restart.'
            down_text = 'Synthetic rejected packaged relay fact while server down.'
            after_text = 'Synthetic packaged relay beta fact after restart.'

            async def scenario():
                params = StdioServerParameters(command=str(prefix / 'bin/memory-os-relay'), env=env, cwd=str(root))
                with anyio.fail_after(120):
                    async with stdio_client(params, errlog=relay_log) as (read, write):
                        async with ClientSession(read, write) as session:
                            await session.initialize()
                            names = {tool.name for tool in (await session.list_tools()).tools}
                            assert {'memory_add_facts', 'memory_search', 'memory_status'} <= names

                            async def add(text):
                                return await session.call_tool('memory_add_facts',
                                                               {'project': 'package-probe', 'facts': [{'text': text}]})

                            first = await add(first_text)
                            assert not first.isError
                            first_id = json.loads(first.content[0].text)['added'][0]
                            found = await session.call_tool('memory_search',
                                                            {'project': 'package-probe', 'query': first_text, 'top_k': 5})
                            assert not found.isError and first_id in {h['id'] for h in json.loads(found.content[0].text)['hits']}
                            report['roundtrip'] = True
                            await anyio.to_thread.run_sync(lambda: stop(holder['server'], abrupt=True))
                            with anyio.fail_after(30):
                                down = await add(down_text)
                            assert down.isError and any(word in down.content[0].text.lower()
                                                        for word in ('uncertain', 'unreachable', 'unavailable'))
                            report['failure_reported'] = True
                            holder['server'] = await anyio.to_thread.run_sync(start)
                            with anyio.fail_after(30):
                                after = await add(after_text)
                                status = await session.call_tool('memory_status', {})
                            assert not after.isError and not status.isError
                            after_id = json.loads(after.content[0].text)['added'][0]
                            assert json.loads(status.content[0].text)['counts']['fact'] == 2
                            report['reconnected'] = True
                            return {first_id, after_id}

            try:
                expected_ids = anyio.run(scenario)
            finally:
                stop(holder['server'])
                server_log.close()
                relay_log.close()
            rows = [json.loads(line) for line in (root / 'store/items.jsonl').read_text().splitlines()]
            latest = {row['id']: row for row in rows}
            live = {key: value for key, value in latest.items() if not value.get('deleted')}
            assert set(live) == expected_ids and {row['text'] for row in live.values()} == {first_text, after_text}
            assert down_text not in {row['text'] for row in rows}
            relay_errors = (root / 'relay.log').read_text()
            assert 'Attempted to exit cancel scope' not in relay_errors
            assert 'an error occurred during closing of asynchronous generator' not in relay_errors
            report.update(no_replay=True, facts=2, clean_shutdown=True, ok=True)
    except Exception as exc:
        report['error'] = type(exc).__name__ + ': ' + str(exc)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

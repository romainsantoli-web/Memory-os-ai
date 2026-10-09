"""Fail closed before upload; record matching installed-relay evidence without publishing."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

REQUIRED_CHECKS = {'a6_review', 'adversarial_probe', 'v3', 'v4', 'lint', 'twine_strict',
                   'wheel_install', 'sdist_install', 'wheel_smoke', 'sdist_smoke'}
EXPECTED = {'memory_os_ai-4.0.1-py3-none-any.whl', 'memory_os_ai-4.0.1.tar.gz'}
CASE_CHECKS = ('installed', 'ready', 'roundtrip', 'failure_reported', 'reconnected', 'no_replay', 'clean_shutdown')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_local(manifest, directory):
    if manifest.get('version') != '4.0.1' or not re.fullmatch(r'[0-9a-f]{40}', manifest.get('source_commit', '')):
        raise ValueError('identité version/source invalide')
    if manifest.get('open_blockers'):
        raise ValueError('blocages ouverts')
    checks = manifest.get('checks', {})
    if not all(checks.get(name) == 'passed' for name in REQUIRED_CHECKS):
        raise ValueError('preuves locales incomplètes ou en échec')
    artifacts = manifest.get('artifacts', [])
    if len(artifacts) != 2 or {a['name'] for a in artifacts} != EXPECTED:
        raise ValueError('inventaire wheel/sdist incorrect')
    for artifact in artifacts:
        path = directory / 'artifacts' / artifact['name']
        if not path.is_file() or sha256(path) != artifact['sha256']:
            raise ValueError('artefact absent/modifié : ' + artifact['name'])


def validate_relay(proof, manifest):
    if proof.get('ok') is not True or proof.get('source_commit') != manifest['source_commit']:
        raise ValueError('preuve relais absente, en échec ou provenant d’une autre source')
    expected_scripts = manifest.get('installed_relay_scripts_sha256', {})
    if set(expected_scripts) != {'runner', 'case'} or proof.get('scripts_sha256') != expected_scripts:
        raise ValueError('scripts de preuve différents')
    cases = proof.get('cases', [])
    if len(cases) != 2 or {case.get('kind') for case in cases} != {'wheel', 'sdist'}:
        raise ValueError('les deux installations doivent être testées')
    for case in cases:
        suffix = '.whl' if case['kind'] == 'wheel' else '.tar.gz'
        artifact = next(a for a in manifest['artifacts'] if a['name'].endswith(suffix))
        if (case.get('ok') is not True or case.get('exit_code') != 0 or case.get('facts') != 2
                or case.get('version') != '4.0.1' or case.get('source_commit') != manifest['source_commit']
                or case.get('artifact_sha256') != artifact['sha256']
                or not all(case.get(check) is True for check in CASE_CHECKS)):
            raise ValueError('cas installé incomplet/en échec ou autre artefact : ' + case['kind'])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--record-relay-proof', type=Path,
                        help='Enregistre les preuves concordantes et lève la garde seulement si tous les contrôles passent')
    args = parser.parse_args(argv)
    try:
        manifest = json.loads(args.manifest.read_text())
        validate_local(manifest, args.manifest.parent)
        proof_path = args.record_relay_proof or args.manifest.parent / 'installed-relay-proof.json'
        if not args.record_relay_proof and manifest.get('release_ready') is not True:
            raise ValueError('release_ready != true ; relais installé encore en attente')
        proof = json.loads(proof_path.read_text())
        validate_relay(proof, manifest)
        if args.record_relay_proof:
            destination = args.manifest.parent / 'installed-relay-proof.json'
            destination.write_bytes(proof_path.read_bytes())
            manifest.update(release_ready=True, relay_proof_sha256=sha256(destination))
            args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
        elif sha256(proof_path) != manifest.get('relay_proof_sha256'):
            raise ValueError('preuve relais modifiée depuis son acceptation')
    except (ValueError, OSError, KeyError, TypeError, StopIteration) as exc:
        print('REFUS : ' + str(exc))
        return 1
    print('Preuves et empreintes concordantes ; aucune publication effectuée. Romain conserve le GO PyPI.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Publication stays blocked for absent/stale/failed installed-artifact evidence."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('release_gate', Path(__file__).parents[1] / 'scripts/verify_release_manifest.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


@pytest.fixture
def candidate(tmp_path):
    directory = tmp_path / 'artifacts'
    directory.mkdir()
    artifacts = []
    for name in sorted(gate.EXPECTED):
        data = ('synthetic ' + name).encode()
        (directory / name).write_bytes(data)
        artifacts.append({'name': name, 'sha256': hashlib.sha256(data).hexdigest()})
    manifest = {'version': '4.0.1', 'source_commit': 'a' * 40, 'release_ready': False, 'open_blockers': [],
                'checks': {name: 'passed' for name in gate.REQUIRED_CHECKS}, 'artifacts': artifacts,
                'installed_relay_scripts_sha256': {'runner': 'b' * 64, 'case': 'c' * 64}}
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    cases = []
    for kind, suffix in [('wheel', '.whl'), ('sdist', '.tar.gz')]:
        artifact = next(a for a in artifacts if a['name'].endswith(suffix))
        cases.append({'kind': kind, 'ok': True, 'exit_code': 0, 'facts': 2, 'version': '4.0.1',
                      'source_commit': manifest['source_commit'], 'artifact_sha256': artifact['sha256'],
                      **{name: True for name in gate.CASE_CHECKS}})
    proof = {'ok': True, 'source_commit': manifest['source_commit'], 'scripts_sha256': manifest['installed_relay_scripts_sha256'],
             'cases': cases}
    return path, manifest, proof


def test_missing_proof_does_not_lift_guard(candidate):
    path, _, _ = candidate
    assert gate.main([str(path)]) == 1
    assert not json.loads(path.read_text())['release_ready']


@pytest.mark.parametrize('failure', ['source', 'artifact', 'skipped_case', 'failed_case', 'local_failure', 'tampered_artifact'])
def test_wrong_incomplete_or_failed_evidence_never_lifts_guard(candidate, failure):
    path, manifest, proof = candidate
    if failure == 'source':
        proof['source_commit'] = 'd' * 40
    elif failure == 'artifact':
        proof['cases'][0]['artifact_sha256'] = 'd' * 64
    elif failure == 'skipped_case':
        proof['cases'].pop()
    elif failure == 'failed_case':
        proof['cases'][1]['no_replay'] = False
    elif failure == 'local_failure':
        manifest['checks']['v4'] = 'failed'
        path.write_text(json.dumps(manifest))
    elif failure == 'tampered_artifact':
        (path.parent / 'artifacts' / manifest['artifacts'][0]['name']).write_bytes(b'tampered')
    evidence = path.parent / 'evidence.json'
    evidence.write_text(json.dumps(proof))
    assert gate.main([str(path), '--record-relay-proof', str(evidence)]) == 1
    assert not json.loads(path.read_text())['release_ready']


def test_complete_evidence_can_be_recorded_but_later_tampering_is_refused(candidate):
    path, _, proof = candidate
    evidence = path.parent / 'evidence.json'
    evidence.write_text(json.dumps(proof))
    assert gate.main([str(path), '--record-relay-proof', str(evidence)]) == 0
    assert json.loads(path.read_text())['release_ready']
    assert gate.main([str(path)]) == 0
    (path.parent / 'installed-relay-proof.json').write_text('{}')
    assert gate.main([str(path)]) == 1

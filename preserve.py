"""Archive this scene, its evidence, and exact local dependencies without simulation.

Uses only the Python standard library. The existing 9 GB virtual environment is
recorded and pinned, not copied. Vendor and mesh symlinks are dereferenced into
regular archived files. Native checkpoints are evidence, not certified resume.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
EXCLUDE_DIRS = {'.git', '.cache', '__pycache__', '.venv', '.pytest_cache', '.mypy_cache', '.ruff_cache'}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def command(argv, *, cwd=None, env=None, timeout=30):
    result = subprocess.run(argv, cwd=cwd, env=env, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, timeout=timeout, check=False)
    return {'argv': list(map(str, argv)), 'returncode': result.returncode,
        'stdout': result.stdout.decode('utf-8', errors='replace'),
        'stderr': result.stderr.decode('utf-8', errors='replace')}


def active_scene_processes():
    """Read /proc, excluding this preservation process and ordinary text viewers."""
    result = []
    for item in Path('/proc').iterdir():
        if not item.name.isdigit() or int(item.name) == os.getpid():
            continue
        try:
            argv = (item / 'cmdline').read_bytes().split(b'\0')
            argv = [s.decode('utf-8', errors='replace') for s in argv if s]
            cwd = (item / 'cwd').resolve()
            launchers = ('run.py', 'release_saved.py')
            runs_scene = any(str(HERE / name) == arg for name in launchers for arg in argv)
            runs_scene |= cwd == HERE and any(arg in launchers for arg in argv)
            if runs_scene:
                result.append({'pid': int(item.name), 'argv': argv, 'cwd': str(cwd)})
        except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
            continue
    return result


def collect_tree(root, archive_prefix, entries, symlinks, *, skip_preservation=False):
    def walk(path, relative, ancestors):
        resolved = path.resolve(strict=True)
        if path.is_symlink():
            symlinks[str(path)] = {'link_target': os.readlink(path), 'resolved_target': str(resolved),
                'archive_path': archive_prefix + ('/' + str(relative) if str(relative) != '.' else ''),
                'archive_behavior': 'dereferenced regular files; no external symlink dependency'}
        if resolved.is_dir():
            if resolved in ancestors:
                raise ValueError(f'Symlink directory cycle: {path}')
            for child in sorted(path.iterdir()):
                if child.is_dir() and child.name in EXCLUDE_DIRS:
                    continue
                if skip_preservation and path == root and child.name == 'preservation':
                    continue
                walk(child, relative / child.name, ancestors | {resolved})
        elif resolved.is_file():
            if path.suffix in {'.pyc', '.pyo'}:
                return
            name = archive_prefix + '/' + relative.as_posix()
            entries[name] = resolved
        else:
            raise ValueError(f'Unsupported nonregular input: {path}')
    walk(root, Path('.'), set())


def runtime_metadata(python_bin, vendor):
    script = r'''
import importlib.metadata as md, json, platform, sys
from pathlib import Path
packages = {}
for d in md.distributions():
    name = d.metadata.get('Name')
    if name:
        packages.setdefault(name.lower().replace('_','-'), {
          'name': name, 'version': d.version, 'distribution_path': str(d._path),
          'direct_url': d.read_text('direct_url.json')})
vendor_packages = {}
for d in md.distributions(path=[sys.argv[1]]):
    name = d.metadata.get('Name')
    if name: vendor_packages[name.lower().replace('_','-')] = d.version
try:
    import torch
    torch_runtime = {'version':torch.__version__, 'compiled_cuda':torch.version.cuda}
except Exception as exc: torch_runtime = {'error':repr(exc)}
print(json.dumps({'python':sys.version, 'python_executable':sys.executable,
  'platform':platform.platform(), 'libc':platform.libc_ver(),
  'packages':packages, 'vendor_packages':vendor_packages, 'torch':torch_runtime}))
'''
    env = dict(os.environ)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['PYTHONPATH'] = str(vendor)
    result = command([str(python_bin), '-c', script, str(vendor)], env=env)
    if result['returncode']:
        raise RuntimeError('Cannot record the preserved runtime: ' + result['stderr'])
    return json.loads(result['stdout'])


def genesis_snapshot(source, entries, symlinks, prefix):
    results = {name: command(['git', '-C', str(source), *args]) for name, args in {
        'head': ['rev-parse', 'HEAD'],
        'status': ['status', '--porcelain=v1', '--untracked-files=all'],
        'working_tree_diff_from_head': ['diff', '--binary', 'HEAD'],
        'staged_diff': ['diff', '--binary', '--cached'],
        'submodules': ['submodule', 'status'],
    }.items()}
    if results['head']['returncode']:
        raise RuntimeError('Genesis source must remain available as the original git checkout')
    listed = subprocess.run(['git', '-C', str(source), 'ls-files', '-z', '--cached', '--others', '--exclude-standard'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout.split(b'\0')
    for raw in sorted(set(listed)):
        if not raw:
            continue
        relative = Path(os.fsdecode(raw))
        if any(part in EXCLUDE_DIRS for part in relative.parts):
            continue
        path = source / relative
        if not path.exists():
            continue  # A deleted tracked file is preserved in the exact git diff.
        if path.is_dir():
            collect_tree(path, prefix + '/' + relative.as_posix(), entries, symlinks)
        elif path.is_file():
            entries[prefix + '/' + relative.as_posix()] = path.resolve(strict=True)
            if path.is_symlink():
                symlinks[str(path)] = {'link_target': os.readlink(path), 'resolved_target': str(path.resolve()),
                    'archive_path': prefix + '/' + relative.as_posix(), 'archive_behavior': 'dereferenced'}
    return results


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def budget_metadata(path=None):
    """Preserve the actual budget history and any user-recorded scoped override."""
    path = Path(path or HERE / 'budget.json')
    if not path.is_file():
        return {'source': str(path), 'available': False, 'effective_authorization': 'not recorded'}
    record = json.loads(path.read_text())
    override = record.get('latest_user_override')
    result = {'source': str(path), 'source_sha256': sha256(path), 'available': True,
        'original_budget_history': {key: value for key, value in record.items() if key != 'latest_user_override'},
        'latest_user_override': override}
    if override:
        result['effective_authorization'] = {
            'instruction': override.get('instruction'), 'scope': override.get('scope'),
            'authorized_run': override.get('replacement_run'),
            'original_wall_deadline_superseded_for_this_attempt_only': True,
            'stop_condition': 'Completion or definite simulation failure of the one authorized unchanged-parameter attempt; then preserve and report',
            'further_trials_authorized': False,
        }
    else:
        result['effective_authorization'] = {
            'hard_computation_deadline_utc': record.get('hard_deadline_utc'),
            'simulation_stop_target_utc': record.get('stop_simulation_by_utc'),
            'instruction': record.get('user_request'),
        }
    return result


def verify_archive(path, prefix, expected):
    """One sequential reread verifies every archived regular file and hardlink."""
    observed, unresolved = {}, []
    with tarfile.open(path, 'r|gz') as archive:
        for member in archive:
            if member.isfile():
                stream = archive.extractfile(member)
                digest = hashlib.sha256()
                size = 0
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                    digest.update(block)
                    size += len(block)
                observed[member.name] = {'size_bytes': size, 'sha256': digest.hexdigest()}
            elif member.islnk():
                if member.linkname in observed:
                    observed[member.name] = observed[member.linkname]
                else:
                    unresolved.append((member.name, member.linkname))
            elif member.issym():
                raise RuntimeError('External symlink unexpectedly present in archive: ' + member.name)
    for name, target in unresolved:
        if target not in observed:
            raise RuntimeError('Unresolved archive hardlink: ' + name)
        observed[name] = observed[target]
    problems = []
    for name, record in expected.items():
        if observed.get(name) != {'size_bytes': record['size_bytes'], 'sha256': record['sha256']}:
            problems.append(name)
    if problems:
        raise RuntimeError('Archive file verification failed: ' + ', '.join(problems[:20]))
    return {'verified': True, 'expected_files_verified': len(expected), 'archive_files_observed': len(observed),
        'unresolved_symlinks': 0, 'method': 'sequential gzip/tar reread and per-file SHA-256 comparison'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, help='New .tar.gz outside the scene; existing files are never overwritten')
    parser.add_argument('--metadata-only', action='store_true', help='Record environment and exact source metadata; do not package')
    parser.add_argument('--allow-active', action='store_true', help='Metadata only: record live diagnostic snapshot; never permits a live archive')
    args = parser.parse_args()
    active = active_scene_processes()
    if active and not (args.metadata_only and args.allow_active):
        raise SystemExit('Scene simulation is still running. Stop it gracefully before preservation: ' + json.dumps(active))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    record_dir = HERE / 'preservation' / stamp
    record_dir.mkdir(parents=True, exist_ok=False)
    prefix = HERE.name
    runtime = HERE.parent / '.venv'
    source = HERE.parent / 'genesis-world'
    entries, symlinks = {}, {}
    env_record = runtime_metadata(runtime / 'bin/python', (HERE / 'vendor').resolve())
    git_results = genesis_snapshot(source, entries, symlinks, prefix + '/preserved_runtime/genesis-world')
    for name in ['working_tree_diff_from_head', 'staged_diff']:
        (record_dir / ('genesis_' + name + '.patch')).write_text(git_results[name]['stdout'])
    write_json(record_dir / 'genesis_git.json', git_results)
    write_json(record_dir / 'runtime_environment.json', env_record)
    packages = env_record['packages']
    bundled = set(env_record['vendor_packages']) | {'genesis-world'}
    (record_dir / 'requirements.lock.txt').write_text(
        '# Exact observed distribution versions; not a wheel-hash lock.\n'
        '# Genesis source and scene vendor are included separately.\n'
        '# Torch runtime CUDA flavor is recorded in runtime_environment.json.\n'
        + ''.join(f"{value['name']}=={value['version']}\n" for key, value in sorted(packages.items()) if key not in bundled))
    pyvenv = runtime / 'pyvenv.cfg'
    if pyvenv.is_file():
        (record_dir / 'pyvenv.cfg.reference.txt').write_bytes(pyvenv.read_bytes())
    gpu = command(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'])
    write_json(record_dir / 'system.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
        'platform': platform.platform(), 'gpu': gpu, 'active_scene_processes': active,
        'runtime_directory_not_copied': str(runtime), 'runtime_python': str(runtime / 'bin/python'),
        'scene_directory': str(HERE), 'genesis_source': str(source),
        'budget_and_user_authorization': budget_metadata(),
        'native_checkpoint_restore_implemented': (HERE / 'release_saved.py').is_file(),
        'checkpoint_restore_scope': 'Scene-specific saved-state release helper; state readback checked, uninterrupted-step equivalence not certified',
        'exact_midpoint_resume_certified': False,
        'supported_continuation': 'Initial-state replay or scene-specific restored-state release with a new output name; exact uninterrupted continuation not certified',
        'limitations': ['No virtual environment or offline wheelhouse is included',
            'Package version list is not a hash-locked reinstall guarantee',
            'No clean-machine installation or exact checkpoint restart is certified']})
    write_json(record_dir / 'source_symlinks.json', symlinks)
    if args.metadata_only:
        print(json.dumps({'metadata_directory': str(record_dir), 'archive_created': False, 'active_processes': active}, ensure_ascii=False))
        return
    archive_path = (args.archive or HERE.parent / (HERE.name + '_saved_' + stamp + '.tar.gz')).absolute()
    if archive_path == HERE or HERE in archive_path.parents:
        raise ValueError('The archive must be outside the scene to avoid recursively archiving itself')
    if archive_path.exists() or Path(str(archive_path) + '.partial').exists():
        raise FileExistsError('Archive or partial file already exists; choose another output name')
    collect_tree(HERE, prefix, entries, symlinks, skip_preservation=True)
    write_json(record_dir / 'source_symlinks.json', symlinks)
    collect_tree(record_dir, prefix + '/preservation/' + stamp, entries, symlinks)
    manifest = {}
    for name, path in sorted(entries.items()):
        info = path.stat()
        digest = sha256(path)
        after = path.stat()
        if (info.st_size, info.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError('Input changed while hashing: ' + str(path))
        manifest[name] = {'size_bytes': info.st_size, 'sha256': digest, 'source_resolved': str(path),
            'source_mtime_ns': info.st_mtime_ns, 'mode': info.st_mode & 0o777}
    manifest_path = record_dir / 'file_manifest.json'
    write_json(manifest_path, {'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'files': manifest, 'excluded': ['.venv', '.git metadata (HEAD/status/diffs and actual source files preserved)',
            'regenerable caches and Python bytecode', 'older preservation archives'],
        'symlinks_dereferenced': True, 'all_scene_inputs_outputs_logs_included': True})
    manifest_name = prefix + '/preservation/' + stamp + '/file_manifest.json'
    entries[manifest_name] = manifest_path
    expected = {**manifest, manifest_name: {'size_bytes': manifest_path.stat().st_size, 'sha256': sha256(manifest_path)}}
    temporary = Path(str(archive_path) + '.partial')
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    deduplicated = {}
    with temporary.open('xb') as raw:
        with gzip.GzipFile(filename='', mode='wb', compresslevel=1, fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w|', format=tarfile.PAX_FORMAT) as archive:
                for name, path in sorted(entries.items()):
                    info = path.stat()
                    record = expected[name]
                    if name in manifest and (info.st_size != record['size_bytes'] or info.st_mtime_ns != record['source_mtime_ns']):
                        raise RuntimeError('Input changed before archiving: ' + str(path))
                    tarinfo = tarfile.TarInfo(name)
                    tarinfo.mode = info.st_mode & 0o777
                    tarinfo.mtime = info.st_mtime
                    key = (record['size_bytes'], record['sha256'], tarinfo.mode)
                    if key in deduplicated and record['size_bytes'] > 0:
                        tarinfo.type = tarfile.LNKTYPE
                        tarinfo.linkname = deduplicated[key]
                        archive.addfile(tarinfo)
                    else:
                        tarinfo.size = info.st_size
                        with path.open('rb') as stream:
                            archive.addfile(tarinfo, stream)
                        deduplicated[key] = name
    verification = verify_archive(temporary, prefix, expected)
    temporary.rename(archive_path)
    archive_sha = sha256(archive_path)
    Path(str(archive_path) + '.sha256').write_text(archive_sha + '  ' + archive_path.name + '\n')
    verification.update(archive=str(archive_path), archive_sha256=archive_sha,
        archive_size_bytes=archive_path.stat().st_size, metadata_directory=str(record_dir),
        exact_midpoint_resume_certified=False,
        native_checkpoint_restore_implemented=(HERE / 'release_saved.py').is_file(),
        checkpoint_restore_scope='Scene-specific saved-state release; uninterrupted-step equivalence not certified')
    write_json(Path(str(archive_path) + '.verification.json'), verification)
    write_json(record_dir / 'archive_verification.json', verification)
    print(json.dumps(verification, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

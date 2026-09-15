"""Download immutable official assets, validate LFS SHA256, and retain provenance."""
from pathlib import Path
import concurrent.futures
import hashlib
import json
import time
import requests

ROOT = Path(__file__).resolve().parent
SOURCES = ROOT / 'sources'
MODELS = ROOT / 'models'


def get_json(url):
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.json()


def one(repo, revision, initial):
    name = repo.split('/')[-1]
    folder = MODELS / name
    folder.mkdir(parents=True, exist_ok=True)
    tree = get_json(f'https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=true')
    original = get_json(f'https://huggingface.co/api/models/{repo}/tree/{initial}?recursive=true')
    (SOURCES / f'{name}_tree.json').write_text(json.dumps(tree, indent=2), encoding='utf-8')
    (SOURCES / f'{name}_initial_tree.json').write_text(json.dumps(original, indent=2), encoding='utf-8')
    initial_weight = next(x for x in original if x['path'] == 'model.safetensors')['lfs']['oid']
    files = []
    for filename in ['README.md', 'config.json', 'model.safetensors']:
        info = next(x for x in tree if x['path'] == filename)
        url = f'https://huggingface.co/{repo}/resolve/{revision}/{filename}'
        path = folder / filename
        expected = info.get('lfs', {}).get('oid')
        if not path.exists() or (expected and hashlib.sha256(path.read_bytes()).hexdigest() != expected):
            for attempt in range(3):
                try:
                    with requests.get(url, stream=True, timeout=(30, 120)) as response:
                        response.raise_for_status()
                        with path.open('wb') as output:
                            for chunk in response.iter_content(1024 * 1024):
                                output.write(chunk)
                    break
                except requests.RequestException:
                    if attempt == 2:
                        raise
                    time.sleep(1)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected:
            assert sha == expected, f'Weight hash mismatch {name}'
            assert sha == initial_weight, f'Weight changed since initial upload {name}'
        files.append(dict(file=str(path.relative_to(ROOT)), bytes=path.stat().st_size,
                          sha256=sha, url=url, lfs_sha256=expected))
        print(f'{name}/{filename}: {path.stat().st_size} bytes verified', flush=True)
    return dict(repo=repo, revision=revision, initial_model_revision=initial,
                weights_identical_to_initial_upload=True, files=files)


if __name__ == '__main__':
    cfg = json.loads((ROOT / 'protocol.json').read_text(encoding='utf-8'))['kronos']
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(one, cfg['model'], cfg['model_revision'], 'ac5c409a313c4eadd1ca78201322f5cadb9c34ab'),
                pool.submit(one, cfg['tokenizer'], cfg['tokenizer_revision'], '9ef143b98ee3c2488eebd85404e0c215c112b46a')]
        result = [job.result() for job in jobs]
    (SOURCES / 'weights_manifest.json').write_text(json.dumps(result, indent=2), encoding='utf-8')

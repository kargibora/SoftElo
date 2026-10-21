"""Build annotation-only Parquet exports from the paper's original JSON files."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .data import SOURCES, SCORE_COLUMNS, PREF_COLUMNS, battle_id


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def number(value):
    if value is None:
        return None
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def score_json(value):
    if not isinstance(value, dict):
        return None
    # Retain the original keys, including missing scores as JSON null.
    return json.dumps({str(k): number(v) for k, v in value.items()}, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)


def paper_valid(row):
    # Exact experiments.bt.clean_scores rule for the numeric dictionaries here.
    return all(isinstance(row.get(k), dict) and all(v is not None and number(v) is not None
               for v in row[k].values()) for k in ('scores_a', 'scores_b'))


def shared_eligible(row):
    if row.get('parse_error'):
        return False
    a, b = row.get('scores_a'), row.get('scores_b')
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    dims = set(a) & set(b)
    return bool(dims) and all(number(a[d]) is not None and number(b[d]) is not None
                             and 1 <= float(a[d]) <= 10 and 1 <= float(b[d]) <= 10
                             for d in dims) and all(number(row.get(k)) in {0., .25, .5, .75, 1.}
                                                   for k in ('human_pref', 'judge_pref'))


def upstream_index(corpus, cache, revision=None):
    repo, id_col = SOURCES[corpus]
    snapshots = cache / ('datasets--' + repo.replace('/', '--')) / 'snapshots'
    candidates = [p for p in snapshots.iterdir() if list(p.rglob('*.parquet')) and (revision is None or p.name == revision)] if snapshots.exists() else []
    if not candidates and revision:
        return remote_upstream_index(corpus, revision)
    if len(candidates) != 1:
        raise ValueError(f'Expected one cached snapshot for {corpus}; found {len(candidates)}')
    snapshot = candidates[0]
    index = {}
    a, b = ('model_a_name', 'model_b_name') if corpus == 'comparia' else ('model_a', 'model_b')
    cols = [id_col, a, b] + (['chosen_model_name', 'both_equal'] if corpus == 'comparia' else ['winner'])
    files = []
    for path in sorted(snapshot.rglob('*.parquet')):
        files.append({'path': str(path.relative_to(snapshot)), 'sha256': sha256(path)})
        for batch in pq.ParquetFile(path).iter_batches(columns=cols, batch_size=10000):
            for row in batch.to_pylist():
                sid = str(row[id_col])
                if corpus == 'comparia':
                    chosen = row['chosen_model_name']
                    hp = .5 if row['both_equal'] or chosen is None else (0. if chosen == row[a] else 1. if chosen == row[b] else .5)
                else:
                    hp = {'model_a': 0., 'model_b': 1.}.get(str(row['winner']).lower(), .5)
                val = (str(row[a]), str(row[b]), hp)
                if sid in index and index[sid] != val:
                    raise ValueError(f'Conflicting source ID {sid}')
                index[sid] = val
    return index, {'repo_id': repo, 'id_column': id_col, 'revision': snapshot.name,
                   'revision_status': 'verified_join_snapshot; original annotation-run revision was not recorded',
                   'files': files}


def remote_upstream_index(corpus, revision):
    from huggingface_hub import HfApi, HfFileSystem
    import tempfile
    repo, id_col = SOURCES[corpus]
    cached = Path(tempfile.gettempdir()) / 'softelo-source-index' / f'{corpus}-{revision}.json'
    if cached.exists():
        data = json.loads(cached.read_text())
        return {k: tuple(v) for k, v in data['index'].items()}, data['source']
    info = HfApi().dataset_info(repo, revision=revision, files_metadata=True)
    fs = HfFileSystem()
    a, b = ('model_a_name', 'model_b_name') if corpus == 'comparia' else ('model_a', 'model_b')
    cols = [id_col, a, b] + (['chosen_model_name', 'both_equal'] if corpus == 'comparia' else ['winner'])
    index, files = {}, []
    for file in info.siblings:
        if not file.rfilename.endswith('.parquet'):
            continue
        files.append({'path': file.rfilename, 'sha256': file.lfs.sha256 if file.lfs else None})
        with fs.open(f'datasets/{repo}@{revision}/{file.rfilename}', 'rb', block_size=1 << 20) as handle:
            table = pq.ParquetFile(handle).read(columns=cols)
        for row in table.to_pylist():
            if corpus == 'comparia':
                chosen = row['chosen_model_name']
                hp = .5 if row['both_equal'] or chosen is None else (0. if chosen == row[a] else 1. if chosen == row[b] else .5)
            else:
                hp = {'model_a': 0., 'model_b': 1.}.get(str(row['winner']).lower(), .5)
            sid = str(row[id_col])
            val = (str(row[a]), str(row[b]), hp)
            if sid in index and index[sid] != val:
                raise ValueError(f'Conflicting source ID {sid}')
            index[sid] = val
    source = {'repo_id':repo, 'id_column':id_col, 'revision':revision,
              'revision_status':'verified_join_snapshot; original annotation-run revision was not recorded', 'files':files}
    write_json(cached, {'index':index, 'source':source})
    return index, source


def sanitize(row, *, corpus, judge, row_index, source_index, revision):
    meta = row.get('metadata') or row.get('instruction_metadata') or {}
    sid = str(meta.get('original_question_id', ''))
    a, b = str(row['model_a']), str(row['model_b'])
    hp = number(row.get('human_pref', row.get('human_preference')))
    jp = number(row.get('judge_pref', row.get('preference')))
    source = source_index.get(sid)
    if source is None or (a, b) not in [source[:2], source[:2][::-1]]:
        raise ValueError(f'{corpus}/{judge}/{row_index}: cannot join source ID and models')
    expected_hp = source[2] if (a, b) == source[:2] else 1 - source[2]
    if hp != expected_hp:
        raise ValueError(f'{corpus}/{judge}/{row_index}: human preference differs from upstream')
    normalized = dict(row, human_pref=hp, judge_pref=jp)
    out = {'battle_id': battle_id(corpus, sid, a, b), 'source_dataset': corpus,
           'source_id': sid, 'source_revision': revision, 'judge': judge, 'row_index': row_index,
           'instruction_id': str(row.get('instruction_id', '')), 'model_a': a, 'model_b': b,
           'source_order_reversed': (a, b) != source[:2], 'human_pref': hp, 'judge_pref': jp,
           'language': str(meta.get('lang') or meta.get('language') or ''),
           'parse_error': bool(row.get('parse_error', False)),
           'paper_valid': paper_valid(normalized), 'shared_eligible': shared_eligible(normalized),
           'has_order_scores': all(isinstance(row.get(k), dict) for k in SCORE_COLUMNS[2:]),
           'has_order_preferences': all(number(row.get(k)) is not None for k in PREF_COLUMNS[2:])}
    for key in SCORE_COLUMNS:
        out[key] = score_json(row.get(key))
    for key in PREF_COLUMNS[2:]:
        out[key] = number(row.get(key))
    for key in ('len_a', 'len_b'):
        out[key] = number(row.get(key))
    return out


def arrow_schema():
    strings = ['battle_id','source_dataset','source_id','source_revision','judge','instruction_id',
               'model_a','model_b','language', *SCORE_COLUMNS]
    flags = ['source_order_reversed','parse_error','paper_valid','shared_eligible',
             'has_order_scores','has_order_preferences']
    return pa.schema([(k, pa.string()) for k in strings] + [('row_index', pa.int64())]
                     + [(k, pa.bool_()) for k in flags]
                     + [(k, pa.float64()) for k in [*PREF_COLUMNS, 'len_a','len_b']])


def safe_run_metadata(data):
    meta = data.get('metadata') or {}
    cfg = data.get('config_snapshot') or {}
    # Explicit allowlist: do not copy provider configs, paths, keys or responses.
    out = {k: meta.get(k) for k in ['judge_model','judge_mode','pairwise_prompt_style',
           'swap_debiasing','judge_temperature','n_trials','criteria_k','balance_by']}
    out['sampling'] = {k: cfg.get(k) for k in ['dataset','n_instructions','seed','language','balance_by']}
    out['criterion_names'] = data.get('criterion_names')
    out['criteria_definition'] = data.get('criteria_definition')
    out['system_prompt'] = data.get('system_prompt')
    return out


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def export(repo, cache, out, source_revisions):
    manifest_path = repo / 'release_softelo_minimal/data/manifest.csv'
    runs = list(csv.DictReader(manifest_path.open()))
    if len(runs) != 22:
        raise ValueError('Expected the 22 paper annotation sources')
    if (out / 'manifest.json').exists():
        raise FileExistsError('Choose a new output directory; do not overwrite a frozen release')
    manifest = {'schema_version': 1, 'sources': {}, 'annotations': [], 'subsets': {}}
    for corpus in source_revisions:
        manifest['sources'][corpus] = {'repo_id': SOURCES[corpus][0], 'snapshots': {}}
        indices = {}
        supports = []
        for run in [r for r in runs if r['source_dataset'] == corpus]:
            path = repo / run['source_json']
            data = json.loads(path.read_text())
            judge = run['judge']
            revision = source_revisions[corpus].get(judge, source_revisions[corpus]['default'])
            if revision not in indices:
                indices[revision] = upstream_index(corpus, cache, revision)
            source_index, source_info = indices[revision]
            manifest['sources'][corpus]['snapshots'][revision] = source_info
            rows = [sanitize(row, corpus=corpus, judge=judge, row_index=i, source_index=source_index,
                             revision=source_info['revision']) for i, row in enumerate(data['per_sample'])]
            filename = Path(run['csv_path']).stem + '.parquet'
            dest = out / 'data' / corpus / filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist(rows, schema=arrow_schema()), dest, compression='zstd')
            support = {}
            for row in rows:
                if row['shared_eligible']:
                    human_canonical = row['human_pref'] if row['model_a'] <= row['model_b'] else 1-row['human_pref']
                    if row['battle_id'] in support:
                        raise ValueError(f'Duplicate eligible battle within {judge}')
                    support[row['battle_id']] = human_canonical
            supports.append(support)
            metadata_file = Path('metadata') / corpus / (filename.replace('.parquet', '.json'))
            write_json(out / metadata_file, safe_run_metadata(data))
            record = {'corpus': corpus, 'judge': judge, 'file': str(dest.relative_to(out)),
                      'rows': len(rows), 'paper_valid_rows': sum(r['paper_valid'] for r in rows),
                      'shared_eligible_rows':len(support), 'order_score_rows':sum(r['has_order_scores'] for r in rows),
                      'order_preference_rows':sum(r['has_order_preferences'] for r in rows),
                      'source_json':run['source_json'], 'source_sha256':sha256(path),
                      'sha256':sha256(dest), 'metadata':str(metadata_file)}
            manifest['annotations'].append(record)
            print(f"{corpus} {judge}: {len(rows)} rows, {record['paper_valid_rows']} paper-valid, "
                  f"{record['order_score_rows']} with separate presentation scores", flush=True)
            del data, rows
        common = set.intersection(*(set(s) for s in supports))
        common = {bid for bid in common if len({s[bid] for s in supports}) == 1}
        dest = out / 'subsets' / f'{corpus}_shared_valid.parquet'
        dest.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({'battle_id': sorted(common)}), dest, compression='zstd')
        manifest['subsets'][corpus] = {'file': str(dest.relative_to(out)), 'battles':len(common),
                                     'judges':len(supports), 'sha256':sha256(dest)}
        if corpus == 'lmarena100k' and len(common) != 18158:
            raise ValueError(f'Paper shared-battle count differs: {len(common)} != 18158')
        print(f'{corpus}: {len(common)} shared valid battles', flush=True)
    write_json(out / 'manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--hf-cache', type=Path, default=Path.home()/'.cache/huggingface/hub')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source-revisions', type=Path, required=True)
    args = parser.parse_args()
    export(args.repo_root, args.hf_cache, args.out, json.loads(args.source_revisions.read_text()))


if __name__ == '__main__':
    main()

"""Validate released schemas, identities, counts, subsets and file checksums."""
import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from .data import battle_id
from .export import arrow_schema, sha256, paper_valid, shared_eligible


def validate(root):
    root=Path(root)
    manifest=json.loads((root/'manifest.json').read_text())
    supports={}
    counts=[]
    total=0
    for spec in manifest['annotations']:
        path=root/spec['file']
        if sha256(path)!=spec['sha256']:
            raise ValueError(f'Checksum mismatch: {path}')
        table=pq.read_table(path)
        if table.schema != arrow_schema():
            raise ValueError(f'Schema differs: {path}')
        rows=table.to_pylist()
        if len(rows)!=spec['rows'] or [r['row_index'] for r in rows]!=list(range(len(rows))):
            raise ValueError(f'Row count/order differs: {path}')
        support={}
        for r in rows:
            if r['battle_id']!=battle_id(r['source_dataset'],r['source_id'],r['model_a'],r['model_b']):
                raise ValueError(f'Invalid battle ID: {path}')
            if r['source_dataset']!=spec['corpus'] or r['judge']!=spec['judge']:
                raise ValueError(f'Misassigned annotation: {path}')
            for k in ('scores_a','scores_b'):
                r[k]=json.loads(r[k]) if r[k] is not None else None
            if r['paper_valid']!=paper_valid(r) or r['shared_eligible']!=shared_eligible(r):
                raise ValueError(f'Incorrect subset flags: {path}')
            if r['shared_eligible']:
                if r['battle_id'] in support:
                    raise ValueError(f'Duplicate eligible battle: {path}')
                support[r['battle_id']]=r['human_pref'] if r['model_a']<=r['model_b'] else 1-r['human_pref']
        if sum(r['paper_valid'] for r in rows)!=spec['paper_valid_rows'] or len(support)!=spec['shared_eligible_rows']:
            raise ValueError(f'Invalid filtered counts: {path}')
        supports.setdefault(spec['corpus'],[]).append(support)
        total+=len(rows)
        counts.append({'corpus':spec['corpus'],'judge':spec['judge'],'rows':len(rows)})
    for corpus,sets in supports.items():
        expected=set.intersection(*(set(s) for s in sets))
        expected={bid for bid in expected if len({s[bid] for s in sets})==1}
        spec=manifest['subsets'][corpus]
        path=root/spec['file']
        actual=pq.read_table(path)['battle_id'].to_pylist()
        if len(actual)!=len(set(actual)) or set(actual)!=expected or len(actual)!=spec['battles'] or sha256(path)!=spec['sha256']:
            raise ValueError(f'Shared subset differs: {corpus}')
    if manifest['subsets']['lmarena100k']['battles']!=18158:
        raise ValueError('Paper committee subset must contain 18,158 battles')
    checksums=root/'checksums.json'
    if checksums.exists():
        for file, expected in json.loads(checksums.read_text()).items():
            if sha256(root/file)!=expected:
                raise ValueError(f'Release checksum mismatch: {file}')
    return {'passed':True,'annotation_rows':total,'annotation_files':len(counts),'counts':counts,
            'shared_battles':{k:v['battles'] for k,v in manifest['subsets'].items()},
            'scope':'schemas, hashes, row identities, subset flags and membership; no Elo fits rerun'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset',type=Path)
    p.add_argument('--report',type=Path)
    args=p.parse_args()
    report=validate(args.dataset)
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='counts'},indent=2))


if __name__=='__main__':
    main()

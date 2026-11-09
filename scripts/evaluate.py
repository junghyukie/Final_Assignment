from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, fields
import csv
import json
from pathlib import Path
from coverevo.contracts import Instance, Program
from coverevo.dsl import parse_program
from coverevo.external_mwds import parse_original
from coverevo.solver import check_solution, greedy_reference, rarity_reference, reverse_delete, solve

ROOT=Path(__file__).resolve().parents[1]
BASELINES={'GR_P':greedy_reference,'RGR_P':rarity_reference,'RD':reverse_delete}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_winners(root=ROOT):
    winners=read(root/'results/winners.json')
    memberships={(w['method'],w['run_id']) for w in winners}
    expected={(m,r) for m in ('IND','EVO_P','EVO_F','RS_PARAM') for r in range(5)}
    if len(winners)!=20 or memberships!=expected:
        raise ValueError('expected all twenty original frozen run memberships')
    for winner in winners:
        parsed=parse_program(winner['expression'],winner['provenance'])
        if parsed.program_hash!=winner['program_hash']:
            raise ValueError('frozen program expression and functional ID disagree')
    return winners


def check_record(instance,row):
    if row['success']:
        selected=tuple(row['selected'])
        if not check_solution(instance,selected):
            raise ValueError('saved solution is not a dominating set')
        cost=sum(instance.costs[v] for v in selected)
        if cost!=row['raw_cost'] or cost!=row['operational_cost']:
            raise ValueError('saved solution cost disagrees with original weights')
        return True
    if row['operational_cost']!=sum(instance.costs):
        raise ValueError('failed call has the wrong operational penalty')
    return False


def check_oracle(instance,row):
    certified=row['status']=='OPTIMAL'
    if row['certified_optimal']!=certified:
        raise ValueError('oracle certification requires OPTIMAL status')
    if row['selected'] is not None:
        if not check_solution(instance,tuple(row['selected'])):
            raise ValueError('oracle returned an infeasible set')
        if sum(instance.costs[v] for v in row['selected'])!=row['incumbent_cost']:
            raise ValueError('oracle cost disagrees with original weights')
    elif certified:
        raise ValueError('certified oracle has no incumbent')
    return certified


def evaluate_frozen(instances,winners):
    rows=[];unique_calls=0
    names={f.name for f in fields(Program)}
    for instance in instances:
        measured={}
        for method,fn in BASELINES.items():
            measured[method]=asdict(fn(instance));unique_calls+=1
        reference=measured['GR_P']['raw_cost']
        if not measured['GR_P']['success']:
            raise ValueError('greedy reference failed')
        for method in BASELINES:
            rows.append(dict(method=method,run_id=None,instance_id=instance.instance_id,
                             program_hash=None,reference_cost=reference,**measured[method]))
        for winner in winners:
            identity=winner['program_hash']
            if identity not in measured:
                program=Program(**{k:winner[k] for k in names})
                measured[identity]=asdict(solve(instance,program));unique_calls+=1
            rows.append(dict(method=winner['method'],run_id=winner['run_id'],
                instance_id=instance.instance_id,program_hash=identity,
                reference_cost=reference,**measured[identity]))
        for row in rows[-(len(winners)+3):]:
            check_record(instance,row)
    return rows,unique_calls


def verify_saved(root=ROOT):
    synthetic={i['instance_id']:Instance.from_mapping(i) for split in ('train','validation','test')
               for i in read(root/f'data/synthetic/{split}.json')}
    external={i['instance_id']:Instance.from_mapping(i) for i in read(root/'data/external_mwds/instances.json')}
    winners={(w['method'],w['run_id']):w for w in load_winners(root)}
    quality=read(root/'results/evaluation.json')
    timings=read(root/'results/timing.json')
    if len(quality)!=10500 or len(timings)!=31500:
        raise ValueError('original evaluation grid is incomplete')
    seen=set();failed=0
    for row in quality:
        key=(row['method'],row['run_id'],row['instance_id'])
        if key in seen:raise ValueError('duplicate quality observation')
        seen.add(key)
        if key[:2] in winners and row['program_hash']!=winners[key[:2]]['program_hash']:
            raise ValueError('saved observation does not use its frozen winner')
        failed+=not check_record(synthetic[row['instance_id']],row)
    for name,instances in (('results/oracles.json',synthetic),('results/external/oracles.json',external)):
        for identity,row in read(root/name).items():check_oracle(instances[identity],row)
    members=read(root/'results/external/quality.json')
    calls=read(root/'results/external/calls.json')
    if len(members)!=1104 or len(calls)!=1413:
        raise ValueError('external evaluation records are incomplete')
    for row in members+calls:failed+=not check_record(external[row['instance_id']],row)
    # Native published matrices provide an independent check of the adapter.
    with (root/'data/external_mwds/selection.csv').open(newline='',encoding='utf-8') as f:
        selected=list(csv.DictReader(f))
    if len(selected)!=48 or len({r['instance_id'] for r in selected})!=48:
        raise ValueError('external instance selection is incomplete or duplicated')
    for source in selected:
        parsed=parse_original((root/'data/external_mwds'/source['file']).read_bytes(),source['source_member'])
        saved=external[source['instance_id']]
        if (parsed.n,parsed.costs,parsed.edges)!=(saved.n,saved.costs,saved.edges):
            raise ValueError('native benchmark and parsed graph disagree')
    return dict(original_quality=len(quality),original_timings=len(timings),
                external_memberships=len(members),external_calls=len(calls),failures=failed,
                frozen_winners=len(winners),native_external_instances=len(selected))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-saved',action='store_true',help='check saved solutions without rerunning solvers')
    parser.add_argument('--dataset',choices=('synthetic','external'),default='synthetic')
    parser.add_argument('--limit',type=int,default=1,help='number of instances for fresh local evaluation; 0 means all')
    parser.add_argument('--output',type=Path,help='optional new local evaluation JSON')
    args=parser.parse_args()
    if args.verify_saved:
        print(json.dumps(verify_saved(),indent=2));return
    if args.limit<0:parser.error('--limit must be nonnegative')
    path=('data/synthetic/test.json' if args.dataset=='synthetic' else 'data/external_mwds/instances.json')
    instances=[Instance.from_mapping(i) for i in read(ROOT/path)]
    if args.limit:instances=instances[:args.limit]
    rows,calls=evaluate_frozen(instances,load_winners())
    saved=read(ROOT/('results/evaluation.json' if args.dataset=='synthetic' else 'results/external/quality.json'))
    aliases={'GR+P':'GR_P','RGR+P':'RGR_P','Reverse-delete':'RD'}
    expected={(aliases.get(r['method'],r['method']),r.get('run_id',r.get('run')),r['instance_id']):r['operational_cost'] for r in saved}
    for row in rows:
        run=0 if row['method'] in BASELINES and args.dataset=='synthetic' else row['run_id']
        if row['operational_cost']!=expected[row['method'],run,row['instance_id']]:
            raise ValueError('fresh frozen evaluation differs from saved quality')
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(instances=len(instances),memberships=len(rows),unique_calls=calls,
                         failures=sum(not r['success'] for r in rows),quality_matches_saved=True),indent=2))


if __name__=='__main__':main()

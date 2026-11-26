"""Only metadata inventory and guarded CPU synthetic checks are executable here."""
from __future__ import annotations
import argparse
import io
import os
from pathlib import Path
import sys
from datetime import datetime,timezone,timedelta
from .artifacts import ArtifactStore,ContractError,PinnedMetadataReader,strict_json,bytes_hash

ROOT=Path(__file__).resolve().parents[2]


def new_store(path):
    target=Path(path).resolve()
    allowed=(ROOT/'runs/research_v4').resolve()
    if not target.is_relative_to(allowed) or target==allowed or target.exists():
        raise ContractError('Output must be a NEW child of runs/research_v4')
    target.mkdir(parents=True,exist_ok=False)
    return ArtifactStore(target)


def metadata_command(args):
    from .metadata import inventory
    config=ROOT/'configs/research_v4/safe_metadata_sources_v1.json'
    reader=PinnedMetadataReader(ROOT,strict_json(config.read_bytes())['assets'])
    report,ledger=inventory(reader)
    store=new_store(args.output)
    store.write_json('capacity_and_exposure_summary.json',report)
    store.write_json('opaque_exposure_ledger.json',ledger)
    store.write_json('metadata_read_receipt.json',{'files':reader.access_log,
                           'feature_values_read':False,'retired_details_read':False,
                           'row_scope_attestation_created':False})
    print(report['status'],report['metadata_players'],report['eligible_not_selected'])


def synthetic_command(args):
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    sys.dont_write_bytecode=True
    store=new_store(args.output)
    denied=[];opens=0
    protected={'data','artifacts','checkpoints','references'}
    def audit(event,values):
        nonlocal opens
        if event=='open' and isinstance(values[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(values[0])).resolve()
            if p.is_relative_to(ROOT):
                rel=p.relative_to(ROOT)
                if rel.parts and (rel.parts[0] in protected or (rel.parts[0]=='runs' and (len(rel.parts)<2 or rel.parts[1]!='research_v4'))):
                    denied.append(rel.as_posix())
                    raise PermissionError('Synthetic process cannot access real/legacy data')
                opens+=1
        if event in ('socket.connect','subprocess.Popen','os.system'):
            raise PermissionError('Synthetic process cannot use network or launch processes')
    sys.addaudithook(audit)
    source_hashes=[{'path':p.relative_to(ROOT).as_posix(),'sha256':bytes_hash(p.read_bytes())}
                   for base,pattern in [('src/goid_v4','*.py'),('tests/research_v4','*.py'),('configs/research_v4','*.json')]
                   for p in sorted((ROOT/base).glob(pattern))]
    import torch
    import unittest
    torch.set_num_threads(2)
    torch.random.default_generator.manual_seed(20261025)
    def no_weights(*args,**kwargs):raise PermissionError('Synthetic checks cannot load or save checkpoints')
    torch.load=torch.save=no_weights
    stream=io.StringIO()
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests/research_v4'),pattern='test_v4_*.py')
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    initialized=torch.cuda.is_initialized()
    report={'status':'passed_cpu_synthetic' if result.wasSuccessful() and not initialized else 'failed',
            'completed_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),
            'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
            'skipped':len(result.skipped),'cuda_initialized':initialized,'gpu_kernels_run':False,
            'actual_data_or_weights_used':False,'quality_evaluated':False,'training_executed':False,
            'synthetic_gradient_backpropagation_tested':True,'torch':torch.__version__,
            'expected_denied_synthetic_probes':denied,'workspace_open_events':opens,
            'g0_real_scope_passed':False,'real_pilot_config_frozen':False}
    report['engineering_source_hashes']=source_hashes
    store.write('unittest.log',stream.getvalue().encode('utf-8'))
    store.write_json('result.json',report)
    print(stream.getvalue() if not result.wasSuccessful() else f'CPU synthetic: {result.testsRun} passed; CUDA initialized={initialized}')
    return 0 if report['status']=='passed_cpu_synthetic' else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('inventory-metadata','check-synthetic'):
        p=sub.add_parser(name);p.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    try:
        if args.command=='inventory-metadata':metadata_command(args);return 0
        return synthetic_command(args)
    except (ContractError,FileExistsError,PermissionError) as exc:
        parser.error(str(exc))


if __name__=='__main__':raise SystemExit(main())

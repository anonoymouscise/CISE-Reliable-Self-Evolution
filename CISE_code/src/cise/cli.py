import argparse
import json
import os
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(prog='cise',description='CISE Gibbs conformal interval and LLEMA materials search')
    sub=ap.add_subparsers(dest='command',required=True)
    for cmd in ['init','run','status','report','validate','doctor']:
        p=sub.add_parser(cmd);p.add_argument('--config',required=True,type=Path)
    p=sub.add_parser('import-calibration');p.add_argument('--source',type=Path,required=True);p.add_argument('--destination',type=Path,required=True)
    p=sub.add_parser('calibrate');p.add_argument('--manifest',type=Path,required=True);p.add_argument('--destination',type=Path,required=True)
    p=sub.add_parser('configure');p.add_argument('--method',choices=['llema','cci'],required=True);p.add_argument('--demo',action='store_true');p.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if hasattr(args,'config'):
        os.environ['CISE_CONFIG']=str(args.config.resolve())
        from .runtime import CONFIG,EXPERIMENT as root
    try:
        if args.command=='init':
            from .assets import initialize
            initialize();print('Initialized '+str(root))
        elif args.command=='run':
            from .controller import run
            run();print(str(root/'REPORT.md'))
        elif args.command=='status':
            paths=[root/'controller_status.json']+list((root/'runs').glob('*/*/status.json'))
            print(json.dumps({str(p.relative_to(root)):json.loads(p.read_text()) for p in paths if p.exists()},indent=2))
        elif args.command=='report':
            from .reporting import collect,audit
            if (root/'audit/final_qe_results.json').exists():audit()
            else:collect()
            print((root/'REPORT.md').read_text())
        elif args.command=='validate':
            from .controller import validate
            from .reporting import audit
            from .provenance import verify_run
            verify_run();validate();audit();print(str(root/'REPORT.md'))
        elif args.command=='doctor':
            from .doctor import inspect
            result=inspect(CONFIG);print(json.dumps(result,indent=2));return 0 if result['passed'] else 1
        elif args.command=='import-calibration':
            from .assets import import_calibration
            import_calibration(args.source,args.destination);print(str(args.destination))
        elif args.command=='calibrate':
            from .calibration import fit_manifest
            fit_manifest(args.manifest,args.destination);print(str(args.destination))
        elif args.command=='configure':
            if args.output.exists():raise FileExistsError(args.output)
            from .config import DEFAULT
            import copy
            cfg=copy.deepcopy(DEFAULT);cfg.update(method=args.method,demo=args.demo,
                output_dir='runs/'+args.method,iterations=2 if args.demo else 100,
                tasks=['wbg'] if args.demo else cfg['tasks'])
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(cfg,indent=2)+'\n');print(str(args.output))
        return 0
    except (ValueError,RuntimeError,FileNotFoundError,FileExistsError) as exc:
        ap.exit(2,type(exc).__name__+': '+str(exc)+'\n')

if __name__=='__main__':raise SystemExit(main())

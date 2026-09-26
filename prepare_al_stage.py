"""Initialize an AL experiment and its HV baseline. Never runs CFD."""
import argparse
from pathlib import Path
from impeller_app.config import AppConfig
from impeller_app.core.al_stages import prepare_stage

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,help='Source application configuration (default: saved GUI config)')
    parser.add_argument('--name',required=True,help='New experiment name; destination must not exist')
    parser.add_argument('--seed-mode',choices=['doe','combined'],default='doe',
                        help='doe: fresh paper experiment; combined: explicitly labeled warm start')
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    if Path(args.name).name != args.name or args.name in ('.','..'):
        parser.error('--name must be a directory name, not a path')
    source=AppConfig.load(args.config).resolved()
    cfg=prepare_stage(source,source.workspace.project_root/'al_stages'/args.name,args.seed_mode,args.seed)
    print('Prepared stage (no CFD started):',cfg.workspace.project_root)
    print('Set IMPELLER_APP_CONFIG to',cfg.workspace.project_root/'stage_config.json')
    print('Then launch python -m impeller_app and explicitly start AL from the GUI.')

if __name__=='__main__':
    main()

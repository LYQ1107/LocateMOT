"""Link this server's existing immutable assets into a clean evaluation checkout."""
import argparse
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--asset-root',type=Path,default=Path('/data2/user/LocateMOT_GroundJEV'))
    args=parser.parse_args();root=Path(__file__).resolve().parents[3]
    for name in ('external','models','third_party','.venv-onejev-py310','.tools'):
        target=root/name;source=args.asset_root/name
        if target.exists():continue
        if not source.exists():raise FileNotFoundError(source)
        target.symlink_to(source.resolve(),target_is_directory=True)
    print('Existing server assets linked. No weights, data or environments were copied.')


if __name__=='__main__':main()

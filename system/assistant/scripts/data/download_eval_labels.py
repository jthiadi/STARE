"""Fetch small official IndustReal annotations; NEVER imported by monitor inference.

PSR evaluates procedure-step completion, AR evaluates action intervals, and OD
evaluates the whole-assembly bounding box/state.  IndustReal does not publish
individual-part masks for this recording, so none of these files may be used to
initialize or repair a SAM2 part track.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile
from .download_industreal import RemoteZip, ROOT

META = 'https://data.4tu.nl/v2/articles/b008dd74-020d-4ea4-a8ba-7bb60769d224'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--archive', nargs='+', default=['test_p1.zip'],
                   help='Archives to search, in order; the first one holding the recording is used.')
    p.add_argument('--recording', default='03_assy_0_1')
    args = p.parse_args()
    if Path(args.recording).name != args.recording:
        raise ValueError('Use a recording ID, not a path.')
    with urllib.request.urlopen(META, timeout=25) as r:
        metadata = json.load(r)
    item = archive = None
    for name in args.archive:
        candidate = next(f for f in metadata['files'] if f['name'] == name)
        opened = zipfile.ZipFile(RemoteZip(candidate['download_url']))
        if args.recording + '/AR_labels.csv' in opened.namelist():
            item, archive = candidate, opened
            break
        opened.close()
    if archive is None:
        raise ValueError(f'{args.recording} not found in {args.archive}')
    folder = ROOT/'data/evaluation/official'/args.recording
    folder.mkdir(parents=True, exist_ok=True)
    provenance = {'metadata': META, 'archive': item, 'recording': args.recording, 'files': [],
                  'use': 'EVALUATION ONLY; never model inputs or reference-gallery labels'}
    with archive:
        for name in (
            'PSR_labels.csv',
            'PSR_labels_raw.csv',
            'PSR_labels_with_errors.csv',
            'AR_labels.csv',
            'OD_labels.json',
        ):
            member = args.recording+'/'+name
            try:
                info = archive.getinfo(member)
            except KeyError:
                print(f'{name}: missing in archive (skipped)', flush=True)
                continue
            if info.file_size > 500_000:
                raise ValueError(f'Unexpected annotation size for {name}: {info.file_size}')
            content = archive.read(info)  # zipfile checks CRC
            target = folder/name
            if target.exists() and target.read_bytes() != content:
                raise ValueError('Existing label file differs; not overwriting.')
            if not target.exists():
                target.write_bytes(content)
            provenance['files'].append({'name': name, 'member': member, 'crc32': f'{info.CRC:08x}',
                                        'sha256': hashlib.sha256(content).hexdigest()})
            print(f'{name}: {len(content)} bytes', flush=True)
    (folder/'source.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()

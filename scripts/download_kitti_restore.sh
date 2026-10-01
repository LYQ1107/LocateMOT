#!/usr/bin/env bash
set -euo pipefail
cd /data2/user/LocateMOT
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
mkdir -p data/downloads outputs/restore_20261001
archive=data/downloads/data_tracking_image_2.zip
url=https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_image_2.zip
date -Is > outputs/restore_20261001/kitti_download_started.txt
curl --noproxy '*' --fail --location --continue-at - --retry 3 --connect-timeout 30 --speed-time 180 --speed-limit 1024 --output "$archive" "$url"
python3 - "$archive" <<'PY'
import json, sys, zipfile
from pathlib import Path
p = Path(sys.argv[1])
assert p.stat().st_size == 15813146295, p.stat().st_size
with zipfile.ZipFile(p) as z:
    bad = z.testzip()
    assert bad is None, bad
    images = [i for i in z.infolist() if i.filename.startswith('training/image_02/') and i.filename.endswith('.png')]
    assert images
    total = sum(i.file_size for i in images)
    Path('outputs/restore_20261001/kitti_archive_validation.json').write_text(json.dumps({'status':'archive_valid','bytes':p.stat().st_size,'training_images':len(images),'training_uncompressed_bytes':total,'test_images_extracted':False},indent=2)+'\n')
    z.extractall('data/kitti_tracking', members=[i.filename for i in images])
Path('outputs/restore_20261001/kitti_download_complete.txt').write_text('download, ZIP CRC validation and training-image extraction complete\n')
PY

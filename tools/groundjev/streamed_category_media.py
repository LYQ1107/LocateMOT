"""Input IO only: gzip rows and exact native category crop sheets on demand."""
import gzip
import json
from pathlib import Path

IO_PROTOCOL='gzip-jsonl-and-native-category-crop-recipes-v1'
RECIPE_KEY='groundjev_native_category_crops'


def install():
    from train import common,mm
    from PIL import Image
    from locatemot.groundjev.category import NativeCategoryHead
    native_rows,native_media=common.load_rows,mm.load_media

    def load_rows(path,limit=None,split=None):
        if Path(path).suffix!='.gz':
            return native_rows(path,limit=limit,split=split)
        rows=[]
        with gzip.open(path,'rt',encoding='utf-8') as source:
            for line in source:
                line=line.strip()
                if not line:
                    continue
                row=json.loads(line)
                if split is not None and row.get('split')!=split:
                    continue
                rows.append(row)
                if limit is not None and len(rows)>=limit:
                    break
        return rows

    def load_media(media):
        if not any(RECIPE_KEY in item for item in media):
            return native_media(media)
        images,videos,metas=[],[],[]
        for item in media:
            if RECIPE_KEY in item:
                assert item['type']=='image'
                with Image.open(mm.MM_ROOT/item['path']) as source:
                    image=source.convert('RGB')
                sheet=NativeCategoryHead(None).crop_sheet(image,[{'box':box} for box in item[RECIPE_KEY]])
                assert sheet.size==(item['w'],item['h'])
                images.append(sheet)
            else:
                im,vi,md=native_media([item]);images+=im;videos+=vi;metas+=md
        return images,videos,metas

    common.load_rows,mm.load_media=load_rows,load_media
    return native_rows,native_media

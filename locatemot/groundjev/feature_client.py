"""GT-free native feature RPC across the isolated Ground/OneJev environments."""
import base64
import io
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen


class NativeFeatureClient:
    def __init__(self,url): self.url=url.rstrip('/')

    def __call__(self,image,candidates,image_path=None):
        start=time.perf_counter()
        if image_path is not None: payload=Path(image_path).read_bytes()
        else:
            buffer=io.BytesIO();image.save(buffer,format='PNG');payload=buffer.getvalue()
        body={'image_base64':base64.b64encode(payload).decode(),'boxes':[c['box'] for c in candidates]}
        request=Request(self.url+'/features',json.dumps(body).encode(),{'Content-Type':'application/json'})
        with urlopen(request,timeout=600) as response: result=json.load(response)
        if len(result['region_features'])!=len(candidates): raise ValueError('Native feature rows changed candidate order/count')
        enriched=[]
        for index,candidate in enumerate(candidates):
            enriched.append(dict(candidate,native_region_feature=result['region_features'][index],
                temporal_region_feature=result['temporal_features'][index],
                feature_source=result['metadata']['feature_source']))
        metadata=dict(result['metadata'],rpc_seconds=time.perf_counter()-start,
            global_feature_dimension=len(result['global_feature']),local_tokens_shape=result['local_tokens_shape'])
        return enriched,metadata

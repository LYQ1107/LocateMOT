"""Serialize native DLM visual requests; never cache a source dataset."""
import argparse
import base64
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    import torch
    from torch import nn
    from PIL import Image
    from locatemot.groundjev.native_features import NativeGroundFeatures
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=8111)
    parser.add_argument('--temporal-checkpoint',type=Path)
    parser.add_argument('--receipt',type=Path,required=True)
    args=parser.parse_args()
    extractor=NativeGroundFeatures()
    temporal=None
    if args.temporal_checkpoint:
        checkpoint=torch.load(args.temporal_checkpoint,map_location='cuda:0',weights_only=True)
        temporal=nn.Sequential(nn.LayerNorm(2560),nn.Linear(2560,256),nn.GELU(),nn.Linear(256,128)).to('cuda:0')
        temporal.load_state_dict(checkpoint['adapter_state_dict']);temporal.eval().requires_grad_(False)
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            start=time.perf_counter()
            body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            with Image.open(io.BytesIO(base64.b64decode(body['image_base64']))) as source: image=source.convert('RGB')
            spatial,metadata=extractor.encode(image)
            regions,local=extractor.pool(spatial,metadata,body['boxes'])
            with torch.no_grad():
                adapted=temporal(regions) if temporal is not None else None
            result={'region_features':regions.float().cpu().tolist(),
                'temporal_features':adapted.float().cpu().tolist() if adapted is not None else [None]*len(regions),
                'global_feature':spatial.float().mean(dim=(-1,-2)).squeeze(0).cpu().tolist(),
                'local_tokens_shape':list(local.shape),'metadata':dict(metadata,
                    temporal_checkpoint=str(args.temporal_checkpoint) if args.temporal_checkpoint else None,
                    temporal_adapter_claim='auxiliary region representation, not JEV or a trained DLM decoder',
                    seconds=time.perf_counter()-start,full_dataset_cache=False,GT_read=False)}
            encoded=json.dumps(result).encode()
            self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(encoded)))
            self.end_headers();self.wfile.write(encoded)
            print(json.dumps({'stage':'real_native_feature_request','regions':len(regions),
                'seconds':result['metadata']['seconds'],'region_dimension':extractor.feature_dimension}),flush=True)
    server=HTTPServer(('127.0.0.1',args.port),Handler)
    receipt={'status':'NATIVE_FEATURE_RPC_READY','url':f'http://127.0.0.1:{args.port}',
        'checkpoint':str(extractor.checkpoint),'native_parameters':extractor.parameter_count,
        'region_dimension':extractor.feature_dimension,'temporal_checkpoint':str(args.temporal_checkpoint) if args.temporal_checkpoint else None,
        'dataset_cache_created':False,'request_concurrency':1,'language_decoder_loaded':False}
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
    server.serve_forever()


if __name__=='__main__':main()

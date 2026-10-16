"""Frame-wide sparse physical competition, independent of the query text."""
import numpy as np


def box_iou(a,b):
    a=np.asarray(a,dtype=np.float32).reshape(-1,4);b=np.asarray(b,dtype=np.float32).reshape(-1,4)
    lt=np.maximum(a[:,None,:2],b[None,:,:2]);rb=np.minimum(a[:,None,2:],b[None,:,2:])
    wh=np.maximum(rb-lt,0);intersection=wh.prod(-1)
    area_a=np.maximum(a[:,2:]-a[:,:2],0).prod(-1)
    area_b=np.maximum(b[:,2:]-b[:,:2],0).prod(-1)
    return intersection/np.maximum(area_a[:,None]+area_b[None,:]-intersection,1e-8)


def sparse_competition(boxes,features,neighbors=32,block=32):
    """Union of overlap, center and appearance neighborhoods; at most96N edges.

    Score blocks are32xN, never a stored whole-frame NxN matrix. Equal-score
    choices use canonical box/feature ordering before original-index fallback.
    """
    boxes=np.asarray(boxes,dtype=np.float32).reshape(-1,4)
    n=len(boxes)
    if n<2:return np.empty((0,2),np.int64),np.empty((0,6),np.float32)
    features=np.asarray(features,dtype=np.float32).reshape(n,-1)
    norm=features/np.maximum(np.linalg.norm(features,axis=1,keepdims=True),1e-8)
    center=(boxes[:,:2]+boxes[:,2:])/2
    scale=max(float(np.ptp(center,axis=0).max()),1.)
    canonical=np.lexsort((np.arange(n),features[:,0],boxes[:,3],boxes[:,2],boxes[:,1],boxes[:,0]))
    edges=[];values=[];k=min(neighbors,n-1)
    for start in range(0,n,block):
        end=min(n,start+block);iou=box_iou(boxes[start:end],boxes)
        distance=np.linalg.norm(center[start:end,None]-center[None],axis=-1)/scale
        cosine=norm[start:end]@norm.T
        for local,i in enumerate(range(start,end)):
            candidates=canonical[canonical!=i]
            selected=set()
            for score in (iou[local],-distance[local],cosine[local]):
                selected.update(candidates[np.argsort(-score[candidates],kind='stable')[:k]].tolist())
            for j in sorted(selected):
                a=np.maximum(boxes[i,2:]-boxes[i,:2],0).prod();b=np.maximum(boxes[j,2:]-boxes[j,:2],0).prod()
                edges.append((i,j));values.append((iou[local,j],distance[local,j],cosine[local,j],
                    np.log(max(a,1)/max(b,1)),float(iou[local,j]>.5),float(iou[local,j]>.1)))
    return np.asarray(edges,np.int64),np.asarray(values,np.float32)

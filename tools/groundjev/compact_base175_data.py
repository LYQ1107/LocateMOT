"""Instantaneous Base teacher ROI inputs; no invented temporal supervision."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


class CompactBaseTeacherInputs:
    def __init__(self,directory):
        import numpy as np
        from locatemot.groundjev.text_features import OneJevTextFeatures
        self.directory=Path(directory)
        proof=json.loads((self.directory/'feature_validation.json').read_text())
        assert proof['status']=='SELECTED_BASE175_NATIVE_FP32_TEACHER_BANK_COMPLETE'
        self.objects=[json.loads(line) for line in (self.directory/'objects.jsonl').open()]
        self.bank=np.load(self.directory/'regions.npy',mmap_mode='r',allow_pickle=False)
        assert self.bank.shape==(len(self.objects),2560) and self.bank.dtype==np.float32
        plan=json.loads((self.directory/'registration.json').read_text())
        categories=json.loads(Path(plan['vocabulary']).read_text())['categories']
        self.vocabulary=[c['name'].replace('_',' ') for c in categories]
        assert len(self.vocabulary)==1203
        self.specification=dict(task='ovmot',vocabulary=self.vocabulary,expression=None)
        self.text=OneJevTextFeatures()
        self.spec_text=self.text.specification(self.specification)
        self.class_text=self.text.vocabulary(self.vocabulary)
        mapping={row['annotation_id']:row['feature_row'] for row in self.objects}
        self.sampled_draw_rows=[mapping[aid] for aid in json.loads((self.directory/'sampled_draws.json').read_text())]

    def inputs(self,index):
        import numpy as np
        import torch
        row=self.objects[index];assert row['feature_row']==index
        width,height=row['image_size'];box=row['box']
        return dict(candidates=torch.from_numpy(np.array(self.bank[index:index+1],copy=True)),
            history=torch.empty(0,3,2560),
            candidate_geo=torch.tensor([[box[0]/width,box[1]/height,box[2]/width,box[3]/height]]),
            history_geo=torch.empty(0,8),pair_geometry=torch.empty(1,0,2),task_index=2,
            spec_text=self.spec_text,class_text=self.class_text)

    def class_target(self,index):
        # Labels are outside inputs(). No NEW/lifecycle/reliability target is invented.
        return self.objects[index]['class_index']

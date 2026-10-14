"""Real frozen OneJev input token features; not language-decoder reasoning."""
from pathlib import Path


class OneJevTextFeatures:
    def __init__(self, checkpoint=None):
        from safetensors import safe_open
        from transformers import AutoTokenizer
        root=Path(__file__).resolve().parents[2]
        self.checkpoint=Path(checkpoint or root/'models/OneJev-0.8B')
        self.tokenizer=AutoTokenizer.from_pretrained(self.checkpoint,local_files_only=True)
        with safe_open(self.checkpoint/'model.safetensors',framework='pt') as f:
            self.embedding=f.get_tensor('model.language_model.embed_tokens.weight')
        self.dimension=self.embedding.shape[1]

    def __call__(self,text):
        import torch
        ids=self.tokenizer.encode(text,add_special_tokens=False)
        if not ids:ids=[self.tokenizer.eos_token_id]
        return self.embedding[torch.tensor(ids)].float().mean(0)

    def specification(self,specification):
        expression=specification.get('expression') or ''
        return self('Task '+specification['task']+'. '+expression+'. Categories: '+', '.join(specification['vocabulary']))

    def vocabulary(self,names):
        import json,torch
        root=Path('/data2/user/LocateMOT/outputs/unified/u2/protocol/lvis_v1_vocabulary.json')
        categories=json.loads(root.read_text())['categories']
        descriptions={c['name'].replace('_',' '):c for c in categories}
        vectors=[]
        for name in names:
            c=descriptions.get(name)
            text=name if c is None else name+'; '+', '.join(c['synonyms'])+'; '+c['def']
            vectors.append(self(text))
        return torch.stack(vectors)

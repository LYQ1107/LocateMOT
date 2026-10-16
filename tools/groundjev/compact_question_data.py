"""Question-level supervision over unchanged native Compact tensors and heads."""
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class CompactQuestionData:
    def __init__(self, directory):
        from tools.groundjev.compact_base175_data import CompactBaseTeacherInputs
        from tools.groundjev.compact_reference_data import CompactReferenceInputs
        self.directory = Path(directory)
        self.plan = json.loads((self.directory / 'registration.json').read_text())
        self.summary = json.loads((self.directory / 'summary.json').read_text())
        assert self.summary['status'] == 'BROADER_COMPACT_TYPED_QUESTION_CURRICULUM_PREPARED'
        base_proof = json.loads((ROOT / self.plan['base'] / 'native_input_validation.json').read_text())
        assert base_proof['all_native_forward_inputs_and_outputs_exact']
        self.base = CompactBaseTeacherInputs(ROOT / self.plan['base'])
        self.native = CompactReferenceInputs(ROOT / self.plan['inputs'])
        self.native.text = self.base.text
        with gzip.open(self.directory / 'questions.jsonl.gz', 'rt') as file:
            self.rows = [json.loads(line) for line in file]
        assert len(self.rows) == 3 * self.plan['optimizer_steps']
        for offset, row in enumerate(self.rows):
            assert row['step'] == offset // 3 + 1
            assert row['unit'].split('/')[0] == ('mot', 'rmot', 'ovmot')[offset % 3]

    def inputs(self, row):
        if row['task'] == 'category': return self.base.inputs(row['object_row'])
        return self.native.inputs(row['unit'], row['group'])

    @staticmethod
    def loss(outputs, row):
        import torch
        import torch.nn.functional as F
        task = row['task']; slot = row.get('slot', 0)
        if task == 'category':
            return F.cross_entropy(outputs['category'][0:1], torch.tensor([row['class_target']]))
        if task in ('memory_update', 'query_membership'):
            head = 'update' if task == 'memory_update' else 'membership'
            return F.binary_cross_entropy_with_logits(outputs[head][slot], torch.tensor(row['target'][1]))
        head = dict(identity='match', lifecycle='life', reliability='reliability')[task]
        logits = outputs[head][slot]
        target = torch.tensor(row['target'], dtype=logits.dtype)
        assert target.shape == logits.shape
        return -(target * F.log_softmax(logits, dim=-1)).sum()

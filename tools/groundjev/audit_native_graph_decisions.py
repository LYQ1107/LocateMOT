"""Audit fixed native graph probe answers without another model/GPU run."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def assignments(answers):
    rows = [answers[key]['probabilities'] for key in sorted(
        (k for k in answers if k.startswith('match_')), key=lambda k: int(k.split('_')[1]))]
    actions = ('NEW', 'IGNORE', 'NO_MATCH')
    ids = sorted({key for row in rows for key in row if key not in actions}, key=int)
    costs = np.full((len(rows), len(ids) + 3 * len(rows)), 1e9)
    for i, row in enumerate(rows):
        for j, tid in enumerate(ids):
            if tid in row:
                costs[i, j] = -np.log(max(row[tid], 1e-12))
        for j, action in enumerate(actions):
            costs[i, len(ids) + 3 * i + j] = -np.log(max(row[action], 1e-12))
    indices, columns = linear_sum_assignment(costs)
    return {str(i): ids[j] if j < len(ids) else actions[(j-len(ids)) % 3]
            for i, j in zip(indices.tolist(), columns.tolist())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe', type=Path, required=True)
    args = parser.parse_args()
    eager = json.loads((args.probe/'eager_results.json').read_text())
    graphs = json.loads((args.probe/'graphs_results.json').read_text())
    cases = {}
    for name, reference in eager.items():
        a, b = reference['answers'], graphs[name]['answers']
        assert a.keys() == b.keys()
        choices = {key: [row['choice'], b[key]['choice']] for key, row in a.items()
                   if row['type'] == 'choice' and row['choice'] != b[key]['choice']}
        gates = {key: [row['noul'], b[key]['noul']] for key, row in a.items()
                 if row['type'] == 'noul' and (row['noul'] >= .5) != (b[key]['noul'] >= .5)}
        assigned_a, assigned_b = assignments(a), assignments(b)
        changed = {key: [value, assigned_b[key]] for key, value in assigned_a.items()
                   if value != assigned_b[key]}
        cases[name] = dict(questions=len(a), choice_questions=sum(r['type']=='choice' for r in a.values()),
            noul_questions=sum(r['type']=='noul' for r in a.values()),
            global_match_rows=len(assigned_a), choice_changes=choices, noul_half_threshold_changes=gates,
            eager_assignment=assigned_a, graph_assignment=assigned_b, global_assignment_changes=changed)
    result = dict(status='SAVED_NATIVE_GRAPH_ANSWERS_RUNTIME_DECISION_AUDIT_COMPLETE', cases=cases,
        tested_decisions_equal=all(not c['choice_changes'] and not c['noul_half_threshold_changes']
                                  and not c['global_assignment_changes'] for c in cases.values()),
        original_numerical_and_speed_rejection_retained=True, adopted=False, new_GPU_or_optimizer_updates=0,
        scope='Posthoc same four saved requests only; tracker -log probability Hungarian with private '
              'NEW/IGNORE/NO_MATCH and Noul>=.5. No RMOT membership or assigned-update requests in this probe. '
              'Score expectations remain continuous reliability outputs, not an invented argmax gate. '
              'Not full-video trajectory, future memory, hierarchical category routing or E2E parity.')
    (args.probe/'runtime_decision_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()

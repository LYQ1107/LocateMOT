"""Explicit new-stage index shuffle; native rendering, batches and loss remain intact."""
import random


def index_shuffle_plan(native_plan):
    def plan(bundle, rows, args, epoch, rank=0, world=1):
        batches, dropped, lengths = native_plan(bundle, rows, args, epoch, rank, world)
        assert world >= 1 and args.max_rows == 1 and dropped == 0
        order = list(range(len(rows)))
        random.Random(args.seed+epoch).shuffle(order)
        return [[index] for index in order], dropped, lengths
    return plan

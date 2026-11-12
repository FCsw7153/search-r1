"""Pure metric helpers: exact finite-sample pass@k; never bootstrap best-of aliases."""
from collections import defaultdict
from math import comb


def pass_metrics(sources, uids, accuracies, expected_counts, ks=(1, 5)):
    groups = defaultdict(list)
    for source, uid, acc in zip(sources, uids, accuracies, strict=True):
        if acc not in (0, 1, 0.0, 1.0):
            raise ValueError(f'pass@k requires binary EM, got {acc!r}')
        groups[(str(source), str(uid))].append(int(acc))
    expected = expected_counts or {key: len(values) for key, values in groups.items()}
    sums, counts = defaultdict(float), defaultdict(int)
    for key, n in expected.items():
        values = groups.get(key, [])
        if len(values) > n:
            raise ValueError('More completed responses than scheduled responses')
        c = sum(values)  # Missing/failed sessions count as incorrect, never silently dropped.
        for k in ks:
            if not 1 <= k <= n:
                raise ValueError(f'pass@{k} requires at least {k} samples, got {n}')
            value = 1.0 - (comb(n-c, k) / comb(n, k) if n-c >= k else 0.0)
            for source in (key[0], 'all'):
                name = f'eval/{source}/pass@{k}'
                sums[name] += value
                counts[name] += 1
    result = {name: value / counts[name] for name, value in sums.items()}
    result['eval/questions'] = len(expected)
    result['eval/missing_responses'] = sum(n-len(groups.get(key, [])) for key, n in expected.items())
    return result

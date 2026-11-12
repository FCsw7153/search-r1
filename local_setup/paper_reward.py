"""Paper outcome-only binary exact match; no format bonus or tag-count penalty."""
from verl.utils.reward_score.search_r1_like_qa_em import em_check, extract_solution


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    answer = extract_solution(solution_str)
    acc = float(answer is not None and em_check(answer, ground_truth['target']))
    return {'score': acc, 'acc': acc}

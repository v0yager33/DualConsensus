from typing import List, Dict, Tuple
from collections import Counter
import numpy as np
from verl.utils.reward_score.dcrl_math import extract_answer, simplify_expression_string, grade
from verl import DataProto


def _is_empty(s: str) -> bool:
    """Empty value check."""
    if s is None:
        return True
    if not isinstance(s, str):
        return True
    return s.strip() == ""


def select_top_k_per_prompt_simple(data, n_votes_per_prompt, n_samples_per_prompt):
    """
    Select the first k rollouts per prompt, used for DCRL downsampling.
    """
    assert len(data) % n_votes_per_prompt == 0, "data length must be divisible by n_votes_per_prompt"
    num_prompts = len(data) // n_votes_per_prompt

    selected_indices = []
    for i in range(num_prompts):
        start = i * n_votes_per_prompt
        selected_indices.extend(range(start, start + n_samples_per_prompt))

    return data[selected_indices]


def select_top_k_per_prompt(gen_batch_anchor, gen_batch_explorer, n_votes_anchor, n_votes_explorer,
                            n_samples_anchor, n_samples_explorer, n_samples_per_prompt):
    """
    Select the first k rollouts per prompt from anchor and explorer.
    """
    assert n_samples_per_prompt == n_samples_anchor + n_samples_explorer, \
        "n_samples_per_prompt must equal to n_samples_anchor plus n_samples_explorer"

    if n_samples_explorer == 0:
        anchor_selected = select_top_k_per_prompt_simple(
            gen_batch_anchor, n_votes_anchor, n_samples_anchor
        )
        return anchor_selected
    elif n_samples_anchor == 0:
        explorer_selected = select_top_k_per_prompt_simple(
            gen_batch_explorer, n_votes_explorer, n_samples_explorer
        )
        return explorer_selected

    anchor_selected = select_top_k_per_prompt_simple(
            gen_batch_anchor, n_votes_anchor, n_samples_anchor
        )
    explorer_selected = select_top_k_per_prompt_simple(
            gen_batch_explorer, n_votes_explorer, n_samples_explorer
        )

    # Merge anchor and explorer samples per prompt
    assert len(anchor_selected) % n_samples_anchor == 0, "anchor_selected length must be divisible by n_samples_anchor"
    assert len(explorer_selected) % n_samples_explorer == 0, "explorer_selected length must be divisible by n_samples_explorer"
    
    num_prompts = len(anchor_selected) // n_samples_anchor
    assert num_prompts == len(explorer_selected) // n_samples_explorer, \
        "Number of prompts must be the same for both anchor and explorer"
    
    # Reorganize the data to ensure the correct pattern per prompt
    final_indices = []
    for i in range(num_prompts):
        # Add anchor samples for this prompt
        anchor_start = i * n_samples_anchor
        final_indices.extend(range(anchor_start, anchor_start + n_samples_anchor))
        
        # Add explorer samples for this prompt
        explorer_start = i * n_samples_explorer
        explorer_offset = len(anchor_selected)  # Offset to access explorer samples
        for j in range(n_samples_explorer):
            final_indices.append(explorer_start + j + explorer_offset)
    
    # Combine both datasets and reorder according to the calculated indices
    combined_data = DataProto.concat([anchor_selected, explorer_selected])
    return combined_data[final_indices]


def apply_original_gt(batch):
    """
    Apply the original ground truth to the batch.
    """
    for i in range(len(batch)):
        data_item = batch[i]
        original_gt = data_item.non_tensor_batch["reward_model"]["original_gt"]
        # Filter out empty values from the original GT
        if _is_empty(original_gt):
            data_item.non_tensor_batch["reward_model"]["ground_truth"] = "None"
        else:
            data_item.non_tensor_batch["reward_model"]["ground_truth"] = original_gt

    return batch


def apply_dcrl_gt(batch, gen_batch_1, gen_batch_2, n_votes_1, n_votes_2, harmony_enable: bool, tokenizer):
    """
    Apply the DCRL ground truth to the batch.
    """
    # Handle None for gen_batch_2
    if gen_batch_1 is None:
        gen_batch_1 = []
    if gen_batch_2 is None:
        gen_batch_2 = []

    # Calculate number of prompts from non-empty batch
    num_prompts = 0
    if len(gen_batch_1) > 0:
        num_prompts = len(gen_batch_1) // n_votes_1
        assert len(gen_batch_1) == num_prompts * n_votes_1, \
            f"gen_batch_1 length ({len(gen_batch_1)}) must be num_prompts × n_votes_1 (×{n_votes_1})"
    elif len(gen_batch_2) > 0:
        num_prompts = len(gen_batch_2) // n_votes_2
        assert len(gen_batch_2) == num_prompts * n_votes_2, \
            f"gen_batch_2 length ({len(gen_batch_2)}) must be num_prompts × n_votes_2 (×{n_votes_2})"

    assert len(batch) == num_prompts, f"Batch length ({len(batch)}) must equal number of prompts ({num_prompts})"

    # Extract all responses for batch 1 and batch 2
    batch1_responses = _extract_all_responses(gen_batch_1, tokenizer) if gen_batch_1 else []
    batch2_responses = _extract_all_responses(gen_batch_2, tokenizer) if gen_batch_2 else []

    # Process votes in batch
    harmonic_gt_list, harmonic_ratio_list, is_harmonic_valid_list = _batch_harmonic_vote(
        batch1_responses, batch2_responses, n_votes_1, n_votes_2, num_prompts
    )
    majority_gt_list, majority_ratio_list = _batch_majority_vote(batch1_responses + batch2_responses,
                                                                 n_votes_1 + n_votes_2, num_prompts)

    # anchor majority vote
    anchor_gt_list, anchor_ratio_list = _batch_majority_vote(batch1_responses, n_votes_1, num_prompts)

    # Update batch with results
    for i in range(num_prompts):
        data_item = batch[i]
        original_gt = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        # Filter out empty values from the original GT
        if _is_empty(original_gt):
            original_gt = "None"

        # Filter out empty values from the voting results
        harmonic_gt = harmonic_gt_list[i] if not _is_empty(harmonic_gt_list[i]) else "None"
        majority_gt = majority_gt_list[i] if not _is_empty(majority_gt_list[i]) else "None"
        anchor_gt = anchor_gt_list[i] if not _is_empty(anchor_gt_list[i]) else "None"

        # Update ground truth
        if harmony_enable:
            data_item.non_tensor_batch["reward_model"]["ground_truth"] = harmonic_gt
            data_item.non_tensor_batch["reward_model"]["dcrl_truth"] = harmonic_gt
        else:
            data_item.non_tensor_batch["reward_model"]["ground_truth"] = majority_gt
            data_item.non_tensor_batch["reward_model"]["dcrl_truth"] = majority_gt

        # Store all computed values
        data_item.non_tensor_batch["reward_model"]["majority_gt"] = majority_gt
        data_item.non_tensor_batch["reward_model"]["original_gt"] = original_gt

        data_item.non_tensor_batch["reward_model"]["anchor_majority_gt"] = anchor_gt
        data_item.non_tensor_batch["reward_model"]["anchor_majority_ratio"] = anchor_ratio_list[i]

        # Calculate enhanced_reward: 1.0 if harmonic vote is valid and result differs from anchor majority vote, else 0.0
        if harmony_enable and (grade(harmonic_gt, anchor_gt) == False) and harmonic_gt != "None" and anchor_gt != "None":
            data_item.non_tensor_batch["reward_model"]["enhanced_reward"] = 1.0
        else:
            data_item.non_tensor_batch["reward_model"]["enhanced_reward"] = 0.0

    # Consensus ratio
    batch.non_tensor_batch["consensus_ratio_list"] = np.array(harmonic_ratio_list,
                                                              dtype=float) if harmony_enable else np.array(
        majority_ratio_list, dtype=float)
    # anchor majority ratio
    batch.non_tensor_batch["anchor_majority_ratio_list"] = np.array(anchor_ratio_list, dtype=float)
    
    # Store enhanced_reward list for subsequent metric calculation
    batch.non_tensor_batch["enhanced_reward_list"] = [
        batch[i].non_tensor_batch["reward_model"]["enhanced_reward"] for i in range(num_prompts)
    ]

    return batch


def apply_dcrl_gt_majority(batch, gen_batch_output, n, tokenizer):
    """
    Apply the majority vote ground truth to the batch.
    """
    assert len(gen_batch_output) % n == 0, "gen_batch_output length must be divisible by n"
    num_prompts = len(gen_batch_output) // n
    assert len(batch) == num_prompts, "batch length must be equal to the number of prompts"

    model_outputs = []
    for i in range(num_prompts):
        start = i * n
        for j in range(n):
            data_item = gen_batch_output[start + j]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            model_outputs.append(response_str)

    # Pass num_prompts parameter to _batch_majority_vote
    majority_gt_list, majority_ratio_list = _batch_majority_vote(model_outputs, n, num_prompts)

    assert len(batch) == len(majority_gt_list), "batch length must be equal to the number of model outputs"

    for i in range(num_prompts):
        data_item = batch[i]
        original_gt = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        # Filter out empty values from the original GT
        if _is_empty(original_gt):
            original_gt = "None"
        # Filter out empty values from the voting results
        majority_gt = majority_gt_list[i] if not _is_empty(majority_gt_list[i]) else "None"
        
        data_item.non_tensor_batch["reward_model"]["ground_truth"] = majority_gt
        data_item.non_tensor_batch["reward_model"]["majority_gt"] = majority_gt
        data_item.non_tensor_batch["reward_model"]["anchor_majority_gt"] = majority_gt
        data_item.non_tensor_batch["reward_model"]["dcrl_truth"] = majority_gt
        data_item.non_tensor_batch["reward_model"]["original_gt"] = original_gt
        # enhanced_reward is 0.0 in pure majority voting scenario
        data_item.non_tensor_batch["reward_model"]["enhanced_reward"] = 0.0

    # Ensure batch has non_tensor_batch attribute
    if not hasattr(batch, 'non_tensor_batch'):
        batch.non_tensor_batch = {}
    batch.non_tensor_batch["consensus_ratio_list"] = np.array(majority_ratio_list, dtype=float)
    batch.non_tensor_batch["anchor_majority_ratio_list"] = np.array(majority_ratio_list, dtype=float)
    batch.non_tensor_batch["enhanced_reward_list"] = [0.0 for _ in range(num_prompts)]

    return batch

def set_all_enhanced_reward_to_zero(batch):
    """
    Set the 'enhanced_reward' field of all data items in the batch to 0.0.
    """
    for data_item in batch:
        if 'reward_model' not in data_item.non_tensor_batch:
            data_item.non_tensor_batch['reward_model'] = {}
        
        # Set the value of 'enhanced_reward' to 0.0
        data_item.non_tensor_batch['reward_model']['enhanced_reward'] = 0.0

    batch.non_tensor_batch['enhanced_reward_list'] = [
        item.non_tensor_batch['reward_model']['enhanced_reward'] for item in batch
    ]
    return batch


def _extract_all_responses(data_items, tokenizer):
    """
    Extract valid response strings from all data items in a batch.
    """
    responses = []
    for data_item in data_items:
        prompt_length = data_item.batch["prompts"].shape[-1]
        attn_mask = data_item.batch["attention_mask"][prompt_length:]
        valid_len = attn_mask.sum()
        valid_ids = data_item.batch["responses"][:valid_len]
        response_str = tokenizer.decode(valid_ids, skip_special_tokens=True)
        responses.append(response_str)
    return responses


def _batch_harmonic_vote(batch1_resp: List[str], batch2_resp: List[str], n_votes_1: int, n_votes_2: int,
                         num_prompts: int) -> Tuple[List[str], List[float], List[bool]]:
    """
    Batch process harmonic vote calculation for all prompts.
    The return value includes an additional is_harmonic_valid_list: marking whether the harmonic vote for each prompt is valid (did not fall back to majority vote).
    """
    harmonic_gt_list = []
    harmonic_ratio_list = []
    is_harmonic_valid_list = []  # New: Marks if harmonic vote is valid (did not fall back)

    for i in range(num_prompts):
        # Get responses for current prompt
        start1, end1 = i * n_votes_1, (i + 1) * n_votes_1
        start2, end2 = i * n_votes_2, (i + 1) * n_votes_2

        prompt1_resp = batch1_resp[start1:end1] if n_votes_1 > 0 else []
        prompt2_resp = batch2_resp[start2:end2] if n_votes_2 > 0 else []

        # Calculate harmonic vote for this prompt
        harmonic_gt, harmonic_ratio, is_harmonic_valid = _harmonic_vote(prompt1_resp, prompt2_resp)
        # Final filtering for empty values
        harmonic_gt = harmonic_gt if not _is_empty(harmonic_gt) else "None"
        harmonic_gt_list.append(harmonic_gt)
        harmonic_ratio_list.append(harmonic_ratio)
        is_harmonic_valid_list.append(is_harmonic_valid)

    return harmonic_gt_list, harmonic_ratio_list, is_harmonic_valid_list


def _harmonic_vote(batch1_resp: List[str], batch2_resp: List[str]) -> Tuple[str, float, bool]:
    """
    Calculate harmonic vote and its ratio for responses from two batches.
    The ratio represents the total proportion of the selected answer across both batches.
    An additional return value is_harmonic_valid: marks whether it is a valid harmonic vote (did not fall back to majority vote).
    """
    count1 = _answer_counter(batch1_resp) if batch1_resp else {}
    count2 = _answer_counter(batch2_resp) if batch2_resp else {}
    total1, total2 = len(batch1_resp), len(batch2_resp)
    total_all = total1 + total2

    harmonic_gt = "None"
    harmonic_ratio = 0.0
    is_harmonic_valid = False 

    if total1 > 0 and total2 > 0:
        # Select answer using harmonic mean voting
        harmonic_gt = _harmonic_vote_gt(count1, count2, total1, total2)
        # Filter out empty values
        harmonic_gt = harmonic_gt if not _is_empty(harmonic_gt) else "None"

        # Calculate total proportion of the selected answer
        count_total = count1.get(harmonic_gt, 0) + count2.get(harmonic_gt, 0)
        harmonic_ratio = count_total / total_all if total_all > 0 else 0.0

        # Determine if it's a valid harmonic vote (did not fall back to majority vote)
        if harmonic_ratio > 0 and count1.get(harmonic_gt, 0) > 0 and count2.get(harmonic_gt, 0) > 0:
            is_harmonic_valid = True
        else:
            # If no valid harmonic consensus, fallback to batch1 majority vote
            is_harmonic_valid = False
            if count1:
                # Filter out empty values in count1
                count1_filtered = {k: v for k, v in count1.items() if not _is_empty(k)}
                if count1_filtered:
                    harmonic_gt, count_val = Counter(count1_filtered).most_common(1)[0]
                    count_total = count_val + count2.get(harmonic_gt, 0)
                    harmonic_ratio = count_total / total_all
                else:
                    harmonic_gt = "None"
            else:
                harmonic_gt = "None"
    elif total1 > 0:
        # Only batch1 available, use majority vote
        is_harmonic_valid = False
        if count1:
            # Filter out empty values in count1
            count1_filtered = {k: v for k, v in count1.items() if not _is_empty(k)}
            if count1_filtered:
                harmonic_gt, count_val = Counter(count1_filtered).most_common(1)[0]
                harmonic_ratio = count_val / total1
            else:
                harmonic_gt = "None"
        else:
            harmonic_gt = "None"
    elif total2 > 0:
        # Only batch2 available, use majority vote
        is_harmonic_valid = False
        if count2:
            # Filter out empty values in count2
            count2_filtered = {k: v for k, v in count2.items() if not _is_empty(k)}
            if count2_filtered:
                harmonic_gt, count_val = Counter(count2_filtered).most_common(1)[0]
                harmonic_ratio = count_val / total2
            else:
                harmonic_gt = "None"
        else:
            harmonic_gt = "None"

    return harmonic_gt, harmonic_ratio, is_harmonic_valid


def _batch_majority_vote(all_responses: List[str], n: int, num_prompts: int) -> Tuple[List[str], List[float]]:
    """
    Batch process majority vote calculation for all prompts.
    """
    majority_gt_list = []
    majority_ratio_list = []

    for i in range(num_prompts):
        start, end = i * n, (i + 1) * n
        prompt_responses = all_responses[start:end]

        majority_gt, majority_ratio = _majority_vote(prompt_responses)
        majority_gt = majority_gt if not _is_empty(majority_gt) else "None"
        majority_gt_list.append(majority_gt)
        majority_ratio_list.append(majority_ratio)

    return majority_gt_list, majority_ratio_list


def _majority_vote(model_outputs: List[str]) -> Tuple[str, float]:
    """
    Calculate majority vote and ratio for a single prompt's responses.
    """
    if not model_outputs:
        return "None", 0.0  

    # Extract answers and filter out empty values
    model_answers = [extract_answer(generated_text) for generated_text in model_outputs]
    model_answers = [answer for answer in model_answers if not _is_empty(answer)]
    model_answers = [simplify_expression_string(answer) for answer in model_answers]
    model_answers = [answer for answer in model_answers if not _is_empty(answer)]

    if not model_answers:
        return "None", 0.0  

    counter = Counter(model_answers)
    majority_answer, majority_count = counter.most_common(1)[0]
    majority_ratio = majority_count / len(model_outputs)

    return majority_answer, majority_ratio


def _answer_counter(model_outputs: List[str]) -> Dict[str, int]:
    """
    Count occurrences of extracted and simplified answers from model outputs.
    """
    if len(model_outputs) == 0:
        return {}
    # Extract answers and filter out empty values
    model_answers = [extract_answer(generated_text) for generated_text in model_outputs]
    # First pass filter: None + empty string
    model_answers = [answer for answer in model_answers if not _is_empty(answer)]
    # Simplify expressions
    model_answers = [simplify_expression_string(answer) for answer in model_answers]
    # Second pass filter: empty strings after simplification
    model_answers = [answer for answer in model_answers if not _is_empty(answer)]

    return dict(Counter(model_answers))


def _harmonic_vote_gt(dict1: Dict[str, int], dict2: Dict[str, int], total1: int, total2: int) -> str:
    """
    Calculate the answer with the maximum harmonic mean of frequencies from two distributions.
    """
    # Filter out empty string keys from the dictionaries
    dict1_filtered = {k: v for k, v in dict1.items() if not _is_empty(k)}
    dict2_filtered = {k: v for k, v in dict2.items() if not _is_empty(k)}
    
    all_answers = set(dict1_filtered.keys()).union(set(dict2_filtered.keys()))
    if not all_answers:
        return "None"  # Replace empty string with "None"

    max_harmonic = -1.0
    harmonic_vote = "None"  # Initial value changed to "None"

    for answer in all_answers:
        freq1 = dict1_filtered.get(answer, 0) / total1 if total1 != 0 else 0.0
        freq2 = dict2_filtered.get(answer, 0) / total2 if total2 != 0 else 0.0

        if freq1 > 0 and freq2 > 0:
            harmonic_mean = (2 * freq1 * freq2) / (freq1 + freq2)  # 2*a*b/(a+b)
        else:
            harmonic_mean = 0.0

        if harmonic_mean > max_harmonic:
            max_harmonic = harmonic_mean
            harmonic_vote = answer

    return harmonic_vote


# === Metrics Computation ===

def compute_dcrl_metrics(batch, n):
    """
    Compute the DCRL metrics.
    Adjusted enhanced_reward related metric calculation: to count the ratio of prompts with enhanced reward.
    """
    assert len(batch) % n == 0, "batch length must be divisible by n"
    num_prompts = len(batch) // n

    idx = sorted(range(len(batch)), key=lambda x: batch[x].non_tensor_batch["extra_info"]["index"])

    dcrl_reward = []
    gt_reward = []
    dcrl_label = []
    majority_label = []
    anchor_majority_label = []
    gt_label = []
    enhanced_reward_flag = []

    for i in range(len(batch)):
        data_item = batch[idx[i]]
        dcrl_reward.append(data_item.batch["token_level_scores"].sum().item())
        gt_reward.append(data_item.batch["token_level_scores_original"].sum().item())

        # Get current DCRL label 
        dcrl_label_val = data_item.non_tensor_batch["reward_model"]["dcrl_truth"]
        dcrl_label.append(dcrl_label_val if not _is_empty(dcrl_label_val) else "None")

        # Always get majority label for comparison
        majority_label_val = data_item.non_tensor_batch["reward_model"]["majority_gt"]
        majority_label.append(majority_label_val if not _is_empty(majority_label_val) else "None")
        
        anchor_majority_label_val = data_item.non_tensor_batch["reward_model"]["anchor_majority_gt"]
        anchor_majority_label.append(anchor_majority_label_val if not _is_empty(anchor_majority_label_val) else "None")
        
        gt_label_val = data_item.non_tensor_batch["reward_model"]["original_gt"]
        gt_label.append(gt_label_val if not _is_empty(gt_label_val) else "None")
        
        # Record if the current prompt has an enhanced reward
        enhanced_reward_flag.append(1.0 if data_item.non_tensor_batch["reward_model"]["enhanced_reward"] == 1.0 else 0.0)

    dcrl_metrics = _batch_compute_dcrl_metrics(
        dcrl_reward, gt_reward, dcrl_label, majority_label,
        anchor_majority_label, gt_label, enhanced_reward_flag, n=n
    )

    # Add consensus ratio
    if hasattr(batch, 'non_tensor_batch') and 'consensus_ratio_list' in batch.non_tensor_batch:
        consensus_ratio_list = batch.non_tensor_batch["consensus_ratio_list"]
        consensus_ratio = sum(consensus_ratio_list) / len(consensus_ratio_list)
        dcrl_metrics["consensus_ratio"] = consensus_ratio

    if hasattr(batch, 'non_tensor_batch') and 'majority_ratio_list' in batch.non_tensor_batch:
        majority_ratio_list = batch.non_tensor_batch["majority_ratio_list"]
        majority_ratio = sum(majority_ratio_list) / len(majority_ratio_list)
        dcrl_metrics["majority_ratio"] = majority_ratio

    if hasattr(batch, 'non_tensor_batch') and 'anchor_majority_ratio_list' in batch.non_tensor_batch:
        anchor_majority_ratio_list = batch.non_tensor_batch["anchor_majority_ratio_list"]
        anchor_majority_ratio = sum(anchor_majority_ratio_list) / len(anchor_majority_ratio_list)
        dcrl_metrics["anchor_majority_ratio"] = anchor_majority_ratio

    # Calculate enhanced reward ratio (number of prompts with enhanced reward / total number of prompts)
    dcrl_metrics["enhanced_reward_ratio"] = sum(enhanced_reward_flag) / len(enhanced_reward_flag)

    return dcrl_metrics


def _batch_compute_dcrl_metrics(
        dcrl_reward: List[float],
        gt_reward: List[float],
        dcrl_label: List[str],
        majority_label: List[str],
        anchor_majority_label: List[str],
        gt_label: List[str],
        enhanced_reward_flag: List[float],
        n: int,
):
    """
    Compute the DCRL metrics for batch inputs.
    Adjusted enhanced_reward related metrics to ratio statistics.
    """
    assert len(dcrl_reward) == len(gt_reward) == len(dcrl_label) == len(majority_label) == len(gt_label) == len(enhanced_reward_flag)
    assert len(dcrl_reward) % n == 0

    n_prompts = len(dcrl_reward) // n
    dcrl_metrics = []

    for i in range(n_prompts):
        prompt_dcrl_reward = dcrl_reward[i * n:(i + 1) * n]
        prompt_gt_reward = gt_reward[i * n:(i + 1) * n]
        prompt_dcrl_label = dcrl_label[i * n:(i + 1) * n]
        prompt_majority_label = majority_label[i * n:(i + 1) * n]
        prompt_anchor_majority_label = anchor_majority_label[i * n:(i + 1) * n]
        prompt_gt_label = gt_label[i * n:(i + 1) * n]
        # Each prompt corresponds to one enhanced reward flag
        enhanced_reward_batch_ratio = enhanced_reward_flag[i]

        # Ensure label consistency within a prompt 
        assert Counter(prompt_dcrl_label).most_common(1)[0][1] == n
        assert Counter(prompt_majority_label).most_common(1)[0][1] == n
        assert Counter(prompt_anchor_majority_label).most_common(1)[0][1] == n
        assert Counter(prompt_gt_label).most_common(1)[0][1] == n

        prompt_dcrl_label = prompt_dcrl_label[0]
        prompt_majority_label = prompt_majority_label[0]
        prompt_anchor_majority_label = prompt_anchor_majority_label[0]
        prompt_gt_label = prompt_gt_label[0]

        dcrl_metric = _prompt_compute_dcrl_metrics(
            prompt_dcrl_reward, prompt_gt_reward, prompt_dcrl_label,
            prompt_majority_label, prompt_anchor_majority_label, prompt_gt_label,
            enhanced_reward_batch_ratio  # Pass the enhanced reward flag
        )
        dcrl_metrics.append(dcrl_metric)

    # Compute the average metrics
    dcrl_metrics = {k: sum(d[k] for d in dcrl_metrics) / len(dcrl_metrics) for k in dcrl_metrics[0]}

    return dcrl_metrics


def _prompt_compute_dcrl_metrics(
        dcrl_reward: List[float],
        gt_reward: List[float],
        dcrl_label: str,
        majority_label: str,
        anchor_majority_label: str,
        gt_label: str,
        enhanced_reward_batch_ratio: float,  # Whether the current prompt has an enhanced reward (1.0/0.0)
):
    """
    Compute DCRL metrics for a single prompt.
    Adjusted enhanced_reward related metrics to enhanced_reward_ratio (whether the current prompt has an enhanced reward).
    """
    assert len(dcrl_reward) == len(gt_reward)

    # Filter out empty labels to avoid misjudgment of empty string equality by the grade function
    dcrl_label = dcrl_label if not _is_empty(dcrl_label) else "None"
    majority_label = majority_label if not _is_empty(majority_label) else "None"
    anchor_majority_label = anchor_majority_label if not _is_empty(anchor_majority_label) else "None"
    gt_label = gt_label if not _is_empty(gt_label) else "None"

    # Calculate label accuracies
    dcrl_hit_rate = 1.0 if grade(dcrl_label, gt_label) else 0.0
    majority_hit_rate = 1.0 if grade(majority_label, gt_label) else 0.0
    anchor_majority_hit_rate = 1.0 if grade(anchor_majority_label, gt_label) else 0.0

    # Calculate reward accuracy
    rewards_hit_rate = 0
    for estimate_reward, true_reward in zip(dcrl_reward, gt_reward):
        if estimate_reward == true_reward:
            rewards_hit_rate += 1
    rewards_hit_rate = rewards_hit_rate / len(dcrl_reward)

    dcrl_metric = {
        "label_accuracy": dcrl_hit_rate,
        "majority_label_accuracy": majority_hit_rate,
        "anchor_majority_label_accuracy": anchor_majority_hit_rate,
        "reward_accuracy": rewards_hit_rate,
        "voting_reward": sum(dcrl_reward) / len(dcrl_reward),
        "ground_truth_reward": sum(gt_reward) / len(gt_reward),
        f"pass@{len(dcrl_reward)}": 1.0 if sum(gt_reward) >= 1 else 0.0,
        "reward_gap": sum(gt_reward) / len(gt_reward) - sum(dcrl_reward) / len(dcrl_reward),
        "enhanced_reward_ratio": enhanced_reward_batch_ratio, 
    }

    return dcrl_metric
from tqdm.auto import tqdm
import torch
from scripts.compute_reward import compute_reward

def eval_reward_model(reward_model, reward_tokenizer, test_dataset, target_label, device='cpu'):
    """
    Evaluate the performance of a reward model by comparing reward scores for chosen and rejected reviews.

    This function selects reviews from a test dataset based on a target label and evaluates the reward model's
    ability to assign higher scores to chosen reviews compared to rejected ones. The evaluation is performed
    in batches for efficiency.
    Note that reward scores are compared on corresponding chosen and rejected reviews:
        chosen_reviews[0] vs rejected_reviews[0],
        chosen_reviews[1] vs rejected_reviews[1],
        etc.

    Parameters:
    reward_model: The model used to compute the reward scores
    reward_tokenizer: The tokenizer for reward_model
    tes_dataset: test Dataset
    target_label (0 or 1): The label used to select chosen reviews. Reviews with this label are considered chosen,
                  while others are considered rejected.
    device (str, optional): The device on which the computation should be performed. Default is 'cpu'.

    Returns:
    float: The accuracy of the reward model, calculated as the proportion of times the model assigns a higher
           reward score to the chosen review compared to the rejected review.

    Example:
    >>> accuracy = eval_reward_model(my_reward_model, my_reward_tokenizer, test_data, target_label=1)
    >>> print(f"Model accuracy: {accuracy:.2%}")
    """

    chosen_reviews = [review['text'] for review in test_dataset if review['label'] == target_label]
    rejected_reviews = [review['text'] for review in test_dataset if review['label'] != target_label]
    
    assert len(chosen_reviews) == len(rejected_reviews)

    correct_predictions = 0
    total_pairs = len(chosen_reviews)
    batch_size = 32
    
    with torch.no_grad():
        for i in tqdm(range(0, total_pairs, batch_size)):
            chosen = chosen_reviews[i:i+batch_size]
            rejected = rejected_reviews[i:i+batch_size]

            chosen_rewards = compute_reward(reward_model, reward_tokenizer, chosen)
            rejected_rewards = compute_reward(reward_model, reward_tokenizer, rejected)

            correct_predictions += torch.sum(chosen_rewards > rejected_rewards).item()

    accuracy = correct_predictions / total_pairs
    return accuracy

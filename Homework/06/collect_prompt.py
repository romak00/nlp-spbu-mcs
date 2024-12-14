def create_prompt(sample: dict) -> str:
    """
    Generates a prompt for a multiple choice question based on the given sample.

    Args:
        sample (dict): A dictionary containing the question, subject, choices, and answer index.

    Returns:
        str: A formatted string prompt for the multiple choice question.
    """
    
    prompt = (
        f"The following are multiple choice questions (with answers) about {sample['subject']}.\n"
        f"{sample['question']}\n"
        f"A. {sample['choices'][0]}\n"
        f"B. {sample['choices'][1]}\n"
        f"C. {sample['choices'][2]}\n"
        f"D. {sample['choices'][3]}\n"
        f"Answer:"
    )

    return prompt


def create_prompt_with_examples(sample: dict, examples: list, add_full_example: bool = False) -> str:
    """
    Generates a 5-shot prompt for a multiple choice question based on the given sample and examples.

    Args:
        sample (dict): A dictionary containing the question, subject, choices, and answer index.
        examples (list): A list of 5 example dictionaries from the dev set.
        add_full_example (bool): whether to add the full text of an answer option

    Returns:
        str: A formatted string prompt for the multiple choice question with 5 examples.
    """
    prompt = ""
    for i in range(0, 5):
        full = f". {examples[i]['choices'][examples[i]['answer']]}." if add_full_example else ""
        example = (
            f"The following are multiple choice questions (with answers) about {examples[i]['subject']}.\n"
            f"{examples[i]['question']}\n"
            f"A. {examples[i]['choices'][0]}\n"
            f"B. {examples[i]['choices'][1]}\n"
            f"C. {examples[i]['choices'][2]}\n"
            f"D. {examples[i]['choices'][3]}\n"
            f"Answer: {chr(examples[i]['answer'] + 65)}"+full+"\n\n"
        )
        prompt += example
    smpl = (
        f"The following are multiple choice questions (with answers) about {sample['subject']}.\n"
        f"{sample['question']}\n"
        f"A. {sample['choices'][0]}\n"
        f"B. {sample['choices'][1]}\n"
        f"C. {sample['choices'][2]}\n"
        f"D. {sample['choices'][3]}\n"
        f"Answer:"
    )
    prompt += smpl

    return prompt
from torch.utils.data import Dataset

class IMDBPairwiseDataset(Dataset):
    """ 
    A dataset of all possible pairs of chosen and rejected texts for TRL reward training format.

    This dataset is designed to facilitate the training of a reward model by providing pairs of
    texts where one is preferred (chosen) and the other is not (rejected). Each sample in the dataset
    is a dictionary containing tokenized input IDs and attention masks for both the chosen and rejected
    texts.

    Parameters:
    imdb: dataset to pairwise
    tokenizer: The tokenizer used to preprocess the texts
    accepted_label (int): The label that indicates a chosen text. Texts with this label are considered
                          preferred, while others are considered rejected.

    Methods:
    __len__(): Returns the total number of possible pairs of chosen and rejected texts.
    __getitem__(index): Returns a dictionary containing tokenized inputs for a specific pair of chosen
                        and rejected texts.
    """
    
    def __init__(self, imdb, tokenizer, accepted_label):
        super().__init__()
        self.tokenizer = tokenizer
        self.chosen_texts = [text['text'] for text in imdb if text['label'] == accepted_label]
        self.rejected_texts = [text['text'] for text in imdb if text['label'] != accepted_label]

        assert self.chosen_texts, f"no texts with label {accepted_label}"
        print(f"Found {len(self.chosen_texts)} chosen and {len(self.rejected_texts)} rejected texts, {len(self)} pairs")

        self.column_names = [
            'input_ids_chosen', 'attention_mask_chosen',
            'input_ids_rejected', 'attention_mask_rejected'
        ]

    def __len__(self):
        return len(self.chosen_texts) * len(self.rejected_texts)

    def __getitem__(self, index: int):
        chosen_text = self.chosen_texts[index // len(self.rejected_texts)]
        rejected_text = self.rejected_texts[index % len(self.rejected_texts)]
        tokenized_chosen = self.tokenizer(chosen_text, truncation=True, return_tensors="pt")
        tokenized_rejected = self.tokenizer(rejected_text, truncation=True, return_tensors="pt")
        
        return dict(
            input_ids_chosen=tokenized_chosen['input_ids'].squeeze(0),
            attention_mask_chosen=tokenized_chosen['attention_mask'].squeeze(0),
            input_ids_rejected=tokenized_rejected['input_ids'].squeeze(0),
            attention_mask_rejected=tokenized_rejected['attention_mask'].squeeze(0)
        )
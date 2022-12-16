import torch
import numpy as np

class DigiDataset(torch.utils.data.Dataset):
    """ Create a PyTorch dataset for Digikala. """

    def __init__(self, tokenizer, comments, max_len=128):
        self.comments = comments
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, item):
        comment = str(self.comments[item])

        encoding = self.tokenizer.encode_plus(
            comment,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt')
        
        inputs = {
            'comment': comment,
            'comment_id': item,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
        }
        
        return inputs


def create_data_loader(x, tokenizer, max_len, batch_size):
    dataset = DigiDataset(
        comments=x,
        tokenizer=tokenizer,
        max_len=max_len)
    
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
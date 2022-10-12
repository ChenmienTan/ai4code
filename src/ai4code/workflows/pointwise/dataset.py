import logging

from torch.utils.data import Dataset
import torch

logger = logging.getLogger(__name__)


class MarkdownDataset(Dataset):

    def __init__(self, df, max_len, tokenizer, add_markdown_context=False):
        super().__init__()
        self.add_markdown_context = add_markdown_context
        self.df = df.reset_index(drop=True)
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.is_first_call = True

    def __getitem__(self, index):
        row = self.df.iloc[index]

        text_pair = None if not self.add_markdown_context else row.source_all

        inputs = self.tokenizer.encode_plus(
            row.source,
            text_pair=text_pair,
            max_length=self.max_len,
            truncation=True
        )
        ids = torch.LongTensor(inputs['input_ids'])
        mask = torch.LongTensor(inputs['attention_mask'])

        if self.is_first_call:
            self.is_first_call = False
            decoded = self.tokenizer.decode(inputs['input_ids'])
            logger.info('Example data: {}'.format(decoded))

        return {'input_ids': ids, 'attention_mask': mask, 'label': torch.FloatTensor([row.pct_rank])}

    def __len__(self):
        return self.df.shape[0]
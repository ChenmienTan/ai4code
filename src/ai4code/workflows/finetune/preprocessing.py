import datasets

from ai4code.workflows.workflow import Preprocessor


class FineTunePreprocessor(Preprocessor):
    def __init__(self, tokenizer, max_len=128):
        self.max_len = max_len
        self.tokenizer = tokenizer

    def fit_transform(self, data):
        return self.transform(data)

    def transform(self, data):
        data = data.copy()
        data.cell_type = data.cell_type.astype('str')
        #data_rank = data.reset_index().groupby(["id", "cell_type"]).cumcount()
        data = data.reset_index()

        # need to do this before dropping code cells so that the counts will be correct
        if 'label' in data:
            data["pct_rank"] = data["label"] / data.groupby("id")["cell_id"].transform("count")

        data = data[data.cell_type == 'markdown']#.reset_index(drop=True)
        ds = datasets.Dataset.from_pandas(data)

        remove_cols = set(ds.features.keys()) - {'input_ids', 'attention_mask', 'token_type_ids'}
        X_train = ds.map(lambda x: self.process_tokenization(x), remove_columns=remove_cols)

        if 'label' in data:
            y = datasets.Dataset.from_pandas(data[['pct_rank']]).rename_column("pct_rank", "label")
            return X_train, y, None
        else:
            return X_train

    def save(self, save_dir):
        pass

    def load(self, save_dir):
        pass

    def process_tokenization(self, row):
        return self.tokenizer(row['source'], max_length=self.max_len)

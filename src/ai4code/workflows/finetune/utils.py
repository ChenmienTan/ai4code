from ai4code.transformer_utils import model_predict


def ranks_to_cell_ids(y_pred):
    y_pred = (
        y_pred
            .sort_values(['id', 'rank'])  # Sort the cells in each notebook by their rank.
            # The cell_ids are now in the order the model predicted.
            .reset_index('cell_id')  # Convert the cell_id index into a column.
            .groupby('id')['cell_id'].apply(list)  # Group the cell_ids for each notebook into a list.
    )
    return y_pred


class ModelWrapper:
    def __init__(self, model, tokenizer, batch_size, target_device):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.target_device = target_device

    def predict(self, X):
        return model_predict(self.model, self.tokenizer, X, self.batch_size, self.target_device)

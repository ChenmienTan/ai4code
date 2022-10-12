from datasets import Dataset
import pandas as pd
from ai4code.workflows.workflow import Postprocessor


class FineTunePostProcessor(Postprocessor):
    def fit_transform(self, raw_data, prep_data, preds):
        pass

    def transform(self, raw_data, prep_data, preds):
        raw_data = raw_data.copy(deep=True)
        raw_data['rank'] = raw_data.groupby(["id", "cell_type"]).cumcount()
        raw_data["pred"] = raw_data.groupby(["id", "cell_type"])["rank"].rank(pct=True)

        if isinstance(preds, Dataset):
            predictions = preds.to_pandas().values
        else:
            predictions = preds

        raw_data.loc[raw_data["cell_type"] == "markdown", "pred"] = predictions

        raw_data['pred_rank'] = raw_data.sort_values(['id', 'pred']).groupby(["id", "cell_type"]).cumcount()

        y_pred = pd.DataFrame({'rank': raw_data.pred_rank}, index=raw_data.index)
        return y_pred

    def save(self, save_dir):
        pass

    def load(self, save_dir):
        pass

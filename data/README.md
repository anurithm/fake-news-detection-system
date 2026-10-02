# Fake or Real News Dataset

## Source Information
- **Dataset Name:** Fake or Real News
- **Original Source:** George McIntire (Hosted by lutzhamel/fake-news on GitHub)
- **Source URL:** `https://github.com/lutzhamel/fake-news/blob/master/data/fake_or_real_news.csv`

## Dataset Statistics
- **Total Samples:** ~6,335
- **REAL Count:** ~3,171
- **FAKE Count:** ~3,164

## Labels
- **Original Format:** `REAL`, `FAKE`
- **Converted Format:** Inherited directly as `REAL` / `FAKE`.

## Preprocessing
The dataset is loaded and cleaned dynamically by the `src/dataset_loader.py` script. The title and text are concatenated into a single `combined_text` string before being processed by `src/preprocessor.py` (which applies lowercasing, HTML/URL removal, punctuation stripping, tokenization, stopword removal, and lemmatization).

## Train/Test Split
During model training (`python src/train.py`), the data is strictly split using an 80/20 stratified partitioning method, enforcing complete isolation of the validation samples during TF-IDF vectorization.

# data/sms_spam.py
import os
import urllib.request
import zipfile

# Use DATA_DIR from environment or fall back to the same default as config.py
DATA_DIR = os.environ.get("SLM_DATA_DIR", "data_cache")

SMS_SPAM_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
SMS_SPAM_CACHE = os.path.join(DATA_DIR, "sms_spam.tsv")


def download_sms_spam(test_fraction: float = 0.2) -> tuple[list[dict], list[dict]]:
    """Download UCI SMS Spam Collection and return (train, test) splits."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(SMS_SPAM_CACHE):
        zip_path = os.path.join(DATA_DIR, "sms_spam.zip")
        urllib.request.urlretrieve(SMS_SPAM_URL, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            with z.open("SMSSpamCollection") as f:
                content = f.read().decode("utf-8", errors="replace")
        with open(SMS_SPAM_CACHE, "w", encoding="utf-8") as f:
            f.write(content)

    examples = []
    with open(SMS_SPAM_CACHE, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                label, text = parts
                examples.append({"text": text, "label": label.strip()})

    split_idx = int(len(examples) * (1 - test_fraction))
    return examples[:split_idx], examples[split_idx:]

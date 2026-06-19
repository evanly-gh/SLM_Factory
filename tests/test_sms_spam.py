# tests/test_sms_spam.py
from data.loaders.sms_spam import download_sms_spam

def test_download_returns_train_test():
    train, test = download_sms_spam()
    assert len(train) > 1000
    assert len(test) > 100
    assert all("text" in ex and "label" in ex for ex in train)
    assert all(ex["label"] in {"spam", "ham"} for ex in train)

def test_class_distribution():
    train, _ = download_sms_spam()
    spam = [e for e in train if e["label"] == "spam"]
    ham = [e for e in train if e["label"] == "ham"]
    assert len(ham) > len(spam)  # UCI is ~87% ham

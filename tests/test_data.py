from pathlib import Path
import pandas as pd
import pytest

from src.data.load_dataset import load_raw_dataset, EXPECTED_COLUMNS
from src.data.reconstruct_threads import reconstruct_conversations
from src.utils.privacy import anonymize_text

def test_anonymize_text():
    sample = "Please email john.doe@example.com or call 555-123-4567 regarding order 112-984712-441. Check https://amazon.com/track"
    anonymized = anonymize_text(sample)
    assert "[EMAIL]" in anonymized
    assert "john.doe@example.com" not in anonymized
    assert "[PHONE]" in anonymized
    assert "[ORDER_ID]" in anonymized
    assert "[URL]" in anonymized

def test_load_raw_dataset_schema():
    df = load_raw_dataset()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    for col in EXPECTED_COLUMNS:
        assert col in df.columns
    assert "author_type" in df.columns
    assert set(df["author_type"].unique()).issubset({"CUSTOMER", "BRAND"})

def test_reconstruct_conversations():
    df = load_raw_dataset()
    convs = reconstruct_conversations(df, target_brand="AmazonHelp")
    assert len(convs) > 0
    c0 = convs[0]
    assert "conversation_id" in c0
    assert "brand" in c0
    assert "messages" in c0
    assert len(c0["messages"]) > 0
    assert c0["messages"][0]["position"] == 0

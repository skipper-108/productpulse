from src import etl

def test_standardize_columns():
    import pandas as pd
    df = pd.DataFrame({"Order Date": ["2025-01-01"], "Revenue($)": [100]})
    result = etl._standardize(df.copy())
    assert set(result.columns) == {"order_date", "revenue_"}

import pytest

import pandas as pd
import tilupy.download_data


def test_import_shaltop_mus_calibrated():
    df, _ = tilupy.download_data.import_shaltop_mus_calibrated()
    assert isinstance(df, pd.DataFrame)
    assert "mus" in df.columns

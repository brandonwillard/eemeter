import tempfile
from datetime import datetime

import pytest
import pandas as pd
import numpy as np
from numpy.testing import assert_allclose
import pytz

from eemeter.weather import ISDWeatherSource
from eemeter.testing.mocks import MockWeatherClient
from eemeter.modeling.formatters import ModelDataFormatter
from eemeter.structures import EnergyTrace
from eemeter.modeling.models import NormalHMMModel

from amimodels.testing import assert_hpd

@pytest.fixture
def mock_isd_weather_source():
    tmp_dir = tempfile.mkdtemp()
    ws = ISDWeatherSource("722880", tmp_dir)
    ws.client = MockWeatherClient()
    return ws


@pytest.fixture
def daily_trace():
    data = {
        "value": np.tile(1, (365,)),
        "estimated": np.tile(False, (365,)),
    }
    columns = ["value", "estimated"]
    index = pd.date_range('2000-01-01', periods=365, freq='D', tz=pytz.UTC)
    df = pd.DataFrame(data, index=index, columns=columns)
    return EnergyTrace("ELECTRICITY_CONSUMPTION_SUPPLIED", df, unit="KWH")


@pytest.fixture
def input_df(mock_isd_weather_source, daily_trace):
    mdf = ModelDataFormatter("D")
    return mdf.create_input(daily_trace, mock_isd_weather_source)


def test_basic(input_df):
    m = NormalHMMModel(65, 65)
    assert str(m).startswith("NormalHMMModel(")
    assert m.cooling_base_temp == 65
    assert m.heating_base_temp == 65
    assert m.model_freq.name == 'D'
    assert m.n is None
    assert m.params is None
    assert m.r2 is None
    assert m.rmse is None
    assert m.y is None

    output = m.fit(input_df)

    assert "r2" in output
    assert "rmse" in output
    assert "cvrmse" in output
    assert "model_params" in output
    assert "upper" in output
    assert "lower" in output
    assert "n" in output

    assert 'init_params' in m.params
    assert 'stoch_traces' in m.params
    assert 'X_design_infos' in m.params
    assert m.r2 == 0.0
    assert m.rmse <= 0.002
    assert m.y.shape == (365, 1)

    predict = m.predict(input_df)

    assert predict.shape == (365,)
    assert_allclose(predict[datetime(2000, 1, 1, tzinfo=pytz.UTC)], 1., atol=0.1)

    assert 'init_params' in m.params
    assert 'stoch_traces' in m.params
    assert 'X_design_infos' in m.params
    assert m.r2 == 0.0
    assert m.rmse <= 0.002
    assert m.y.shape == (365, 1)

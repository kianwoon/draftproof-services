from datetime import datetime, timezone

import pytest

from app.db_cleanup import _retention_days, _utc_datetime


def test_retention_days_rejects_non_positive_values():
    with pytest.raises(ValueError, match="retention"):
        _retention_days(0)


def test_utc_datetime_treats_naive_value_as_utc():
    value = datetime(2026, 6, 3, 12, 0, 0)

    assert _utc_datetime(value).tzinfo == timezone.utc


def test_utc_datetime_converts_aware_value_to_utc():
    sg = timezone.utc
    value = datetime(2026, 6, 3, 12, 0, 0, tzinfo=sg)

    assert _utc_datetime(value).tzinfo == timezone.utc

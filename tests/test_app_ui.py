import pytest

from demandpulse.app import ui


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (2_500_000_000, "2.50B"),
        (66_700_000, "66.7M"),
        (412_000, "412K"),
        (950, "950"),
        (-1_500_000, "-1.5M"),
        (0, "0"),
        (None, "n/a"),
    ],
)
def test_money_is_compact_and_survives_missing_values(value, expected):
    assert ui.money(value) == expected


def test_pct_formats_signed_unsigned_and_missing():
    assert ui.pct(0.1234) == "12.3%"
    assert ui.pct(0.1234, signed=True) == "+12.3%"
    assert ui.pct(-0.05, signed=True) == "-5.0%"
    assert ui.pct(None) == "n/a"


def test_number_has_a_thousands_separator_and_survives_missing_values():
    assert ui.number(12.5) == "12.50"
    assert ui.number(12345.678, 1) == "12,345.7"
    assert ui.number(None) == "n/a"

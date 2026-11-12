__all__ = ["Mt5Trade", "TimeConversion"]


def __getattr__(name):
    if name in __all__:
        from .mt5_trade import Mt5Trade, TimeConversion

        values = {
            "Mt5Trade": Mt5Trade,
            "TimeConversion": TimeConversion,
        }
        return values[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


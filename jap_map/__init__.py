def classFactory(iface):
    """Load the Historical Map Tools plugin."""
    from .plugin import HistoricalMapTools

    return HistoricalMapTools(iface)

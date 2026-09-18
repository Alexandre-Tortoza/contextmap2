"""Concrete source adapter implementations.

Every module here may import a heavy, source-specific SDK (ROS 1, ROS 2,
...) that the rest of the ``ingestion`` capability, and every other
capability, must never depend on directly — see
``src/contextmap/ingestion/docs/adapters.md``. Concrete adapters are
reached by their full dotted path (e.g.
``contextmap.ingestion.adapters.ros1_bag.Ros1BagSourceAdapter``), never
re-exported from ``contextmap.ingestion``, matching how other capabilities'
``backends/`` packages are consumed only by the composition root.
"""

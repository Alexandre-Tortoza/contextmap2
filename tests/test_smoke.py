from contextmap import __version__


def test_package_exposes_version() -> None:
    assert isinstance(__version__, str)
    assert __version__

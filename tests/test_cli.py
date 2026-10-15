from opentac import __version__
from opentac.cli import main


def test_no_command_prints_help(capsys):
    assert main([]) == 0
    assert "opentac" in capsys.readouterr().out


def test_version_matches_package():
    assert __version__.startswith("0.")

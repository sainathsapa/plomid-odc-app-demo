"""Startup banner tests — logo bytes, copper color, stderr-only output."""

from __future__ import annotations

from odc import banner as b


def test_logo_lines_match_brand_source():
    assert len(b.LOGO_LINES) == 6
    assert b.LOGO_LINES[0].startswith("    **      **********")
    assert b.LOGO_LINES[-1].strip().endswith("**********")
    assert all(line.strip() for line in b.LOGO_LINES)


def test_plain_render_has_no_ansi_and_names_version():
    from odc import __version__

    text = b.render_banner(color=False)
    assert "\033" not in text
    assert "PLOMID ODC" in text and __version__ in text


def test_copper_render_uses_brand_accent():
    from odc import __version__

    text = b.render_banner(color=True)
    assert "\033[38;2;226;104;60m" in text  # --accent #e2683c
    assert "\033[0m" in text
    assert text.endswith(f"PLOMID ODC — occupant console v{__version__}\n")


def test_no_color_env_disables_color(monkeypatch):
    class _Tty:
        def isatty(self):
            return True

    assert b.supports_color(_Tty()) is True
    monkeypatch.setenv("NO_COLOR", "1")
    assert b.supports_color(_Tty()) is False

    class _Pipe:
        def isatty(self):
            return False

    monkeypatch.delenv("NO_COLOR")
    assert b.supports_color(_Pipe()) is False


def test_print_banner_goes_to_stderr_only(capsys):
    b.print_banner()
    captured = capsys.readouterr()
    assert "PLOMID ODC" in captured.err
    assert captured.out == ""


def test_start_script_is_valid_sh():
    import subprocess

    r = subprocess.run(["sh", "-n", "start.sh"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr

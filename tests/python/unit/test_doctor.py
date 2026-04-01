from __future__ import annotations

import pytest

from jarvis.config import JarvisConfig
from jarvis.doctor import DoctorReport, DoctorStatus, _check, run_doctor


class TestDoctorReport:
    def test_has_blockers_reflects_blocking_checks(self):
        report = DoctorReport(
            checks=[
                _check("Python:mlx_lm", DoctorStatus.OK, "ok"),
                _check("SpeechAnalyzerCLI", DoctorStatus.BLOCKING, "missing"),
            ]
        )

        assert report.has_blockers

    def test_render_lines_includes_summary(self):
        report = DoctorReport(checks=[_check("Sistema", DoctorStatus.OK, "macOS")])

        lines = report.render_lines()

        assert lines[0] == "[ok] Sistema: macOS"
        assert lines[-1].startswith("\nResumo:")


@pytest.mark.asyncio
async def test_run_doctor_reports_ready_environment_without_blockers(monkeypatch):
    monkeypatch.setattr("jarvis.doctor.platform.system", lambda: "Darwin")
    monkeypatch.setattr("jarvis.doctor.platform.machine", lambda: "arm64")
    monkeypatch.setattr(
        "jarvis.doctor.platform.mac_ver", lambda: ("26.0", ("", "", ""), "")
    )
    monkeypatch.setattr(
        "jarvis.doctor.shutil.which", lambda name: f"/opt/homebrew/bin/{name}"
    )
    monkeypatch.setattr("jarvis.doctor._python_module_check", lambda module_name: None)

    async def probe_speech(binary_path, locale):
        del binary_path, locale
        return _check("SpeechAnalyzerCLI", DoctorStatus.OK, "ok")

    async def probe_foundation(binary_path):
        del binary_path
        return _check("Foundation Models", DoctorStatus.WARNING, "optional")

    monkeypatch.setattr("jarvis.doctor._probe_speech_analyzer", probe_speech)
    monkeypatch.setattr("jarvis.doctor._probe_foundation_models", probe_foundation)

    report = await run_doctor(
        JarvisConfig(activation_backend="push_to_talk_terminal"),
        print_output=False,
    )

    assert not report.has_blockers
    assert any(check.name == "Foundation Models" for check in report.checks)


@pytest.mark.asyncio
async def test_run_doctor_blocks_non_macos_environments(monkeypatch):
    monkeypatch.setattr("jarvis.doctor.platform.system", lambda: "Linux")
    monkeypatch.setattr("jarvis.doctor.platform.machine", lambda: "x86_64")
    monkeypatch.setattr(
        "jarvis.doctor.platform.mac_ver", lambda: ("", ("", "", ""), "")
    )
    monkeypatch.setattr("jarvis.doctor.shutil.which", lambda name: None)
    monkeypatch.setattr(
        "jarvis.doctor._python_module_check",
        lambda module_name: "ModuleNotFoundError: missing",
    )

    async def probe_speech(binary_path, locale):
        del binary_path, locale
        return _check("SpeechAnalyzerCLI", DoctorStatus.BLOCKING, "missing")

    async def probe_foundation(binary_path):
        del binary_path
        return _check("Foundation Models", DoctorStatus.WARNING, "optional")

    monkeypatch.setattr("jarvis.doctor._probe_speech_analyzer", probe_speech)
    monkeypatch.setattr("jarvis.doctor._probe_foundation_models", probe_foundation)

    report = await run_doctor(print_output=False)

    assert report.has_blockers
    assert any(
        check.name == "Sistema" and check.status == DoctorStatus.BLOCKING
        for check in report.checks
    )

from __future__ import annotations

import asyncio
import importlib
import json
import platform
import shutil
import socket
import urllib.error
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .config import JarvisConfig


class DoctorStatus(str, Enum):
    OK = "ok"
    WARNING = "warning"
    BLOCKING = "blocking"


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    status: DoctorStatus
    message: str
    detail: str | None = None


@dataclass
class DoctorReport:
    checks: list[DoctorCheck]

    @property
    def has_blockers(self) -> bool:
        return any(check.status == DoctorStatus.BLOCKING for check in self.checks)

    def render_lines(self) -> list[str]:
        lines = []
        for check in self.checks:
            line = f"[{check.status.value}] {check.name}: {check.message}"
            if check.detail:
                line = f"{line} ({check.detail})"
            lines.append(line)
        if self.has_blockers:
            lines.append(
                "\nResumo: existem bloqueios para o runtime real do J.A.R.V.I.S."
            )
        else:
            lines.append("\nResumo: ambiente pronto para rodar o J.A.R.V.I.S.")
        return lines


def _check(
    name: str, status: DoctorStatus, message: str, detail: str | None = None
) -> DoctorCheck:
    return DoctorCheck(name=name, status=status, message=message, detail=detail)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _stderr_summary(lines: Iterable[str]) -> str:
    values = [line.strip() for line in lines if line.strip()]
    if not values:
        return "sem stderr"
    return " ".join(values[-5:])


def _python_module_check(module_name: str) -> str | None:
    try:
        importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - defensive wrapper
        return f"{exc.__class__.__name__}: {exc}"
    return None


async def _probe_speech_analyzer(binary_path: Path, locale: str) -> DoctorCheck:
    if not binary_path.exists():
        return _check(
            "SpeechAnalyzerCLI",
            DoctorStatus.BLOCKING,
            "binario STT nao encontrado",
            str(binary_path),
        )

    process = await asyncio.create_subprocess_exec(
        str(binary_path),
        "--mock-transcript",
        "doctor",
        "--locale",
        locale,
        "--format",
        "ndjson",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        return _check(
            "SpeechAnalyzerCLI",
            DoctorStatus.BLOCKING,
            "probe do bridge STT falhou",
            _stderr_summary(stderr.decode("utf-8", errors="replace").splitlines()),
        )

    event_types: list[str] = []
    for line in stdout.decode("utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        event_type = payload.get("type")
        if isinstance(event_type, str):
            event_types.append(event_type)

    if "final_transcript" not in event_types:
        return _check(
            "SpeechAnalyzerCLI",
            DoctorStatus.BLOCKING,
            "probe do bridge STT nao retornou transcript final",
        )

    return _check(
        "SpeechAnalyzerCLI",
        DoctorStatus.OK,
        "bridge STT pronto para sessoes de voz",
    )


async def _probe_foundation_models(binary_path: Path) -> DoctorCheck:
    if not binary_path.exists():
        return _check(
            "Foundation Models",
            DoctorStatus.WARNING,
            "binario do hot path nao encontrado; o runtime vai degradar para o fallback local",
            str(binary_path),
        )

    port = _find_free_port()
    process = await asyncio.create_subprocess_exec(
        str(binary_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    stderr_lines: list[str] = []

    async def drain_stderr() -> None:
        if process.stderr is None:
            return
        async for raw_line in process.stderr:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if line:
                stderr_lines.append(line)

    stderr_task = asyncio.create_task(drain_stderr())
    try:
        payload = None
        for _ in range(40):
            if process.returncode is not None:
                break
            try:
                request = urllib.request.Request(
                    f"http://127.0.0.1:{port}/health",
                    method="GET",
                )
                response = await asyncio.to_thread(
                    urllib.request.urlopen, request, timeout=2.0
                )
                try:
                    body = await asyncio.to_thread(response.read)
                finally:
                    await asyncio.to_thread(response.close)
                payload = json.loads(body.decode("utf-8"))
                break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
                await asyncio.sleep(0.25)

        if payload is None:
            return _check(
                "Foundation Models",
                DoctorStatus.WARNING,
                "nao foi possivel iniciar o bridge do hot path; o runtime vai usar o fallback local",
                _stderr_summary(stderr_lines),
            )

        if payload.get("status") == "ok":
            return _check(
                "Foundation Models",
                DoctorStatus.OK,
                "hot path nativo disponivel",
                str(payload.get("availability", "available")),
            )

        return _check(
            "Foundation Models",
            DoctorStatus.WARNING,
            "hot path nativo indisponivel; o runtime vai usar o fallback local",
            str(payload.get("detail") or payload.get("availability") or "unknown"),
        )
    finally:
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
        stderr_task.cancel()
        try:
            await stderr_task
        except asyncio.CancelledError:
            pass


async def run_doctor(
    config: JarvisConfig | None = None, *, print_output: bool = True
) -> DoctorReport:
    config = config or JarvisConfig()
    checks: list[DoctorCheck] = []

    system_name = platform.system()
    checks.append(
        _check(
            "Sistema",
            DoctorStatus.OK if system_name == "Darwin" else DoctorStatus.BLOCKING,
            "macOS detectado"
            if system_name == "Darwin"
            else "somente macOS e suportado",
            system_name,
        )
    )

    machine = platform.machine()
    checks.append(
        _check(
            "Arquitetura",
            DoctorStatus.OK if machine == "arm64" else DoctorStatus.BLOCKING,
            "Apple Silicon detectado"
            if machine == "arm64"
            else "Apple Silicon e obrigatorio",
            machine,
        )
    )

    mac_version = platform.mac_ver()[0] or "desconhecida"
    checks.append(_check("macOS", DoctorStatus.OK, "versao detectada", mac_version))

    for binary_name, blocking in (
        ("ffmpeg", True),
        ("espeak-ng", True),
        ("swiftlint", False),
    ):
        path = shutil.which(binary_name)
        checks.append(
            _check(
                binary_name,
                DoctorStatus.OK
                if path
                else (DoctorStatus.BLOCKING if blocking else DoctorStatus.WARNING),
                "binario encontrado" if path else "binario nao encontrado",
                path,
            )
        )

    runtime_modules = {
        "mlx_lm": DoctorStatus.BLOCKING,
        "mlx_audio": DoctorStatus.BLOCKING,
        "numpy": DoctorStatus.BLOCKING,
        "sounddevice": DoctorStatus.BLOCKING,
        "pynput": DoctorStatus.WARNING,
        "sqlite_vec": DoctorStatus.BLOCKING,
        "sentence_transformers": DoctorStatus.BLOCKING,
    }
    for module_name, failure_status in runtime_modules.items():
        error = _python_module_check(module_name)
        checks.append(
            _check(
                f"Python:{module_name}",
                DoctorStatus.OK if error is None else failure_status,
                "modulo importado" if error is None else "modulo ausente ou quebrado",
                error,
            )
        )

    eventkit_error = _python_module_check("EventKit")
    checks.append(
        _check(
            "Python:EventKit",
            DoctorStatus.OK if eventkit_error is None else DoctorStatus.WARNING,
            "bindings nativas do calendario disponiveis"
            if eventkit_error is None
            else "bindings do calendario indisponiveis; tools de calendario ficam degradadas",
            eventkit_error,
        )
    )

    env_path = Path(".env")
    checks.append(
        _check(
            ".env",
            DoctorStatus.OK if env_path.exists() else DoctorStatus.WARNING,
            "arquivo .env presente"
            if env_path.exists()
            else "arquivo .env ausente; defaults do projeto serao usados",
            str(env_path),
        )
    )

    checks.append(
        await _probe_speech_analyzer(Path(config.stt_bridge_bin), config.stt_locale)
    )
    checks.append(await _probe_foundation_models(Path(config.llm_hot_path_bridge_bin)))

    if config.activation_backend != "push_to_talk_terminal":
        checks.append(
            _check(
                "Permissoes macOS",
                DoctorStatus.WARNING,
                "o modo voz pode solicitar permissoes de Microfone e Acessibilidade na primeira execucao",
            )
        )

    report = DoctorReport(checks=checks)
    if print_output:
        for line in report.render_lines():
            print(line)
    return report

"""The shared skeleton of experiment drivers: run layout, retained outputs and child measurement."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from contextmap.evaluation.experiment_driver import (
    ChildProcessMeasurement,
    DriverRun,
    ExperimentDriverError,
    _measure_as_only_child,
    measure_child_process,
)
from contextmap.runtime import ArtifactRef, StageRequest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
# Um driver de experimento novo vive sob `experiments/<experimento>/`; o arquivo não precisa
# existir para a resolução da raiz, que depende só de onde o driver está.
DRIVER = REPOSITORY_ROOT / "experiments" / "arm-retention-probe" / "scripts" / "run_all.py"

MIB = 1024
"""KiB per MiB: ``ru_maxrss`` is reported in KiB on Linux."""

# Filho trivial: aloca N MiB tocando cada byte (entra no RSS), dorme e imprime N.
ALLOCATE = (
    "import sys, time\n"
    "data = b'x' * (int(sys.argv[1]) << 20)\n"
    "time.sleep(float(sys.argv[2]))\n"
    "print(len(data) >> 20)\n"
)


def _allocating_child(mebibytes: int, *, sleep_seconds: float = 0.0) -> list[str]:
    return [sys.executable, "-c", ALLOCATE, str(mebibytes), str(sleep_seconds)]


def _state_estimation_request(run: DriverRun, name: str) -> StageRequest:
    return StageRequest(
        stage_id="state_estimation",
        inputs={
            "sequence": (
                ArtifactRef(
                    stage_id="ingestion",
                    contract="SequenceArtifact",
                    artifact_id="bag",
                    content_hash="sha256:bag-content",
                    location="ingest/sequences/bag",
                ),
            )
        },
        components={},
        config_digest=run.config_digest(name),
        output_dir=run.output_dir(name),
        workspace=run.outputs_root,
    )


class TestRunLayout:
    """Red 1: one run number derives every per-run value, under the root it is given."""

    def test_a_run_number_derives_the_output_dir_location_and_configuration_identity(
        self, tmp_path: Path
    ) -> None:
        run = DriverRun(experiment="e2e-real", run_number=3, outputs_root=tmp_path)

        assert run.run_id == "run-0003"
        assert run.output_dir("state_estimation") == (
            tmp_path / "e2e-real" / "run-0003" / "state_estimation"
        )
        assert run.location("state_estimation") == "e2e-real/run-0003/state_estimation"
        assert run.config_digest("state_estimation") == "e2e-real/run-0003/state_estimation"

    def test_the_layout_is_the_one_the_e2e_drivers_wrote_by_hand(self) -> None:
        # Caracterização: scripts-run3/ fixava à mão `OUTPUT_DIR = WORKSPACE /
        # "e2e-real/run-0003/<estágio>"` e a mesma string como `location` do `ArtifactRef` do
        # estágio anterior. O esqueleto deriva as duas do número do run.
        workspace = REPOSITORY_ROOT / "outputs"
        run = DriverRun(experiment="e2e-real", run_number=3, outputs_root=workspace)

        assert run.output_dir("sensor_association") == (
            workspace / "e2e-real/run-0003/sensor_association"
        )
        assert run.location("state_estimation") == "e2e-real/run-0003/state_estimation"

    def test_everything_follows_the_root_and_no_fixed_path(self, tmp_path: Path) -> None:
        first = DriverRun(experiment="probe", run_number=1, outputs_root=tmp_path / "a")
        second = DriverRun(experiment="probe", run_number=1, outputs_root=tmp_path / "b")

        layout = Path("probe/run-0001/arm-C")
        assert first.output_dir("arm-C") == tmp_path / "a" / layout
        assert second.output_dir("arm-C") == tmp_path / "b" / layout
        # O que o report registra é relativo à raiz, então não muda com ela.
        assert first.location("arm-C") == second.location("arm-C")
        assert first.config_digest("arm-C") == second.config_digest("arm-C")

    def test_a_rerun_gets_another_directory_and_another_configuration_identity(
        self, tmp_path: Path
    ) -> None:
        first = DriverRun(experiment="probe", run_number=1, outputs_root=tmp_path)
        rerun = DriverRun(experiment="probe", run_number=2, outputs_root=tmp_path)

        assert first.output_dir("state_estimation") != rerun.output_dir("state_estimation")
        assert (
            _state_estimation_request(first, "state_estimation").identity()
            != _state_estimation_request(rerun, "state_estimation").identity()
        )

    def test_arms_of_one_run_over_the_same_inputs_get_distinct_identities(
        self, tmp_path: Path
    ) -> None:
        # Dois braços do mesmo estágio sobre as mesmas entradas só diferem pela configuração;
        # um rótulo por run daria a eles a mesma identidade de execução.
        run = DriverRun(experiment="probe", run_number=1, outputs_root=tmp_path)

        assert (
            _state_estimation_request(run, "arm-C").identity()
            != _state_estimation_request(run, "arm-D").identity()
        )

    def test_the_runtime_resolves_the_same_directory_and_run_number(self, tmp_path: Path) -> None:
        run = DriverRun(experiment="probe", run_number=12, outputs_root=tmp_path)
        request = _state_estimation_request(run, "state_estimation")
        upstream = ArtifactRef(
            stage_id="state_estimation",
            contract="StateEstimationRunArtifact",
            artifact_id="trajectory",
            location=run.location("state_estimation"),
        )

        assert request.run_number() == 12
        assert request.directory_of(upstream) == run.output_dir("state_estimation")

    @pytest.mark.parametrize("run_number", [0, -1, True])
    def test_a_run_number_must_be_a_positive_integer(self, tmp_path: Path, run_number: int) -> None:
        with pytest.raises(ExperimentDriverError, match="run number"):
            DriverRun(experiment="probe", run_number=run_number, outputs_root=tmp_path)

    @pytest.mark.parametrize("experiment", ["", ".", "..", "a/b", "a\\b"])
    def test_the_experiment_is_one_directory_name(self, tmp_path: Path, experiment: str) -> None:
        with pytest.raises(ExperimentDriverError, match="experiment"):
            DriverRun(experiment=experiment, run_number=1, outputs_root=tmp_path)

    @pytest.mark.parametrize("name", ["", ".", "..", "arm/C", "../escape"])
    def test_an_artifact_name_is_one_directory_name(self, tmp_path: Path, name: str) -> None:
        run = DriverRun(experiment="probe", run_number=1, outputs_root=tmp_path)

        with pytest.raises(ExperimentDriverError, match="artifact name"):
            run.output_dir(name)
        with pytest.raises(ExperimentDriverError, match="artifact name"):
            run.location(name)

    def test_a_relative_outputs_root_is_refused(self) -> None:
        # Uma raiz relativa dependeria do diretório corrente de quem executa o driver.
        with pytest.raises(ExperimentDriverError, match="absolute"):
            DriverRun(experiment="probe", run_number=1, outputs_root=Path("outputs"))


class TestRetainedOutputsRoot:
    """Red 3: arms are retained under the checkout's ``outputs/`` unless a root is given."""

    def test_the_default_root_is_the_outputs_directory_of_the_checkout_holding_the_driver(
        self,
    ) -> None:
        run = DriverRun.for_driver(DRIVER, experiment="arm-retention-probe", run_number=1)

        assert run.outputs_root == REPOSITORY_ROOT / "outputs"
        assert run.output_dir("arm-C") == (
            REPOSITORY_ROOT / "outputs" / "arm-retention-probe" / "run-0001" / "arm-C"
        )

    def test_the_default_does_not_depend_on_how_deep_the_driver_is(self) -> None:
        nested = REPOSITORY_ROOT / "experiments" / "probe" / "scripts-run2" / "stages" / "d.py"

        run = DriverRun.for_driver(nested, experiment="probe", run_number=1)

        assert run.outputs_root == REPOSITORY_ROOT / "outputs"

    def test_a_temporary_root_is_used_only_when_given_explicitly(self, tmp_path: Path) -> None:
        run = DriverRun.for_driver(
            DRIVER, experiment="arm-retention-probe", run_number=1, outputs_root=tmp_path
        )

        assert run.outputs_root == tmp_path

    def test_a_driver_outside_the_experiments_of_a_checkout_has_no_default_root(
        self, tmp_path: Path
    ) -> None:
        outside = REPOSITORY_ROOT / "tests" / "evaluation" / "driver.py"
        # Um `experiments/` qualquer num diretório temporário não é um checkout: sem
        # `pyproject.toml` ao lado, a raiz não é adivinhada.
        stray = tmp_path / "experiments" / "probe" / "driver.py"

        for driver in (outside, stray):
            with pytest.raises(ExperimentDriverError, match="explicit outputs root"):
                DriverRun.for_driver(driver, experiment="probe", run_number=1)


class TestChildMeasurement:
    """Red 2: one runner reports the wall time and peak RSS of the measured child itself."""

    def test_the_runner_reports_the_wall_time_and_peak_rss_of_the_child(self) -> None:
        measurement = measure_child_process(_allocating_child(128, sleep_seconds=0.3))

        assert measurement.returncode == 0
        assert measurement.stdout.strip() == "128"
        assert measurement.wall_time_seconds >= 0.3
        assert measurement.peak_rss_kib >= 128 * MIB
        # O interpretador do filho soma poucos MiB; nada perto do processo que mede.
        assert measurement.peak_rss_kib < (128 + 64) * MIB

    def test_a_later_smaller_child_is_not_charged_the_peak_of_an_earlier_one(self) -> None:
        # É o motivo do processo novo por medição: RUSAGE_CHILDREN é um máximo corrente sobre
        # todo filho que um processo já esperou. Medidos a partir do mesmo pai, o segundo filho
        # herdaria os 256 MiB do primeiro.
        large = measure_child_process(_allocating_child(256))
        small = measure_child_process(_allocating_child(0))

        assert large.peak_rss_kib >= 256 * MIB
        assert small.peak_rss_kib < 64 * MIB

    def test_a_failing_child_is_measured_not_raised(self) -> None:
        command = [sys.executable, "-c", "import sys; sys.stderr.write('boom'); sys.exit(3)"]

        measurement = measure_child_process(command)

        assert measurement.returncode == 3
        assert measurement.stderr == "boom"
        assert measurement.command == tuple(command)

    def test_the_record_keeps_the_fields_both_historical_runners_report(self) -> None:
        # Caracterização do contrato comum aos dois `_run_one.py`: `returncode`, o wall time
        # arredondado a 3 casas e `peak_rss_mb` = ru_maxrss (KiB) / 1024, com 1 casa.
        measurement = ChildProcessMeasurement(
            command=("python", "stage.py"),
            returncode=0,
            wall_time_seconds=0.26649,
            peak_rss_kib=73_421,
            stdout="",
            stderr="",
        )

        assert measurement.to_record() == {
            "returncode": 0,
            "wall_time_seconds": 0.266,
            "peak_rss_mb": 71.7,
        }

    def test_a_command_that_cannot_start_is_a_measurement_error(self) -> None:
        with pytest.raises(ExperimentDriverError, match="measuring process failed"):
            measure_child_process(["contextmap-no-such-executable"])

    def test_an_empty_command_is_refused(self) -> None:
        with pytest.raises(ExperimentDriverError, match="empty command"):
            measure_child_process([])

    def test_peak_rss_is_only_read_where_its_unit_is_known(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # ru_maxrss é KiB no Linux e bytes no macOS: ler em outra plataforma inflaria o pico
        # 1024 vezes em silêncio.
        monkeypatch.setattr(sys, "platform", "darwin")

        with pytest.raises(ExperimentDriverError, match="Linux"):
            measure_child_process(_allocating_child(0))

    def test_the_measuring_process_refuses_to_measure_after_reaping_another_child(self) -> None:
        # Este processo já esperou um filho: o pico dele contaminaria a medição seguinte.
        subprocess.run([sys.executable, "-c", "pass"], check=True)

        with pytest.raises(ExperimentDriverError, match="already waited for a child"):
            _measure_as_only_child([sys.executable, "-c", "pass"])

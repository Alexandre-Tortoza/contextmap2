"""Tests for the public ingestion application service, with a scripted fake source adapter."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest
from runtime_documents import effective_from, selected_document
from runtime_ingestion import FakeAdapter, factory, image, sequence
from runtime_ingestion import request as make_request

from contextmap.ingestion import (
    MissingRequiredTopicError,
    SequenceArtifactError,
    SequenceArtifactReader,
    SourceAdapterCapabilities,
    SourceTopicMapping,
    SourceWindow,
    SynchronizationConfig,
    compute_source_content_hash,
)
from contextmap.runtime import (
    ArtifactRef,
    CancellationToken,
    ExecutionEvent,
    FileArtifactStore,
    IngestionRequest,
    IngestionService,
    IngestionStageExecutor,
    ReusePolicy,
    RunJournal,
    StageExecutionError,
    ValidationPolicy,
    read_run,
    resolve_plan,
    run_plan,
)


class Sink:
    def __init__(self) -> None:
        self.events: list[ExecutionEvent] = []

    def emit(self, event: ExecutionEvent) -> None:
        self.events.append(event)

    @property
    def kinds(self) -> list[str]:
        return [event.kind for event in self.events]


def _service(**options: Any) -> IngestionService:
    return IngestionService(factory(**options))


def _published(workspace: Path, name: str = "corridor-02") -> list[Path]:
    root = workspace / "sequences" / name
    return sorted(root.iterdir()) if root.exists() else []


class TestPreflight:
    def test_a_valid_request_passes_and_reports_what_the_source_provides(
        self, tmp_path: Path
    ) -> None:
        report = _service().preflight(make_request(tmp_path))

        assert report.ok
        assert report.capabilities["rgb"] is True and report.capabilities["lidar"] is False
        assert report.identity == make_request(tmp_path).identity
        assert any("no calibration" in warning for warning in report.warnings)

    def test_it_reads_no_observation_and_publishes_nothing(self, tmp_path: Path) -> None:
        built: list[FakeAdapter] = []

        _service(built=built).preflight(make_request(tmp_path))

        assert built and built[0].read_calls == 0
        assert not (tmp_path / "ws").exists()

    def test_a_required_topic_without_a_configured_topic_is_a_configuration_problem(
        self, tmp_path: Path
    ) -> None:
        report = _service().preflight(make_request(tmp_path, required_topics=frozenset({"lidar"})))

        assert not report.ok
        assert any(
            p.path == "request.required_topics" and "lidar" in p.message for p in report.problems
        )

    def test_a_required_topic_missing_from_the_source_is_reported(self, tmp_path: Path) -> None:
        topics = SourceTopicMapping(rgb="/camera", lidar="/lidar")

        report = _service().preflight(
            make_request(tmp_path, topics=topics, required_topics=frozenset({"lidar"}))
        )

        assert any(
            p.path == "source.required_topics" and "lidar" in p.message for p in report.problems
        )

    def test_an_unavailable_optional_dependency_is_explained_with_its_install_hint(
        self, tmp_path: Path
    ) -> None:
        def missing(_config: Any) -> Any:
            raise ModuleNotFoundError("No module named 'rosbags'", name="rosbags")

        report = IngestionService(missing).preflight(make_request(tmp_path))

        problem = next(p for p in report.problems if p.path == "adapter.dependency")
        assert "rosbags" in problem.message and "contextmap[ros1]" in problem.message

    def test_an_empty_clock_identity_is_refused(self, tmp_path: Path) -> None:
        report = _service().preflight(make_request(tmp_path, timestamp_clock_id="  "))

        assert any(p.path == "request.timestamp_clock_id" for p in report.problems)

    def test_a_reference_modality_without_a_topic_is_refused(self, tmp_path: Path) -> None:
        report = _service().preflight(make_request(tmp_path, reference="lidar"))

        assert any(p.path == "request.synchronization" for p in report.problems)

    def test_a_reference_modality_absent_from_the_source_is_refused(self, tmp_path: Path) -> None:
        provides = SourceAdapterCapabilities(imu=True)  # o bag não tem imagens

        report = _service(provides=provides).preflight(make_request(tmp_path))

        assert any(p.path == "source.synchronization" for p in report.problems)

    def test_a_missing_or_unwritable_path_is_reported_before_anything_else(
        self, tmp_path: Path
    ) -> None:
        missing = make_request(tmp_path, source_path=str(tmp_path / "nowhere.bag"))
        not_a_directory = tmp_path / "file"
        not_a_directory.write_text("x", encoding="utf-8")
        blocked = make_request(tmp_path, output_dir=str(not_a_directory))

        assert any(p.path == "source.path" for p in _service().preflight(missing).problems)
        assert any(p.path == "output.output_dir" for p in _service().preflight(blocked).problems)

    def test_an_adapter_that_cannot_open_the_source_is_a_problem_naming_the_error(
        self, tmp_path: Path
    ) -> None:
        class Broken(FakeAdapter):
            def capabilities(self) -> SourceAdapterCapabilities:
                raise OSError("truncated bag header")

        service = IngestionService(
            lambda config: Broken(config, [], provides=SourceAdapterCapabilities())
        )

        report = service.preflight(make_request(tmp_path))

        assert any(
            p.path == "source.read" and "OSError" in p.message and "truncated" in p.message
            for p in report.problems
        )

    def test_it_never_falls_back_to_another_adapter_family(self, tmp_path: Path) -> None:
        report = IngestionService(factory(family="ros1_bag")).preflight(
            make_request(tmp_path, source_type="ros2_bag")
        )

        assert any(
            p.path == "adapter.selection" and "ros2_bag" in p.message and "ros1_bag" in p.message
            for p in report.problems
        )

    def test_an_unsound_request_is_reported_once_and_never_reaches_the_adapter(
        self, tmp_path: Path
    ) -> None:
        built: list[FakeAdapter] = []

        report = _service(built=built).preflight(
            make_request(tmp_path, required_topics=frozenset({"thermal"}))
        )

        assert [p.path for p in report.problems] == ["request.required_topics"]
        assert built == []

    def test_problems_are_reported_together(self, tmp_path: Path) -> None:
        report = _service().preflight(
            make_request(
                tmp_path,
                source_path=str(tmp_path / "nowhere.bag"),
                required_topics=frozenset({"lidar"}),
                reference="lidar",
            )
        )

        assert len(report.problems) >= 3


class TestRequestIdentity:
    def test_it_is_deterministic_and_independent_of_the_output_directory(
        self, tmp_path: Path
    ) -> None:
        first = make_request(tmp_path)

        assert first.identity == make_request(tmp_path).identity
        assert (
            first.identity
            == make_request(tmp_path, output_dir=str(tmp_path / "elsewhere")).identity
        )

    @pytest.mark.parametrize(
        "change",
        [
            {"sequence_name": "other"},
            {"required_topics": frozenset({"rgb"})},
            {"timestamp_clock_id": "another:clock"},
            {"tolerance_ns": 5},
            {"hash_source": False},
            {"config_identity": "sha256:abc"},
            {"validation": ValidationPolicy(on_problems="warn")},
            {"topics": SourceTopicMapping(rgb="/other", imu="/imu")},
        ],
    )
    def test_every_input_that_changes_the_result_changes_the_identity(
        self, tmp_path: Path, change: dict[str, Any]
    ) -> None:
        assert make_request(tmp_path, **change).identity != make_request(tmp_path).identity

    def test_the_document_carries_no_secret_and_round_trips(self, tmp_path: Path) -> None:
        original = make_request(tmp_path, required_topics=frozenset({"rgb"}))
        document = original.to_document()

        rebuilt = IngestionRequest.from_document(
            document, output_dir=original.output_dir, source_type=None
        )

        assert rebuilt.identity == original.identity
        assert json.loads(json.dumps(document)) == document
        assert "output_dir" not in document

    def test_a_configured_window_changes_the_identity(self, tmp_path: Path) -> None:
        base = make_request(tmp_path)
        window = SourceWindow(
            clock_id=f"{base.source_type}:{base.source_path}:recording_time",
            start_seconds=0.0,
            end_seconds=1.0,
        )

        windowed = make_request(tmp_path, window=window)

        assert windowed.identity != base.identity
        assert base.window is None
        assert windowed.window == window

    def test_the_window_round_trips_through_the_document(self, tmp_path: Path) -> None:
        window = SourceWindow(
            clock_id="ros1_bag:recording.bag:recording_time", start_seconds=1.0, end_seconds=2.0
        )
        original = make_request(tmp_path, window=window)

        document = original.to_document()
        rebuilt = IngestionRequest.from_document(
            document, output_dir=original.output_dir, source_type=None
        )

        assert document["window"] == {
            "clock_id": window.clock_id,
            "start_seconds": 1.0,
            "end_seconds": 2.0,
        }
        assert rebuilt.window == window
        assert rebuilt.identity == original.identity
        assert json.loads(json.dumps(document)) == document

    def test_an_invalid_document_is_refused(self, tmp_path: Path) -> None:
        good = make_request(tmp_path).to_document()

        with pytest.raises(ValueError, match="reference_modality"):
            IngestionRequest.from_document(
                {
                    **good,
                    "synchronization": {"reference_modality": "nope", "tolerance_nanoseconds": 1},
                },
                output_dir="ws",
            )
        with pytest.raises(ValueError, match="invalid ingestion request"):
            IngestionRequest.from_document({"source_path": "x"}, output_dir="ws")

    @pytest.mark.parametrize("name", ["", "a/b", "..", "a\\b"])
    def test_a_sequence_name_must_be_a_single_safe_segment(self, tmp_path: Path, name: str) -> None:
        with pytest.raises(ValueError, match="sequence_name"):
            make_request(tmp_path, name=name)


class TestSuccessfulRun:
    def test_it_publishes_a_reopenable_immutable_sequence_artifact(self, tmp_path: Path) -> None:
        result = _service().run(make_request(tmp_path))

        assert result.status == "completed" and result.failure is None
        assert result.artifact_path is not None and result.artifact_id is not None
        reader = SequenceArtifactReader(Path(result.artifact_path))
        assert reader.verify_integrity() == []
        assert str(reader.manifest.artifact_id) == result.artifact_id
        assert dict(reader.manifest.observation_counts)["image"] == 3
        assert [p.name for p in _published(tmp_path / "ws")] == ["artifact-1"]
        assert SequenceArtifactReader(_published(tmp_path / "ws")[0]).manifest.artifact_id == (
            result.artifact_id
        )

    def test_the_result_carries_counts_diagnostics_and_operational_metrics(
        self, tmp_path: Path
    ) -> None:
        result = _service().run(make_request(tmp_path))

        assert result.observation_counts == {"external_pose": 0, "image": 3, "imu": 3, "lidar": 0}
        assert result.diagnostics["processing_observations"] == 3
        assert result.diagnostics["synchronization_policy"] == "nearest_within_tolerance"
        metrics = result.metrics
        assert metrics.observations_read == 6 and metrics.files_written >= 2
        assert metrics.bytes_written > 0 and metrics.elapsed_s >= 0
        assert set(metrics.phase_seconds) >= {"reading-source", "validating", "synchronizing"}
        assert json.loads(json.dumps(result.to_document()))["status"] == "completed"

    def test_the_provenance_binds_the_effective_request_and_the_source(
        self, tmp_path: Path
    ) -> None:
        request = make_request(tmp_path, config_identity="sha256:runtime-config")
        result = _service().run(request)
        assert result.artifact_path is not None

        provenance = SequenceArtifactReader(Path(result.artifact_path)).read_provenance()

        assert provenance is not None
        assert provenance.source_type == "ros1_bag" and provenance.adapter_type == "ros1_bag"
        assert provenance.source_path == request.source_path
        assert provenance.source_content_hash is not None
        assert provenance.configuration_hash is not None
        assert provenance.synchronization_policy == "nearest_within_tolerance"
        assert provenance.ingestion_config["request_identity"] == request.identity
        assert provenance.ingestion_config["config_identity"] == "sha256:runtime-config"

    def test_skipping_the_source_hash_is_recorded_not_hidden(self, tmp_path: Path) -> None:
        result = _service().run(make_request(tmp_path, hash_source=False))
        assert result.artifact_path is not None

        provenance = SequenceArtifactReader(Path(result.artifact_path)).read_provenance()

        assert provenance is not None and provenance.source_content_hash is None
        assert provenance.ingestion_config["hash_source"] is False

    def test_a_configured_window_reaches_the_adapter_and_the_persisted_provenance(
        self, tmp_path: Path
    ) -> None:
        """#506: a janela deve ser alcançável pelo caminho canônico, não só pelos adapters.

        Exercita o caminho real ``IngestionRequest`` -> ``IngestionService.run()``
        (não um script adapter->writer->provenance montado à mão): o pedido
        carrega ``window``, a config construída para o adapter carrega
        ``window`` (`_adapter_config()`), e a proveniência publicada declara
        a janela realmente usada, com um hash de conteúdo derivado do que o
        adapter efetivamente leu (`adapter.content_hash()`), nunca de uma
        passada separada sobre a fonte inteira.
        """
        built: list[FakeAdapter] = []
        base_request = make_request(tmp_path)
        window = SourceWindow(
            clock_id=f"{base_request.source_type}:{base_request.source_path}:recording_time",
            start_seconds=0.0,
            end_seconds=10.0,
        )
        request = dataclasses.replace(base_request, window=window)

        result = _service(built=built).run(request)

        assert result.status == "completed" and result.failure is None
        # preflight() constructs its own adapter to inspect capabilities(), and _execute()
        # builds a second, separate one for the actual read -- both must carry the window.
        assert built and all(adapter.config.window == window for adapter in built)
        read_adapter = next(adapter for adapter in built if adapter.read_calls > 0)
        assert result.artifact_path is not None
        provenance = SequenceArtifactReader(Path(result.artifact_path)).read_provenance()
        assert provenance is not None
        assert provenance.ingestion_config["window"] == {
            "clock_id": window.clock_id,
            "start_seconds": 0.0,
            "end_seconds": 10.0,
        }
        assert provenance.source_content_hash == read_adapter.content_hash()
        assert provenance.source_content_hash is not None
        # Uma janela configurada nunca deve custar uma passada de hash sobre a
        # fonte inteira: o hash persistido é o do adapter (só o que foi lido),
        # não igual ao hash do arquivo inteiro no disco.
        assert provenance.source_content_hash != compute_source_content_hash(
            Path(request.source_path)
        )

    def test_adapter_warnings_reach_the_result_and_the_artifact(self, tmp_path: Path) -> None:
        result = _service(warnings=["skipped a malformed message"]).run(make_request(tmp_path))
        assert result.artifact_path is not None

        provenance = SequenceArtifactReader(Path(result.artifact_path)).read_provenance()

        assert result.status == "completed"
        assert "skipped a malformed message" in result.warnings
        assert provenance is not None and "skipped a malformed message" in provenance.warnings
        assert result.metrics.warnings == 1

    def test_events_follow_the_lifecycle_and_report_progress(self, tmp_path: Path) -> None:
        sink = Sink()

        _service().run(make_request(tmp_path), event_sink=sink, progress_interval=2)

        assert sink.kinds == [
            "ingestion.planned",
            "ingestion.reading-source",
            "ingestion.progress",
            "ingestion.progress",
            "ingestion.progress",
            "ingestion.validating",
            "ingestion.synchronizing",
            "ingestion.writing-artifact",
            "ingestion.completed",
        ]
        assert [e.sequence for e in sink.events] == list(range(1, len(sink.events) + 1))
        assert sink.events[-1].data["artifact_id"]

    def test_the_same_request_run_twice_publishes_two_distinct_immutable_artifacts(
        self, tmp_path: Path
    ) -> None:
        request = make_request(tmp_path)
        first = _service().run(request)
        second = _service().run(
            dataclasses.replace(
                request,
                output_dir=str(tmp_path / "ws" / "sequences" / "corridor-02" / "artifact-2"),
            )
        )

        assert first.artifact_id != second.artifact_id
        assert first.request_identity == second.request_identity
        assert len(_published(tmp_path / "ws")) == 2

    def test_a_large_payload_is_streamed_and_only_metadata_is_kept(self, tmp_path: Path) -> None:
        observations = [image(i, float(i), data=bytes(6)) for i in range(1, 50)]
        built: list[FakeAdapter] = []

        result = IngestionService(factory(observations=observations, built=built)).run(
            make_request(tmp_path, topics=SourceTopicMapping(rgb="/camera"))
        )

        assert result.status == "completed" and result.metrics.observations_read == 49


class TestExpectedFailures:
    def _assert_nothing_published(self, tmp_path: Path) -> None:
        root = tmp_path / "ws" / "sequences" / "corridor-02"
        assert not root.exists() or list(root.iterdir()) == []

    def test_a_preflight_problem_fails_the_run_with_a_category_and_publishes_nothing(
        self, tmp_path: Path
    ) -> None:
        result = _service().run(make_request(tmp_path, source_path=str(tmp_path / "nowhere.bag")))

        assert result.status == "failed" and result.failure is not None
        assert result.failure.category == "source" and result.failure.phase == "planned"
        assert result.artifact_id is None
        self._assert_nothing_published(tmp_path)

    def test_a_missing_optional_dependency_fails_explicitly(self, tmp_path: Path) -> None:
        def missing(_config: Any) -> Any:
            raise ModuleNotFoundError("rosbags", name="rosbags")

        result = IngestionService(missing).run(make_request(tmp_path))

        assert result.failure is not None and result.failure.category == "dependency"
        assert "rosbags" in result.failure.message

    def test_an_execution_failure_is_a_source_failure_with_nothing_published(
        self, tmp_path: Path
    ) -> None:
        result = _service(fail_after=2).run(make_request(tmp_path))

        assert result.status == "failed" and result.failure is not None
        assert result.failure.category == "source"
        assert result.failure.phase == "reading-source"
        assert "bag corrupt" in result.failure.message
        assert result.failure.exception_type == "RuntimeError"
        self._assert_nothing_published(tmp_path)

    def test_a_required_topic_absent_at_read_time_is_a_source_failure(self, tmp_path: Path) -> None:
        result = _service(
            fail_after=0, error=MissingRequiredTopicError("topic /lidar not in bag")
        ).run(make_request(tmp_path))

        assert result.failure is not None and result.failure.category == "source"
        assert "/lidar" in result.failure.message

    def test_an_import_failure_while_decoding_is_a_dependency_failure(self, tmp_path: Path) -> None:
        result = _service(fail_after=1, error=ImportError("no rosbags.typesys")).run(
            make_request(tmp_path)
        )

        assert result.failure is not None and result.failure.category == "dependency"

    def test_a_structurally_invalid_observation_fails_validation_and_publishes_nothing(
        self, tmp_path: Path
    ) -> None:
        observations = [image(1, 1.0, data=b"short"), *sequence()[1:]]

        result = IngestionService(factory(observations=observations)).run(make_request(tmp_path))

        assert result.failure is not None
        assert result.failure.category == "validation" and result.failure.phase == "validating"
        assert "frame-0001" in result.failure.message
        self._assert_nothing_published(tmp_path)

    def test_the_warn_policy_publishes_and_records_each_problem(self, tmp_path: Path) -> None:
        observations = [image(1, 1.0, data=b"short"), *sequence()[1:]]

        result = IngestionService(factory(observations=observations)).run(
            make_request(tmp_path, validation=ValidationPolicy(on_problems="warn"))
        )

        assert result.status == "completed"
        assert any("frame-0001" in warning for warning in result.warnings)
        assert result.diagnostics["validation_problems"]

    def test_a_sequence_without_its_reference_modality_cannot_be_synchronized(
        self, tmp_path: Path
    ) -> None:
        only_imu = [obs for obs in sequence() if type(obs).__name__ == "ImuObservation"]

        result = IngestionService(factory(observations=only_imu)).run(make_request(tmp_path))

        assert result.failure is not None and result.failure.category == "validation"
        assert "nothing to synchronize" in result.failure.message

    def test_out_of_order_timestamps_are_a_validation_failure(self, tmp_path: Path) -> None:
        observations = [image(1, 2.0), image(2, 1.0)]

        result = IngestionService(factory(observations=observations)).run(
            make_request(tmp_path, topics=SourceTopicMapping(rgb="/camera"))
        )

        assert result.failure is not None and "non-monotonic" in result.failure.message

    def test_a_writer_failure_leaves_no_temporary_or_final_artifact(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def broken(self: Any) -> Any:
            raise SequenceArtifactError("disk full while publishing")

        monkeypatch.setattr("contextmap.ingestion.SequenceArtifactWriter.finalize", broken)

        result = _service().run(make_request(tmp_path))

        assert result.failure is not None and result.failure.category == "output"
        assert "disk full" in result.failure.message
        self._assert_nothing_published(tmp_path)
        leftovers = [p for p in (tmp_path / "ws").rglob(".tmp-*")]
        assert leftovers == []

    def test_a_failure_message_never_carries_a_secret(self, tmp_path: Path) -> None:
        secret = "s3cr3t-token-value"
        sink = Sink()

        result = _service(fail_after=1, error=RuntimeError(f"401 for {secret}")).run(
            make_request(tmp_path),
            event_sink=sink,
            redact=lambda text: text.replace(secret, "***"),
        )

        assert result.failure is not None and secret not in result.failure.message
        assert secret not in json.dumps([e.to_document() for e in sink.events])
        assert sink.kinds[-1] == "ingestion.failed"


class TestCancellation:
    def test_cancellation_between_observations_stops_and_publishes_nothing(
        self, tmp_path: Path
    ) -> None:
        token = CancellationToken()
        sink = Sink()

        result = _service(
            on_observation=lambda index: token.cancel("stop") if index == 2 else None
        ).run(make_request(tmp_path), event_sink=sink, cancellation=token)

        assert result.status == "cancelled" and result.failure is not None
        assert result.failure.category == "cancelled" and result.failure.message == "stop"
        assert result.artifact_id is None
        assert sink.kinds[-1] == "ingestion.cancelled"
        root = tmp_path / "ws" / "sequences" / "corridor-02"
        assert not root.exists() or list(root.iterdir()) == []

    def test_a_token_cancelled_before_reading_starts_reads_nothing(self, tmp_path: Path) -> None:
        token = CancellationToken()
        token.cancel()
        built: list[FakeAdapter] = []

        result = IngestionService(factory(built=built)).run(
            make_request(tmp_path), cancellation=token
        )

        assert result.status == "cancelled"
        assert built and built[-1].read_calls == 0
        assert result.metrics.observations_read == 0

    def test_an_interrupt_is_recorded_as_cancelled_and_propagates(self, tmp_path: Path) -> None:
        sink = Sink()

        def interrupt(index: int) -> None:
            if index == 1:
                raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            _service(on_observation=interrupt).run(make_request(tmp_path), event_sink=sink)

        assert sink.kinds[-1] == "ingestion.cancelled"
        root = tmp_path / "ws" / "sequences" / "corridor-02"
        assert not root.exists() or list(root.iterdir()) == []
        assert [p for p in (tmp_path / "ws").rglob(".tmp-*")] == []


class TestUnexpectedFailures:
    def test_a_bug_outside_the_expected_failures_is_emitted_and_re_raised(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sink = Sink()

        def boom(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("programming error")

        monkeypatch.setattr("contextmap.runtime.ingestion_service.synchronize", boom)

        with pytest.raises(RuntimeError, match="programming error"):
            _service().run(make_request(tmp_path), event_sink=sink)

        assert sink.kinds[-1] == "ingestion.failed"
        assert sink.events[-1].data["category"] == "unexpected"
        assert [p for p in (tmp_path / "ws").rglob(".tmp-*")] == []


class TestAsTheIngestionStageOfTheDag:
    def _plan(self, tmp_path: Path) -> Any:
        effective = effective_from(tmp_path, selected_document())
        return effective, resolve_plan(effective).scope(targets=["ingestion"])

    def _run(self, tmp_path: Path, service: IngestionService, **options: Any) -> Any:
        effective, execution = self._plan(tmp_path)
        journal = RunJournal.create(tmp_path / "ws", effective, execution)
        executor = IngestionStageExecutor(service, make_request(tmp_path))
        return journal, run_plan(
            execution,
            {"ingestion": executor},
            environ={},
            module_available=lambda _n: True,
            journal=journal,
            **options,
        )

    def test_the_executor_publishes_into_the_stage_directory_of_the_run(
        self, tmp_path: Path
    ) -> None:
        journal, record = self._run(tmp_path, _service())

        artifact = record.stages[0].output
        assert artifact.contract == "SequenceArtifact" and artifact.content_hash
        assert artifact.location == f"S1/{journal.directory.name}/ingestion"
        directory = tmp_path / "ws" / str(artifact.location)
        assert SequenceArtifactReader(directory).manifest.artifact_id == artifact.artifact_id
        # Nada é publicado fora da pasta do estágio: o executor só usa o diretório que recebeu.
        assert not (tmp_path / "ws" / "sequences").exists()

    def test_the_artifact_identity_is_derived_from_the_stage_and_is_repeatable(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        _, first = self._run(tmp_path / "a", _service())
        _, second = self._run(tmp_path / "b", _service())

        assert first.stages[0].output.artifact_id == second.stages[0].output.artifact_id

    def test_the_same_ingestion_is_reused_by_identity_and_a_changed_one_is_recomputed(
        self, tmp_path: Path
    ) -> None:
        effective, execution = self._plan(tmp_path)

        def verify(ref: ArtifactRef) -> bool:
            return ref.location is not None and (tmp_path / "ws" / ref.location).is_dir()

        policy = ReusePolicy(
            store=FileArtifactStore(tmp_path / "index", verify=verify), code_identity="c1"
        )
        service = _service()

        def run() -> Any:
            journal = RunJournal.create(tmp_path / "ws", effective, execution)
            executor = IngestionStageExecutor(service, make_request(tmp_path))
            return run_plan(
                execution,
                {"ingestion": executor},
                environ={},
                module_available=lambda _n: True,
                reuse=policy,
                journal=journal,
            )

        first, second = run(), run()

        assert first.stages[0].output == second.stages[0].output
        assert second.stages[0].decision is not None and second.stages[0].decision.kind == "reused"
        assert len(list((tmp_path / "ws" / "S1").glob("run-*/ingestion"))) == 1

    def test_an_ingestion_failure_becomes_a_categorized_failure_record(
        self, tmp_path: Path
    ) -> None:
        effective, execution = self._plan(tmp_path)
        journal = RunJournal.create(tmp_path / "records", effective, execution)
        executor = IngestionStageExecutor(_service(fail_after=1), make_request(tmp_path))

        with pytest.raises(StageExecutionError, match="bag corrupt"):
            run_plan(
                execution,
                {"ingestion": executor},
                environ={},
                module_available=lambda _n: True,
                journal=journal,
            )

        failure = read_run(journal.directory).failure
        assert failure is not None
        assert failure.stage_id == "ingestion" and failure.category == "source"


def test_the_synchronization_config_of_a_request_is_the_ingestion_capabilitys_own_type(
    tmp_path: Path,
) -> None:
    request = make_request(tmp_path)

    assert isinstance(request.synchronization, SynchronizationConfig)
    assert isinstance(request.topics, SourceTopicMapping)

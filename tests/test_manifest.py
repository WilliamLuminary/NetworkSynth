# SPDX-License-Identifier: GPL-3.0-or-later
import json
import os

import pytest

from networksynth.handlers.manifest import (
    MANIFEST_NAME,
    MANIFEST_VERSION,
    STATUS_CANCELLED,
    STATUS_FAILED,
    STATUS_OK,
    collect_outputs,
    read_manifest,
    write_manifest,
)
from networksynth.handlers.run_paths import RunPaths

pytestmark = pytest.mark.unit


def _touch(root, *relative):
    for rel in relative:
        path = os.path.join(root, *rel.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            fh.write("x")


class TestClassification:
    def test_buckets_by_location_and_name(self, tmp_path):
        root = str(tmp_path)
        _touch(
            root,
            "sample_1/synthetic/net_edgelist.csv",
            "sample_1/synthetic/net_positions.csv",
            "sample_1/synthetic/graph.webp",
            "sample_1/original/orig.png",
            "sample_1/snapshots/step_00001.png",
            "sample_1/original/net.graphml.gz",
            "sample_1/analysis_data.json",
            "sample_1/spectra_analysis_figure.webp",
        )

        buckets = collect_outputs(root)

        assert buckets["edge_lists"] == ["sample_1/synthetic/net_edgelist.csv"]
        assert buckets["positions"] == ["sample_1/synthetic/net_positions.csv"]
        assert "sample_1/snapshots/step_00001.png" in buckets["snapshots"]
        assert "sample_1/synthetic/graph.webp" in buckets["previews"]
        assert "sample_1/original/net.graphml.gz" in buckets["networks"]
        assert "sample_1/analysis_data.json" in buckets["analysis"]
        assert "sample_1/spectra_analysis_figure.webp" in buckets["analysis"]
        assert "sample_1/analysis_data.json" not in buckets.get("networks", [])

    def test_paths_use_forward_slashes(self, tmp_path):
        _touch(str(tmp_path), "a/b/net_edgelist.csv")

        buckets = collect_outputs(str(tmp_path))

        assert buckets["edge_lists"] == ["a/b/net_edgelist.csv"]
        assert "\\" not in buckets["edge_lists"][0]

    def test_paths_are_relative_to_the_run_root(self, tmp_path):
        _touch(str(tmp_path), "sample_1/synthetic/net_edgelist.csv")

        path = collect_outputs(str(tmp_path))["edge_lists"][0]

        assert not os.path.isabs(path)

    def test_manifest_excludes_itself(self, tmp_path):
        _touch(str(tmp_path), "sample_1/synthetic/net_edgelist.csv")
        write_manifest(RunPaths(root=str(tmp_path)), status=STATUS_OK)

        buckets = collect_outputs(str(tmp_path))

        assert all(MANIFEST_NAME not in p for paths in buckets.values() for p in paths)

    def test_empty_run(self, tmp_path):
        assert collect_outputs(str(tmp_path)) == {}


class TestWriteManifest:
    def test_records_status_and_version(self, tmp_path):
        write_manifest(RunPaths(root=str(tmp_path)), status=STATUS_OK)

        data = read_manifest(str(tmp_path))

        assert data["status"] == STATUS_OK
        assert data["manifest_version"] == MANIFEST_VERSION
        assert os.path.isabs(data["run_root"])

    def test_records_failure_with_the_error(self, tmp_path):
        write_manifest(
            RunPaths(root=str(tmp_path)),
            status=STATUS_FAILED,
            error="RuntimeError: nope",
        )

        data = read_manifest(str(tmp_path))

        assert data["status"] == STATUS_FAILED
        assert data["error"] == "RuntimeError: nope"

    def test_cancelled_is_distinct_from_failed(self, tmp_path):
        write_manifest(RunPaths(root=str(tmp_path)), status=STATUS_CANCELLED)

        assert read_manifest(str(tmp_path))["status"] == STATUS_CANCELLED
        assert STATUS_CANCELLED != STATUS_FAILED

    def test_omits_error_key_on_success(self, tmp_path):
        write_manifest(RunPaths(root=str(tmp_path)), status=STATUS_OK)

        assert "error" not in read_manifest(str(tmp_path))

    def test_carries_optional_metrics(self, tmp_path):
        write_manifest(
            RunPaths(root=str(tmp_path)), status=STATUS_OK, metrics={"avg_error": 0.081}
        )

        assert read_manifest(str(tmp_path))["metrics"]["avg_error"] == 0.081

    def test_counts_files(self, tmp_path):
        _touch(str(tmp_path), "a/net_edgelist.csv", "a/net_positions.csv", "a/x.webp")

        write_manifest(RunPaths(root=str(tmp_path)), status=STATUS_OK)

        assert read_manifest(str(tmp_path))["file_count"] == 3

    def test_is_valid_json_on_one_read(self, tmp_path):
        path = write_manifest(RunPaths(root=str(tmp_path)), status=STATUS_OK)

        with open(path) as fh:
            json.load(fh)


@pytest.mark.integration
@pytest.mark.requires_fixture_data
class TestManifestFromARealRun:
    def test_generate_writes_a_manifest_listing_its_outputs(self, tmp_path):
        from dataclasses import replace

        from networksynth.pipelines.generate import main
        from tests.fixture_config import FIXTURE

        config = replace(
            FIXTURE,
            BASE_OUTPUT_PATH=str(tmp_path),
            OUTPUT_DENOTE="ManifestCfg",
            ERROR_CHECKER="none",
            MAX_ATTEMPTS=2,
            SEED=99,
            SYNTHETIC_NETWORK_NUMBER=1,
            SYNTHETIC_GRAPH_NUMBER=1,
            SNAPSHOT_INTERVAL=0,
            DATASETS=FIXTURE.DATASETS[:1],
        )

        main(config=config)

        roots = [
            os.path.join(str(tmp_path), d)
            for d in os.listdir(str(tmp_path))
            if d.startswith("ManifestCfg_results_")
        ]
        assert len(roots) == 1, f"expected one run dir, got {roots}"

        data = read_manifest(roots[0])
        assert data["status"] == STATUS_OK
        assert data["file_count"] > 0
        assert data["outputs"], "manifest listed no outputs"
        assert data["outputs"].get("edge_lists"), data["outputs"]

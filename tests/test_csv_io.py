import pytest

from graphs.csv_io import read_edge_list_csv, read_graph_csv, read_positions_csv

pytestmark = pytest.mark.unit


def _write(path, rows):
    path.write_text("\n".join(",".join(str(c) for c in r) for r in rows) + "\n")
    return str(path)


SGT_EDGES_UNWEIGHTED = [["Source", "Target"], [0, 1], [1, 2], [2, 0]]
SGT_EDGES_WEIGHTED = [
    ["Source", "Target", "Weight", "Length", "Width", "Angle"],
    [0, 1, 2.5, 10.0, 1.0, 45.0],
    [1, 2, 3.5, 12.0, 1.2, 90.0],
]
POSITIONS = [["x", "y"], [0.0, 0.0], [10.0, 0.0], [5.0, 8.0]]


class TestStructuralGTFormat:
    def test_unweighted_edge_list(self, tmp_path):
        graph = read_graph_csv(
            _write(tmp_path / "e.csv", SGT_EDGES_UNWEIGHTED),
            _write(tmp_path / "p.csv", POSITIONS),
        )

        assert graph.number_of_nodes() == 3
        assert graph.number_of_edges() == 3
        assert graph.has_edge(0, 1)

    def test_weighted_edge_list_keeps_weights(self, tmp_path):
        graph = read_graph_csv(
            _write(tmp_path / "e.csv", SGT_EDGES_WEIGHTED),
            _write(tmp_path / "p.csv", POSITIONS),
        )

        assert graph.is_weighted()
        assert graph.weight(0, 1) == pytest.approx(2.5)
        assert graph.weight(1, 2) == pytest.approx(3.5)

    def test_extra_columns_are_ignored(self, tmp_path):
        graph = read_graph_csv(
            _write(tmp_path / "e.csv", SGT_EDGES_WEIGHTED),
            _write(tmp_path / "p.csv", POSITIONS),
        )

        assert graph.number_of_edges() == 2

    def test_positions_are_read_in_row_order(self, tmp_path):
        positions = read_positions_csv(_write(tmp_path / "p.csv", POSITIONS))

        assert positions.shape == (3, 2)
        assert tuple(positions[2]) == (5.0, 8.0)

    def test_headers_are_case_insensitive(self, tmp_path):
        graph = read_graph_csv(
            _write(tmp_path / "e.csv", [["SOURCE", "target"], [0, 1]]),
            _write(tmp_path / "p.csv", [["X", "Y"], [0, 0], [1, 1]]),
        )

        assert graph.number_of_edges() == 1


class TestOurOwnFormat:
    def test_round_trips_save_network_csv(
        self, tmp_path, load_unweighted_test_synth_graph
    ):
        from configs.file_definitions import save_network_csv

        original = load_unweighted_test_synth_graph
        save_network_csv(original, str(tmp_path / "net.csv"))

        loaded = read_graph_csv(
            str(tmp_path / "net_edgelist.csv"), str(tmp_path / "net_positions.csv")
        )

        assert loaded.number_of_nodes() == original.number_of_nodes()
        assert loaded.number_of_edges() == original.number_of_edges()
        assert sorted(loaded.edges()) == sorted(original.edges())

    def test_round_trip_preserves_unweightedness(
        self, tmp_path, load_unweighted_test_synth_graph
    ):
        from configs.file_definitions import save_network_csv

        original = load_unweighted_test_synth_graph
        assert not original.is_weighted()

        save_network_csv(original, str(tmp_path / "net.csv"))
        loaded = read_graph_csv(
            str(tmp_path / "net_edgelist.csv"), str(tmp_path / "net_positions.csv")
        )

        assert not loaded.is_weighted()

    def test_round_trip_preserves_weights(
        self, tmp_path, load_weighted_test_synth_graph
    ):
        from configs.file_definitions import save_network_csv

        original = load_weighted_test_synth_graph
        assert original.is_weighted()

        save_network_csv(original, str(tmp_path / "net.csv"))
        loaded = read_graph_csv(
            str(tmp_path / "net_edgelist.csv"), str(tmp_path / "net_positions.csv")
        )

        assert loaded.is_weighted()
        for u, v in original.edges():
            assert loaded.weight(u, v) == pytest.approx(original.weight(u, v))

    def test_round_trip_preserves_positions(
        self, tmp_path, load_unweighted_test_synth_graph
    ):
        from configs.file_definitions import save_network_csv

        original = load_unweighted_test_synth_graph
        save_network_csv(original, str(tmp_path / "net.csv"))
        loaded = read_graph_csv(
            str(tmp_path / "net_edgelist.csv"), str(tmp_path / "net_positions.csv")
        )

        assert loaded.positions().shape == original.positions().shape
        assert loaded.positions() == pytest.approx(original.positions())


class TestFailsLoudly:

    def test_edge_index_beyond_positions(self, tmp_path):
        with pytest.raises(ValueError, match="only 2 rows"):
            read_graph_csv(
                _write(tmp_path / "e.csv", [["Source", "Target"], [0, 5]]),
                _write(tmp_path / "p.csv", [["x", "y"], [0, 0], [1, 1]]),
            )

    def test_negative_index(self, tmp_path):
        with pytest.raises(ValueError, match="negative node index"):
            read_graph_csv(
                _write(tmp_path / "e.csv", [["Source", "Target"], [0, -1]]),
                _write(tmp_path / "p.csv", [["x", "y"], [0, 0], [1, 1]]),
            )

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_graph_csv(
                str(tmp_path / "nope.csv"), _write(tmp_path / "p.csv", POSITIONS)
            )

    def test_unrecognised_edge_header(self, tmp_path):
        with pytest.raises(ValueError, match="no column matching"):
            read_edge_list_csv(_write(tmp_path / "e.csv", [["a", "b"], [0, 1]]))

    def test_missing_y_column(self, tmp_path):
        with pytest.raises(ValueError, match="no column matching"):
            read_positions_csv(_write(tmp_path / "p.csv", [["x"], [0]]))

    def test_empty_file(self, tmp_path):
        path = tmp_path / "p.csv"
        path.write_text("")
        with pytest.raises(ValueError, match="file is empty"):
            read_positions_csv(str(path))

    def test_header_only_positions(self, tmp_path):
        with pytest.raises(ValueError, match="no position rows"):
            read_positions_csv(_write(tmp_path / "p.csv", [["x", "y"]]))

    def test_malformed_edge_row_names_the_line(self, tmp_path):
        with pytest.raises(ValueError, match="line 3"):
            read_edge_list_csv(
                _write(tmp_path / "e.csv", [["Source", "Target"], [0, 1], [2, "abc"]])
            )


class TestEdgeCases:
    def test_no_edges_is_allowed(self, tmp_path):
        graph = read_graph_csv(
            _write(tmp_path / "e.csv", [["Source", "Target"]]),
            _write(tmp_path / "p.csv", POSITIONS),
        )

        assert graph.number_of_nodes() == 3
        assert graph.number_of_edges() == 0

    def test_float_indices_are_accepted(self, tmp_path):
        graph = read_graph_csv(
            _write(tmp_path / "e.csv", [["Source", "Target"], ["0.0", "1.0"]]),
            _write(tmp_path / "p.csv", POSITIONS),
        )

        assert graph.has_edge(0, 1)

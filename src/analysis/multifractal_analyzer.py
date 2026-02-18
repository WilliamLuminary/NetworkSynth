# src/analysis/multifractal_analyzer.py
import logging
from collections import Counter
from contextlib import contextmanager
from dataclasses import astuple, dataclass
from typing import Dict, List

import networkit as nk
import numpy as np
from scipy.optimize import linprog

from config import BaseConfig
from graph.synth_graph import SynthGraph

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Native Ollivier-Ricci curvature (NetworKit + scipy, no networkx needed)
# ---------------------------------------------------------------------------


def _wasserstein_lp(p: np.ndarray, q: np.ndarray, cost: np.ndarray) -> float:
    """Exact 1-Wasserstein distance between discrete measures via LP (HiGHS).

    Parameters
    ----------
    p, q : 1-D arrays summing to 1
        Source / target probability masses.
    cost : 2-D array of shape (len(p), len(q))
        Pairwise transport costs.

    Returns
    -------
    float
        The earth-mover distance, or NaN on solver failure.
    """
    n, m = len(p), len(q)
    c = cost.ravel()

    # Row-sum constraints: sum_j f[i,j] = p[i]
    # Col-sum constraints: sum_i f[i,j] = q[j]
    A_eq = np.zeros((n + m, n * m))
    for i in range(n):
        A_eq[i, i * m : (i + 1) * m] = 1.0
    for j in range(m):
        A_eq[n + j, j::m] = 1.0
    b_eq = np.concatenate([p, q])

    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    return res.fun if res.success else float("nan")


def _ollivier_ricci_curvature(
    graph: SynthGraph,
    alpha: float = 0.5,
    weighted: bool = False,
) -> List[float]:
    """Compute Ollivier-Ricci curvature for every edge.

    For each edge (u, v) the curvature is

        κ(u, v) = 1 − W₁(μ_u, μ_v) / d(u, v)

    where μ_x places mass *alpha* on x and spreads (1 − alpha) uniformly
    over x's neighbours, and W₁ is the 1-Wasserstein (earth-mover) distance
    under the shortest-path metric d.

    Parameters
    ----------
    graph : SynthGraph
    alpha : float, default 0.5
        Laziness parameter for the random walk measure.
    weighted : bool
        If True, use inverted edge weights (1/w) as distances.

    Returns
    -------
    list[float]
        One curvature value per edge, in ``graph.edges()`` order.
    """
    nk_graph = graph.nk

    # Build the distance graph (inverted weights when weighted)
    if weighted:
        n = nk_graph.numberOfNodes()
        dist_graph = nk.Graph(n, weighted=True)
        for u, v, w in nk_graph.iterEdgesWeights():
            dist_graph.addEdge(u, v, 1.0 / w if w != 0 else float("inf"))
    else:
        dist_graph = nk_graph

    # Pre-compute all-pairs shortest paths (very fast in NetworKit)
    apsp = nk.distance.APSP(dist_graph)
    apsp.run()

    curvatures: List[float] = []
    for u, v in graph.edges():
        # Probability measure at u: alpha·δ_u + (1−alpha)·Uniform(N(u))
        nbrs_u = list(nk_graph.iterNeighbors(u))
        support_u = [u] + nbrs_u
        mu_u = np.empty(len(support_u))
        if nbrs_u:
            mu_u[0] = alpha
            mu_u[1:] = (1.0 - alpha) / len(nbrs_u)
        else:
            mu_u[0] = 1.0

        # Probability measure at v
        nbrs_v = list(nk_graph.iterNeighbors(v))
        support_v = [v] + nbrs_v
        mu_v = np.empty(len(support_v))
        if nbrs_v:
            mu_v[0] = alpha
            mu_v[1:] = (1.0 - alpha) / len(nbrs_v)
        else:
            mu_v[0] = 1.0

        # Cost matrix: shortest-path distances between the two supports
        cost = np.empty((len(support_u), len(support_v)))
        for i, s in enumerate(support_u):
            for j, t in enumerate(support_v):
                cost[i, j] = apsp.getDistance(s, t)

        w1 = _wasserstein_lp(mu_u, mu_v, cost)

        d_uv = apsp.getDistance(u, v)
        kappa = 1.0 - w1 / d_uv if d_uv > 0 else 0.0
        curvatures.append(kappa)

    return curvatures


@dataclass
class MultifractalErrorFeatures:
    holder_exponent: float
    spectrum_width: float


def _generate_range(scale):
    return [q / 100 for q in range(-scale, scale + 1, 10)]


class MultifractalAnalyzer:
    small_q = _generate_range(300)
    full_q = _generate_range(2000)

    def __init__(self, graph: SynthGraph):
        self.graph = graph
        self.f_digit = 0
        self.q_ = None
        self.weighted = BaseConfig.MEASURE_WEIGHTED if graph.is_weighted() else False
        if BaseConfig.MEASURE_WEIGHTED is not self.weighted:
            print("Unweighted graph! Can't perform weighted analysis.")

    # ---- helpers ----

    def _get_nk_graph(self) -> nk.Graph:
        """Return the underlying nk.Graph for analysis."""
        return self.graph.nk

    def _make_inverted_weight_graph(self) -> nk.Graph:
        """Create a nk.Graph copy with inverted weights (1/w) for distance-based algorithms."""
        g = self.graph.nk
        n = g.numberOfNodes()
        inv = nk.Graph(n, weighted=True)
        for u, v, w in g.iterEdgesWeights():
            inv_w = 1.0 / w if w != 0 else float("inf")
            inv.addEdge(u, v, inv_w)
        return inv

    # ---- public ----

    def analyze_error_features(self) -> MultifractalErrorFeatures:
        with self.set_q(self.small_q):
            tau_list, _ = self._compute_multifractal_taus()
            alpha_0, width, _, _ = self._compute_n_spectrum(tau_list)
        return MultifractalErrorFeatures(alpha_0, width)

    @staticmethod
    def analyze_error(this, other) -> float:
        assert this and other, "Invalid input: One error feature is None."
        from scipy.spatial.distance import euclidean

        return euclidean(astuple(this), astuple(other))

    # ---- multifractal core ----

    def _compute_multifractal_taus(self):
        nk_graph = self._get_nk_graph()

        n_list = []
        r_g_all_set = set()
        for node in self.graph.nodes():
            distances = (
                nk.distance.Dijkstra(nk_graph, node, storePaths=False)
                .run()
                .getDistances()
            )
            grow = [d for d in distances if 0 < d < 99999]
            grow.sort()
            if self.f_digit == 0:
                import math

                grow = [math.ceil(d) for d in grow]
            else:
                grow = [
                    round(d, self.f_digit) for d in grow if round(d, self.f_digit) != 0
                ]
            num = Counter(grow)
            r_g_all_set.update(num.keys())
            n_list.append(num)

        r_g_all = np.array(sorted(list(r_g_all_set)))
        if len(r_g_all) < 2:
            return [0.0 for _ in self.q_]

        diameter = r_g_all[-1]
        ntw_mat = np.ones((len(n_list), len(r_g_all)))
        for i, num_counter in enumerate(n_list):
            for j, r_val in enumerate(r_g_all):
                ntw_mat[i, j] += sum(
                    count for dist_, count in num_counter.items() if dist_ <= r_val
                )

        zq_list = []
        for q in self.q_:
            row_sums = []
            for i in range(len(ntw_mat)):
                if ntw_mat[i, -1] == 0:
                    row_sums.append(0)
                else:
                    row_sums.append((ntw_mat[i, :] / ntw_mat[i, -1]) ** q)
            zq = np.sum(row_sums, axis=0)
            zq_list.append(zq)

        from scipy.stats import linregress

        tau_list = []
        for idx, q in enumerate(self.q_):
            x = np.log(r_g_all / diameter)
            y = np.log(zq_list[idx])
            slope, _, _, _, _ = linregress(x, y)
            tau_list.append(slope)
        return tau_list, zq_list

    def _compute_n_spectrum(self, tau_list):
        q_ = self.q_
        al_list, fal_list = [], []
        for i in range(1, len(q_)):
            al = (tau_list[i] - tau_list[i - 1]) / (q_[i] - q_[i - 1])
            al_list.append(al)
        for j in range(len(q_) - 1):
            fal = q_[j] * al_list[j] - tau_list[j]
            fal_list.append(fal)
        alpha_0 = al_list[np.argmax(fal_list)]
        width = np.max(al_list) - np.min(al_list)
        return alpha_0, width, al_list, fal_list

    def _compute_n_dimension(self, tau_list):
        q_list = self.q_
        valid_pairs = [(q, tau) for q, tau in zip(q_list, tau_list) if q != 0]
        valid_q, valid_tau = zip(*valid_pairs)
        dim_list = [tau / q for q, tau in zip(valid_q, valid_tau)]

        dim_max = np.max(dim_list)
        dim_min = np.min(dim_list)
        diff = dim_max - dim_min
        return dim_list, dim_max, dim_min, diff, valid_q

    def _compute_node_dimension(self) -> Dict[int, float]:
        if self.weighted:
            nk_graph = self._make_inverted_weight_graph()
        else:
            nk_graph = self._get_nk_graph()

        node_dimensions = {}
        for node in self.graph.nodes():
            distances = (
                nk.distance.Dijkstra(nk_graph, int(node), storePaths=False)
                .run()
                .getDistances()
            )
            distances.sort()
            if self.weighted:
                distances = [
                    round(d, self.f_digit)
                    for d in distances
                    if round(d, self.f_digit) != 0
                ]

            dist_count = Counter(distances)
            unique_dist, cum_count = [], []
            s = 0
            for dist_val, count in dist_count.items():
                s += count
                if dist_val > 0:
                    unique_dist.append(dist_val)
                    cum_count.append(s)

            if len(unique_dist) > 2:
                x = np.log(unique_dist)
                y = np.log(cum_count)
                from scipy.stats import linregress

                slope, _, _, _, _ = linregress(x, y)
                node_dimensions[node] = slope
            else:
                node_dimensions[node] = 0
        return node_dimensions

    def _compute_centralities(self) -> Dict[str, List[float]]:
        nk_graph = self._get_nk_graph()

        nfd_centrality = self._compute_node_dimension()

        # Closeness centrality (use inverted weights as distance for weighted graphs)
        if self.weighted:
            inv_graph = self._make_inverted_weight_graph()
            closeness_values = (
                nk.centrality.Closeness(inv_graph, True, True).run().scores()
            )
        else:
            closeness_values = (
                nk.centrality.Closeness(nk_graph, True, True).run().scores()
            )

        # Degree centrality (weighted = sum of edge weights, unweighted = count)
        if self.weighted:
            degree_values = [self.graph.weighted_degree(u) for u in self.graph.nodes()]
        else:
            degree_values = [self.graph.degree(u) for u in self.graph.nodes()]

        # Local clustering coefficient
        cluster_values = (
            nk.centrality.LocalClusteringCoefficient(nk_graph).run().scores()
        )

        return {
            "nfd": list(nfd_centrality.values()),
            "closeness": list(closeness_values),
            "degree": degree_values,
            "clustering": list(cluster_values),
        }

    def _compute_betweenness(self) -> List[float]:
        if self.weighted:
            inv_graph = self._make_inverted_weight_graph()
            bt = nk.centrality.Betweenness(inv_graph, normalized=True).run().scores()
        else:
            nk_graph = self._get_nk_graph()
            bt = nk.centrality.Betweenness(nk_graph, normalized=True).run().scores()
        return bt

    def _compute_ollivier_ricci_curvature(self) -> List[float]:
        """Dispatch to the configured ORC backend.

        Controlled by ``BaseConfig.ORC_BACKEND``:
        * ``"native"`` – pure NetworKit + scipy  (default, no extra deps)
        * ``"grc"``    – GraphRicciCurvature lib  (requires networkx + GRC)
        """
        backend = BaseConfig.ORC_BACKEND.lower()
        if backend == "native":
            return _ollivier_ricci_curvature(
                self.graph, alpha=0.5, weighted=self.weighted
            )
        if backend == "grc":
            return self._compute_orc_grc()
        raise ValueError(f"Unknown ORC_BACKEND={backend!r}. " "Use 'native' or 'grc'.")

    def _compute_orc_grc(self) -> List[float]:
        """Ollivier-Ricci curvature via the GraphRicciCurvature library."""
        import networkx as nx
        from GraphRicciCurvature.OllivierRicci import OllivierRicci

        nx_graph = self.graph.to_networkx()

        if self.weighted:
            for u, v, d in nx_graph.edges(data=True):
                if d.get("weight", 0) != 0:
                    d["weight"] = 1.0 / d["weight"]
            orc = OllivierRicci(
                nx.convert_node_labels_to_integers(nx_graph),
                alpha=0.5,
                verbose="ERROR",
                weight="weight",
            )
        else:
            orc = OllivierRicci(
                nx.convert_node_labels_to_integers(nx_graph),
                alpha=0.5,
                verbose="ERROR",
                weight=None,
            )
        orc.compute_ricci_curvature()
        return [d["ricciCurvature"] for _, _, d in orc.G.edges(data=True)]

    def _compute_assortativity(self) -> float:
        """Degree-degree Pearson correlation coefficient (manual computation)."""
        from scipy.stats import pearsonr

        x, y = [], []
        if self.weighted:
            for u, v in self.graph.edges():
                x.append(self.graph.weighted_degree(u))
                y.append(self.graph.weighted_degree(v))
        else:
            for u, v in self.graph.edges():
                x.append(self.graph.degree(u))
                y.append(self.graph.degree(v))

        # Symmetrize for undirected graphs
        all_x = x + y
        all_y = y + x
        if len(set(all_x)) < 2 or len(set(all_y)) < 2:
            return 0.0
        r, _ = pearsonr(all_x, all_y)
        return float(r)

    def _compute_eigenvector_centrality(self) -> List[float]:
        if self.graph.is_connected():
            nk_graph = self._get_nk_graph()
        else:
            lcc = self.graph.largest_connected_component()
            nk_graph = lcc.nk

        try:
            ec = nk.centrality.EigenvectorCentrality(nk_graph, tol=1e-6)
            ec.run()
            return ec.scores()
        except Exception:
            return [float("nan")] * nk_graph.numberOfNodes()

    def _compute_diameter(self) -> float:
        if self.graph.is_connected():
            nk_graph = self._get_nk_graph()
        else:
            lcc = self.graph.largest_connected_component()
            nk_graph = lcc.nk
        diam = nk.distance.Diameter(nk_graph, algo=nk.distance.DiameterAlgo.exact)
        diam.run()
        return diam.getDiameter()[0]

    def analyze_graph(self) -> Dict[str, List]:
        with self.set_q(self.full_q):
            with self.set_f_digit(1):
                tau_list, zq_list = self._compute_multifractal_taus()

            alpha_0, width, al_list, fal_list = self._compute_n_spectrum(tau_list)

            with self.set_f_digit(2):
                dim_list, dim_max, dim_min, dim_diff, valid_q = (
                    self._compute_n_dimension(tau_list)
                )

            centralities = self._compute_centralities()
            betweenness = self._compute_betweenness()
            ricci_list = self._compute_ollivier_ricci_curvature()
            assort = self._compute_assortativity()
            eigen_list = self._compute_eigenvector_centrality()
            diam = self._compute_diameter()

        return {
            "tau_list": tau_list,
            "alpha_0": alpha_0,
            "width": width,
            "al_list": al_list,
            "fal_list": fal_list,
            "dim_list": dim_list,
            "dim_diff": dim_diff,
            "valid_q": valid_q,
            "diameter": diam,
            "assortativity": assort,
            "nfd_dist": centralities["nfd"],
            "closeness_dist": centralities["closeness"],
            "degree_dist": centralities["degree"],
            "clustering_dist": centralities["clustering"],
            "betweenness_dist": betweenness,
            "ricci_dist": ricci_list,
            "eigen_dist": eigen_list,
        }

    @contextmanager
    def set_q(self, q_list):
        original_q = self.q_
        try:
            self.q_ = q_list
            yield
        finally:
            self.q_ = original_q

    @contextmanager
    def set_f_digit(self, f_digit_value):
        original_f_digit = self.f_digit
        try:
            self.f_digit = f_digit_value
            yield
        finally:
            self.f_digit = original_f_digit

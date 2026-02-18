# src/analysis/multifractal_analyzer.py
import logging
import math
from collections import Counter
from contextlib import contextmanager
from dataclasses import astuple, dataclass
from typing import Dict, List, Tuple

import networkit as nk
import numpy as np
from scipy import sparse as sp
from scipy.optimize import linprog
from scipy.stats import linregress

from config import BaseConfig
from graph.synth_graph import SynthGraph

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sparse transport-LP for Wasserstein distance
# ---------------------------------------------------------------------------

_transport_cache: Dict[Tuple[int, int], sp.csc_matrix] = {}


def _build_transport_constraints(n: int, m: int) -> sp.csc_matrix:
    """Sparse equality-constraint matrix for the 1-Wasserstein LP.

    Shape (n+m, n*m).  Rows 0..n-1 enforce row-sum = p[i],
    rows n..n+m-1 enforce col-sum = q[j].
    """
    nm = n * m
    row_r = np.repeat(np.arange(n), m)
    col_r = np.arange(nm)
    row_c = n + np.repeat(np.arange(m), n)
    col_c = np.tile(np.arange(n), m) * m + np.repeat(np.arange(m), n)

    data = np.ones(2 * nm)
    rows = np.concatenate([row_r, row_c])
    cols = np.concatenate([col_r, col_c])
    return sp.csc_matrix((data, (rows, cols)), shape=(n + m, nm))


def _get_transport_constraints(n: int, m: int) -> sp.csc_matrix:
    key = (n, m)
    mat = _transport_cache.get(key)
    if mat is None:
        mat = _build_transport_constraints(n, m)
        _transport_cache[key] = mat
    return mat


def _wasserstein_lp(p: np.ndarray, q: np.ndarray, cost: np.ndarray) -> float:
    """Exact 1-Wasserstein distance between discrete measures via sparse LP."""
    n, m = len(p), len(q)
    A_eq = _get_transport_constraints(n, m)
    b_eq = np.concatenate([p, q])
    res = linprog(cost.ravel(), A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    return res.fun if res.success else float("nan")


# ---------------------------------------------------------------------------
# Native Ollivier-Ricci curvature (NetworKit + scipy, no networkx needed)
# ---------------------------------------------------------------------------


def _ollivier_ricci_curvature(
    graph: SynthGraph,
    alpha: float = 0.5,
    weighted: bool = False,
    base: float = math.e,
    exp_power: float = 2,
) -> List[float]:
    """Compute Ollivier-Ricci curvature for every edge.

    For each edge (u, v) the curvature is

        κ(u, v) = 1 − W₁(μ_u, μ_v) / d(u, v)

    where μ_x places mass *alpha* on x and spreads (1 − alpha) over x's
    neighbours (weighted by ``base ** (-dist ** exp_power)`` when *weighted*,
    uniformly otherwise), W₁ is the 1-Wasserstein distance under the
    shortest-path metric, and d(u,v) is the direct edge distance.

    The default ``base`` / ``exp_power`` values match the convention used by
    the GraphRicciCurvature library.
    """
    nk_graph = graph.nk

    if weighted:
        n = nk_graph.numberOfNodes()
        dist_graph = nk.Graph(n, weighted=True)
        for u, v, w in nk_graph.iterEdgesWeights():
            dist_graph.addEdge(u, v, 1.0 / w if w != 0 else float("inf"))
    else:
        dist_graph = nk_graph

    apsp = nk.distance.APSP(dist_graph)
    apsp.run()
    dist_matrix = np.asarray(apsp.getDistances())

    nbrs = [list(nk_graph.iterNeighbors(u)) for u in range(nk_graph.numberOfNodes())]

    curvatures: List[float] = []
    for u, v in graph.edges():
        nbrs_u = nbrs[u]
        support_u = [u] + nbrs_u
        mu_u = np.empty(len(support_u))
        if nbrs_u:
            mu_u[0] = alpha
            if weighted:
                w_u = np.array(
                    [
                        base ** (-(dist_graph.weight(u, nb) ** exp_power))
                        for nb in nbrs_u
                    ]
                )
                mu_u[1:] = (1.0 - alpha) * w_u / w_u.sum()
            else:
                mu_u[1:] = (1.0 - alpha) / len(nbrs_u)
        else:
            mu_u[0] = 1.0

        nbrs_v = nbrs[v]
        support_v = [v] + nbrs_v
        mu_v = np.empty(len(support_v))
        if nbrs_v:
            mu_v[0] = alpha
            if weighted:
                w_v = np.array(
                    [
                        base ** (-(dist_graph.weight(v, nb) ** exp_power))
                        for nb in nbrs_v
                    ]
                )
                mu_v[1:] = (1.0 - alpha) * w_v / w_v.sum()
            else:
                mu_v[1:] = (1.0 - alpha) / len(nbrs_v)
        else:
            mu_v[0] = 1.0

        cost = dist_matrix[np.ix_(support_u, support_v)]

        w1 = _wasserstein_lp(mu_u, mu_v, cost)

        d_uv = dist_graph.weight(u, v) if weighted else dist_matrix[u, v]
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
        self._inv_graph: nk.Graph | None = None

    # ---- helpers ----

    def _get_nk_graph(self) -> nk.Graph:
        """Return the underlying networkit graph."""
        return self.graph.nk

    def _get_inverted_weight_graph(self) -> nk.Graph:
        """Lazily build and cache a graph with inverted edge weights (1/w)."""
        if self._inv_graph is None:
            g = self.graph.nk
            n = g.numberOfNodes()
            inv = nk.Graph(n, weighted=True)
            for u, v, w in g.iterEdgesWeights():
                inv_w = 1.0 / w if w != 0 else float("inf")
                inv.addEdge(u, v, inv_w)
            self._inv_graph = inv
        return self._inv_graph

    @staticmethod
    def _weighted_clustering(nk_graph: nk.Graph) -> List[float]:
        """Weighted clustering matching ``nx.clustering(G, weight=...)``."""
        n = nk_graph.numberOfNodes()
        max_w = 0.0
        for u, v, w in nk_graph.iterEdgesWeights():
            if w > max_w:
                max_w = w
        if max_w == 0:
            max_w = 1.0

        nbrs = [set(nk_graph.iterNeighbors(u)) for u in range(n)]
        result = [0.0] * n
        for i in range(n):
            inbrs = nbrs[i]
            deg = len(inbrs)
            if deg < 2:
                continue
            wt_tri = 0.0
            seen = set()
            for j in inbrs:
                seen.add(j)
                wij = nk_graph.weight(i, j) / max_w
                jnbrs = nbrs[j] - seen
                common = inbrs & jnbrs
                for k in common:
                    wjk = nk_graph.weight(j, k) / max_w
                    wki = nk_graph.weight(k, i) / max_w
                    wt_tri += (wij * wjk * wki) ** (1.0 / 3.0)
            result[i] = (2.0 * wt_tri) / (deg * (deg - 1))
        return result

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

        apsp = nk.distance.APSP(nk_graph)
        apsp.run()
        all_distances = apsp.getDistances()

        n_list = []
        r_g_all_set = set()
        for node in self.graph.nodes():
            distances = all_distances[node]
            grow = [d for d in distances if 0 < d < 99999]
            grow.sort()
            if self.f_digit == 0:
                grow = [math.ceil(d) for d in grow]
            else:
                grow = [
                    round(d, self.f_digit) for d in grow if round(d, self.f_digit) != 0
                ]
            num = Counter(grow)
            r_g_all_set.update(num.keys())
            n_list.append(num)

        r_g_all = np.array(sorted(r_g_all_set))
        if len(r_g_all) < 2:
            return [0.0 for _ in self.q_], []

        diameter = r_g_all[-1]
        R = len(r_g_all)

        # --- vectorised ntw_mat via cumsum + searchsorted ---
        ntw_mat = np.ones((len(n_list), R))
        for i, num_counter in enumerate(n_list):
            if not num_counter:
                continue
            keys = sorted(num_counter.keys())
            dists = np.array(keys)
            counts = np.array([num_counter[k] for k in keys])
            cum = np.cumsum(counts)
            idx = np.searchsorted(dists, r_g_all, side="right")
            ntw_mat[i] += np.where(idx > 0, cum[idx - 1], 0)

        # --- vectorised partition function Z(q) ---
        norm_mat = ntw_mat / ntw_mat[:, -1:]  # (V, R)
        log_norm = np.log(norm_mat)  # (V, R)
        q_arr = np.asarray(self.q_)  # (Q,)
        # (Q,1,1) * (1,V,R) -> (Q,V,R)  then sum over V -> (Q,R)
        zq_arr = np.exp(q_arr[:, None, None] * log_norm[None, :, :]).sum(axis=1)

        # --- batch OLS (same formula as scipy.stats.linregress) ---
        log_r = np.log(r_g_all / diameter)  # (R,)
        log_zq = np.log(zq_arr)  # (Q, R)
        x_c = log_r - log_r.mean()
        y_c = log_zq - log_zq.mean(axis=1, keepdims=True)
        ss_xx = (x_c * x_c).sum()
        tau_arr = (y_c * x_c).sum(axis=1) / ss_xx  # (Q,)

        return tau_arr.tolist(), zq_arr.tolist()

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
            nk_graph = self._get_inverted_weight_graph()
        else:
            nk_graph = self._get_nk_graph()

        apsp = nk.distance.APSP(nk_graph)
        apsp.run()
        all_distances = apsp.getDistances()

        node_dimensions = {}
        for node in self.graph.nodes():
            distances = sorted(all_distances[node])
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
                slope, _, _, _, _ = linregress(x, y)
                node_dimensions[node] = slope
            else:
                node_dimensions[node] = 0
        return node_dimensions

    def _compute_centralities(self) -> Dict[str, List[float]]:
        nk_graph = self._get_nk_graph()

        nfd_centrality = self._compute_node_dimension()

        if self.weighted:
            inv_graph = self._get_inverted_weight_graph()
            closeness_values = (
                nk.centrality.Closeness(inv_graph, True, True).run().scores()
            )
        else:
            closeness_values = (
                nk.centrality.Closeness(nk_graph, True, True).run().scores()
            )

        if self.weighted:
            degree_values = [self.graph.weighted_degree(u) for u in self.graph.nodes()]
        else:
            degree_values = [self.graph.degree(u) for u in self.graph.nodes()]

        if self.weighted:
            cluster_values = self._weighted_clustering(nk_graph)
        else:
            cluster_values = list(
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
            inv_graph = self._get_inverted_weight_graph()
            bt = nk.centrality.Betweenness(inv_graph, normalized=True).run().scores()
        else:
            nk_graph = self._get_nk_graph()
            bt = nk.centrality.Betweenness(nk_graph, normalized=True).run().scores()
        return bt

    def _compute_ollivier_ricci_curvature(self) -> List[float]:
        return _ollivier_ricci_curvature(self.graph, alpha=0.5, weighted=self.weighted)

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
        """Unweighted hop diameter (matches nx.diameter behaviour)."""
        if self.graph.is_connected():
            nk_graph = self._get_nk_graph()
        else:
            lcc = self.graph.largest_connected_component()
            nk_graph = lcc.nk

        if nk_graph.isWeighted():
            uw = nk.Graph(nk_graph.numberOfNodes(), weighted=False)
            for u, v in nk_graph.iterEdges():
                uw.addEdge(u, v)
            nk_graph = uw

        algo = (
            getattr(nk.distance.DiameterAlgo, "Exact", None)
            or nk.distance.DiameterAlgo.exact
        )
        diam = nk.distance.Diameter(nk_graph, algo=algo)
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

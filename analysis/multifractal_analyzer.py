import logging
import math
from collections import Counter
from contextlib import contextmanager
from dataclasses import astuple, dataclass
from typing import Dict, List

import networkit as nk
import numpy as np
import ot
from scipy import sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.stats import linregress

from graphs.synth_graph import SynthGraph

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Wasserstein distance via optimal transport
# ---------------------------------------------------------------------------


def _wasserstein_lp(p: np.ndarray, q: np.ndarray, cost: np.ndarray) -> float:
    """Exact 1-Wasserstein distance between discrete measures.

    ``ot.emd2`` is a network simplex in C.  This was a hand-built sparse LP fed
    to ``scipy.optimize.linprog``; on the supports that arise here — 2x3 up to
    6x6 — scipy's per-call setup dominated, and it measured 14.4x slower over
    the 1052 problems one curvature run solves, for answers identical to 4e-16.
    """
    return ot.emd2(
        np.ascontiguousarray(p, dtype=np.float64),
        np.ascontiguousarray(q, dtype=np.float64),
        np.ascontiguousarray(cost, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Native Ollivier-Ricci curvature (NetworKit + POT, no networkx needed)
# ---------------------------------------------------------------------------


def _ollivier_ricci_curvature(
    graph: SynthGraph,
    dist_graph: nk.Graph,
    dist_matrix: np.ndarray,
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

    *dist_graph* and its *dist_matrix* come from the caller.  This used to build
    an equivalent graph itself and run its own all-pairs shortest path, making
    curvature the third full APSP of a single analysis.

    The default ``base`` / ``exp_power`` values match the convention used by
    the GraphRicciCurvature library.
    """
    nk_graph = graph.nk

    nbrs = [list(nk_graph.iterNeighbors(u)) for u in range(nk_graph.numberOfNodes())]

    curvatures: List[float] = []
    for u, v in graph.edges():
        nbrs_u = nbrs[u]
        support_u = [u] + nbrs_u
        mu_u = np.empty(len(support_u))
        if nbrs_u:
            mu_u[0] = alpha
            if weighted:
                d_u = np.fromiter(
                    (dist_graph.weight(u, nb) for nb in nbrs_u),
                    dtype=np.float64,
                    count=len(nbrs_u),
                )
                mu_u[1:] = _neighbour_masses(d_u, alpha, base, exp_power)
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
                d_v = np.fromiter(
                    (dist_graph.weight(v, nb) for nb in nbrs_v),
                    dtype=np.float64,
                    count=len(nbrs_v),
                )
                mu_v[1:] = _neighbour_masses(d_v, alpha, base, exp_power)
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


def _neighbour_masses(
    distances: np.ndarray, alpha: float, base: float, exp_power: float
) -> np.ndarray:
    """Spread ``1 - alpha`` over neighbours in proportion to ``base ** -d**p``.

    Written as a shifted softmax rather than the literal expression, which
    underflows on real data: edge widths are of the order 0.03, so the inverted
    distance ``1/w`` reaches ~30, and ``e ** -(30**2)`` is exactly 0 in double
    precision.  Once every neighbour of a node underflows the normalisation
    divides 0 by 0, and the resulting NaN aborts the transport solver — which is
    why weighted analysis could not complete at all.

    Subtracting the largest exponent scales numerator and denominator by the
    same constant, so it cancels exactly.  This is the same number, kept inside
    the representable range; nothing is clamped or defaulted.
    """
    exponents = -(distances**exp_power) * math.log(base)
    affinities = np.exp(exponents - exponents.max())
    return (1.0 - alpha) * affinities / affinities.sum()


def _principal_eigenvector(nk_graph: nk.Graph) -> List[float]:
    """Eigenvector centrality: the principal eigenvector of the adjacency matrix.

    Lanczos rather than power iteration.  ``nk.centrality.EigenvectorCentrality``
    iterates, and on a graph with a small spectral gap it is both slow *and*
    inaccurate: 22s at ``tol=1e-9`` for 705 nodes, and still 2.5% (max relative)
    away from the ``tol=1e-12`` answer.  ``eigsh`` matches that answer to ~7e-6
    in 0.01s, so tightening the tolerance stopped being a trade-off at all.

    Returned unit-L2-normalised and non-negative, matching networkit's
    convention.  Perron-Frobenius makes the principal eigenvector of a connected
    graph single-signed, so whichever sign ``eigsh`` returns carries no meaning.
    """
    n = nk_graph.numberOfNodes()
    weighted = nk_graph.isWeighted()

    rows: List[int] = []
    cols: List[int] = []
    values: List[float] = []
    for u, v in nk_graph.iterEdges():
        w = nk_graph.weight(u, v) if weighted else 1.0
        rows += [u, v]
        cols += [v, u]
        values += [w, w]

    adjacency = sp.coo_matrix((values, (rows, cols)), shape=(n, n)).tocsr()
    _, vectors = eigsh(adjacency.astype(np.float64), k=1, which="LA")
    principal = np.abs(vectors[:, 0])
    return (principal / np.linalg.norm(principal)).tolist()


@dataclass
class MultifractalErrorFeatures:
    holder_exponent: float
    spectrum_width: float


def _generate_range(scale):
    return [q / 100 for q in range(-scale, scale + 1, 10)]


class MultifractalAnalyzer:
    small_q = _generate_range(300)
    full_q = _generate_range(2000)

    def __init__(
        self,
        graph: SynthGraph,
        measure_weighted: bool,
        full_q_band: bool,
    ):
        """Analyze *graph*.

        Both settings are required: a spawned child re-imports ``configs``
        unmutated, so a ``BaseConfig`` fallback here would silently analyze
        with the wrong q-band rather than fail.
        """
        self.graph = graph
        self.f_digit = 0
        self.q_ = None
        self._full_q_band = full_q_band
        assert graph.is_weighted() or not measure_weighted, (
            "measure_weighted=True but this graph carries no edge weights. "
            "Either the input has no weights (set MEASURE_WEIGHTED=False) or "
            "the weights were lost on the way here."
        )
        self.weighted = measure_weighted
        self._inv_graph: nk.Graph | None = None
        self._uw_graph: nk.Graph | None = None
        self._distances: Dict[int, list] = {}

    # ---- helpers ----

    def _get_nk_graph(self) -> nk.Graph:
        return self.graph.nk

    def _get_analysis_graph(self) -> nk.Graph:
        """Return a graph matching the current analysis mode.

        When ``self.weighted`` is False but the underlying graph carries
        edge weights, returns a cached unweighted copy so that algorithms
        like APSP and betweenness ignore the stored weights.
        """
        g = self.graph.nk
        if not self.weighted and g.isWeighted():
            if self._uw_graph is None:
                n = g.numberOfNodes()
                uw = nk.Graph(n, weighted=False)
                for u, v in g.iterEdges():
                    uw.addEdge(u, v)
                self._uw_graph = uw
            return self._uw_graph
        return g

    def _get_distances(self, nk_graph: nk.Graph) -> list:
        """All-pairs shortest paths for *nk_graph*, computed at most once.

        A full ``analyze_graph`` asked for the same distances three times.  Both
        graphs this is ever called with are cached attributes, so their identity
        is stable for the analyzer's lifetime and safe to key on.
        """
        key = id(nk_graph)
        if key not in self._distances:
            apsp = nk.distance.APSP(nk_graph)
            apsp.run()
            self._distances[key] = apsp.getDistances()
        return self._distances[key]

    def _get_inverted_weight_graph(self) -> nk.Graph:
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
        q_band = self.full_q if self._full_q_band else self.small_q
        with self.set_q(q_band):
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
        nk_graph = self._get_analysis_graph()
        all_distances = self._get_distances(nk_graph)

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
            nk_graph = self._get_analysis_graph()

        all_distances = self._get_distances(nk_graph)

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
        nk_graph = self._get_analysis_graph()

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
            nk_graph = self._get_analysis_graph()
            bt = nk.centrality.Betweenness(nk_graph, normalized=True).run().scores()
        return bt

    def _compute_ollivier_ricci_curvature(self) -> List[float]:
        dist_graph = (
            self._get_inverted_weight_graph()
            if self.weighted
            else self._get_analysis_graph()
        )
        return _ollivier_ricci_curvature(
            self.graph,
            dist_graph,
            np.asarray(self._get_distances(dist_graph)),
            alpha=0.5,
            weighted=self.weighted,
        )

    def _compute_assortativity(self) -> float:
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
            nk_graph = self._get_analysis_graph()
        else:
            lcc = self.graph.largest_connected_component()
            nk_graph = lcc.nk
            if not self.weighted and nk_graph.isWeighted():
                n = nk_graph.numberOfNodes()
                uw = nk.Graph(n, weighted=False)
                for u, v in nk_graph.iterEdges():
                    uw.addEdge(u, v)
                nk_graph = uw

        try:
            return _principal_eigenvector(nk_graph)
        except Exception:
            return [float("nan")] * nk_graph.numberOfNodes()

    def _compute_diameter(self) -> float:
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

        # nk.distance.Diameter returns an int, silently flooring weighted
        # distances (4.7 -> 4, 0.3 -> 0).  Safe only because the graph above
        # is unweighted, making the hop diameter integral by definition.
        assert not nk_graph.isWeighted(), "hop diameter needs an unweighted graph"

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

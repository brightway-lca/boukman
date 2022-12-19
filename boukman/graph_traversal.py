from typing import List

import numpy as np
import sknetwork as skn
from scipy import sparse

try:
    import bw2calc as bc
    import bw2data as bd
except:
    bc = None

    class Dummy:
        Node = None
        Edge = None

    bd = Dummy()


def to_normalized_adjacency_matrix(
    matrix: sparse.spmatrix, log_transform: bool = True
) -> sparse.csr_matrix:
    """Take a technosphere matrix constructed with Brightway conventions, and return a normalized adjacency matrix.

    In the adjacency matrix A, `A[i,j]` indicates a directed edge **from** row `i` **to** column `j`. This fits
    the Brightway mode, where `A[i,j]` means **activity** `j` consumes the output of activity `i`. In other words,
    this matrix is ready for traversal from a functional unit to a supplier; to go in the opposite direction,
    use the transpose of this matrix.

    Normalization is done to remove the effect of activities which don't produce one unit of their reference product.
    For example, if activity `foo` produces two units of `bar` and consumes two units of `baz`, the weight of the
    `baz` edge should be 2 / 2 = 1.

    In addition to this normalization, we subtract the diagonal and flip the signs of all matrix values. Flipping
    the sign is needed because we want to use a shortest path algorithm, but actually want the longest path. The
    longest path is the path with the highest weight, i.e. the path where the most consumption occurs on.

    By default, we also take the natural log of the data values. This is because our supply chain is multiplicative,
    not additive, and :math:`a \\cdot b = e^{\\ln(a) + \\ln(b)}`. The idea of using the log was borrowed from `David Richardby on Stack Overflow <https://cs.stackexchange.com/questions/83656/traverse-direct-graph-with-multiplicative-edges>`__.

    Assumes that production amounts are on the diagonal.
    """
    matrix = matrix.tocsr().T

    # TBD these values should the NET production values, i.e. we use the production exchanges to get the matrix
    # indices, ensure that we have 1-1 match for production-activity, construct the normalization vector, turn
    # into a diagonal matrix, and then multiply
    normalization = sparse.diags(-1 * matrix.diagonal())
    normalized = (matrix * normalization) + sparse.eye(*matrix.shape)

    if log_transform:
        normalized = normalized.tocoo()
        normalized.data = np.log(normalized.data) * -1
        normalized = normalized.tocsr()

    return normalized


def get_path_from_matrix(
    matrix: sparse.spmatrix, source: int, target: int, algorithm: str = "BF"
) -> List:
    """Get the path with the most mass or energetic flow from ``source`` (the function unit) to ``target`` (something deep in the supply chain). Both ``source`` and ``target`` are integer matrix indices.

    ``algorithm`` should be either ``BF`` (Bellman-Ford) or ``J`` (Johnson). Dijkstra is not recommended as we have negative weights.

    Returns a list like ``[source, int, int, int, target]``."""
    return skn.path.get_shortest_path(
        adjacency=to_normalized_adjacency_matrix(matrix=matrix),
        sources=source,
        targets=target,
        method=algorithm,
        unweighted=False,
    )


def path_as_brightway_objects(
    source_node: bd.Node, target_node: bd.Node
) -> List[bd.Edge]:
    if bc is None:
        raise ImportError("Brightway not available")

    lca = bc.LCA({source_node: 1, target_node: 1})
    lca.lci()

    path = skn.path.get_shortest_path(
        adjacency=to_normalized_adjacency_matrix(mapped_matrix=lca.technosphere_mm),
        sources=lca.activity_dict[source_node.id],
        targets=lca.activity_dict[target_node.id],
        method="BF",
        unweighted=False,
    )

    return [
        (
            bd.get_node(id=lca.dicts.product.reversed[x]),
            bd.get_node(id=lca.dicts.activity.reversed[y]),
            -1 * lca.technosphere_matrix[x, y],
        )
        for x, y in zip(path[:-1], path[1:])
    ]

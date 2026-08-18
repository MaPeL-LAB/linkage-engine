"""Multi-source entity resolution orchestrator and global crosswalk exporter."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mapel_linkage.assignment.contracts import AssignmentEdgeBatch

from mapel_linkage.clustering.contracts import (
    CandidateEdge,
    ClusteringPlan,
    GlobalCrosswalkEntry,
    MultiSourceGraph,
    MultiSourceResolutionResult,
    _compute_clustering_digest,
)
from mapel_linkage.clustering.solvers import MultiSourceClusterer
from mapel_linkage.domain.errors import ClusteringError


class MultiSourceEntityResolver:
    """Multi-source entity resolver supporting N >= 3 datasets and constraint-safe clustering."""

    @classmethod
    def resolve(
        cls,
        *,
        datasets: Mapping[str, Iterable[str]],
        candidate_batches: Sequence[AssignmentEdgeBatch]
        | Mapping[tuple[str, str], AssignmentEdgeBatch]
        | None = None,
        candidate_edges: Iterable[CandidateEdge] | None = None,
        edges: Mapping[tuple[str, str], float] | Iterable[tuple[str, str, float]] | None = None,
        cannot_link_pairs: Iterable[tuple[str, str]] | None = None,
        cannot_link: Iterable[tuple[str, str]] | None = None,
        plan: ClusteringPlan | None = None,
        min_datasets: int = 1,
    ) -> MultiSourceResolutionResult:
        """Resolve entities across multiple source datasets."""
        clustering_plan = plan or ClusteringPlan()

        if not datasets:
            raise ClusteringError(
                "ML-CLUSTER-001", "The multi-source graph node set cannot be empty."
            )
        if len(datasets) < min_datasets:
            raise ClusteringError(
                "ML-CLUSTER-001",
                f"Multi-source resolution requires at least {min_datasets} datasets.",
            )

        # 1. Build node dictionary {record_key: dataset_id}
        node_map: dict[str, str] = {}
        for ds_id, keys in datasets.items():
            ds_str = str(ds_id).strip()
            if not ds_str:
                raise ClusteringError("ML-CLUSTER-001", "Dataset ID cannot be empty.")
            for k in keys:
                rec_str = str(k).strip()
                if not rec_str:
                    raise ClusteringError("ML-CLUSTER-001", "Record key cannot be empty.")
                if rec_str in node_map:
                    raise ClusteringError(
                        "ML-CLUSTER-001", "Duplicate record key found across datasets."
                    )
                node_map[rec_str] = ds_str

        if not node_map:
            raise ClusteringError(
                "ML-CLUSTER-001", "The multi-source graph node set cannot be empty."
            )

        # 2. Build edge dictionary {(u, v): weight}
        edge_map: dict[tuple[str, str], float] = {}

        if edges is not None:
            if isinstance(edges, Mapping):
                for (u, v), w in edges.items():
                    edge_map[(u, v)] = float(w)
            else:
                for u, v, w in edges:
                    edge_map[(u, v)] = float(w)

        if candidate_edges is not None:
            for c_edge in candidate_edges:
                edge_map[(c_edge.source_record_key, c_edge.target_record_key)] = c_edge.probability

        if candidate_batches is not None:
            batch_list = (
                candidate_batches.values()
                if isinstance(candidate_batches, Mapping)
                else candidate_batches
            )
            for batch in batch_list:
                for idx, (src_k, tgt_k) in enumerate(batch.pair_references):
                    prob = float(batch.probabilities[idx])
                    edge_map[(src_k, tgt_k)] = prob

        # 3. Build cannot-link constraints
        cl_list: list[tuple[str, str]] = []
        if cannot_link_pairs is not None:
            cl_list.extend(cannot_link_pairs)
        if cannot_link is not None:
            cl_list.extend(cannot_link)

        # 4. Construct MultiSourceGraph
        graph = MultiSourceGraph.build(
            nodes=node_map,
            edges=edge_map,
            cannot_link=cl_list,
        )

        # 5. Solve clustering
        clustering_res = MultiSourceClusterer.solve(graph, clustering_plan)

        # 6. Generate GlobalCrosswalkEntry rows
        crosswalk_entries: list[GlobalCrosswalkEntry] = []
        for cluster in clustering_res.clusters:
            for member_key in cluster.member_record_keys:
                ds_id = graph.get_dataset(member_key)
                is_canon = member_key == cluster.canonical_record_key
                conf = cluster.mean_edge_weight

                crosswalk_entries.append(
                    GlobalCrosswalkEntry(
                        record_key=member_key,
                        dataset_id=ds_id,
                        global_entity_id=cluster.entity_digest,
                        is_canonical=is_canon,
                        cluster_size=len(cluster.member_record_keys),
                        confidence=conf,
                    )
                )

        crosswalk_entries.sort(key=lambda entry: (entry.dataset_id, entry.record_key))
        crosswalk_tuple = tuple(crosswalk_entries)

        singletons = sum(c.is_singleton for c in clustering_res.clusters)
        multi_clusters = len(clustering_res.clusters) - singletons

        # Compute resolution digest
        resolution_digest = _compute_clustering_digest(
            graph_digest=graph.graph_digest,
            algorithm=clustering_plan.algorithm,
            threshold=clustering_plan.threshold,
            clusters=clustering_res.clusters,
        )

        return MultiSourceResolutionResult(
            clustering_result=clustering_res,
            crosswalk_entries=crosswalk_tuple,
            graph=graph,
            total_records=graph.node_count,
            total_clusters=len(clustering_res.clusters),
            singleton_count=singletons,
            multi_record_cluster_count=multi_clusters,
            resolution_digest=resolution_digest,
            algorithm=clustering_plan.algorithm,
            threshold=clustering_plan.threshold,
        )

    @classmethod
    def export_crosswalk(
        cls,
        result: MultiSourceResolutionResult,
        destination_path: str | Path | None = None,
    ) -> list[dict[str, Any]]:
        """Export global crosswalk rows to dictionary list and optional file."""
        records = result.to_crosswalk_records()
        if destination_path is not None:
            dest = Path(destination_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.suffix.lower() == ".json":
                dest.write_text(
                    json.dumps(records, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            elif dest.suffix.lower() == ".csv":
                with dest.open("w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=[
                            "record_key",
                            "dataset_id",
                            "global_entity_id",
                            "is_canonical",
                            "cluster_size",
                            "confidence",
                        ],
                    )
                    writer.writeheader()
                    writer.writerows(records)
        return records

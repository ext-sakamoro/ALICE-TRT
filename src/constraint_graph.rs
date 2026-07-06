//! Deterministic constraint graph and greedy coloring for Phase 2
//! parallel constraint solve (v1.5.0 foundation, not yet wired into
//! `dispatch_iterations`).
//!
//! # What this module solves
//!
//! When the GPU distance-constraint kernel (v1.4.2) runs on N
//! constraints, the adapter currently dispatches them **sequentially**
//! — one workgroup per constraint per PGS iteration. That preserves
//! Gauss-Seidel order but leaves the GPU idle between dispatches.
//!
//! Phase 2 rearranges the constraint list so that mutually
//! **body-disjoint** groups (colors) can be dispatched **in one
//! compute call** with `dispatch_workgroups(constraints_in_color, ...)`.
//! Constraints inside a color touch disjoint body sets, so there is no
//! race hazard when they execute concurrently.
//!
//! This module provides:
//!
//! - [`ConstraintGraph`] — the conflict graph (one node per constraint;
//!   two constraints share an edge iff they share at least one body).
//! - [`ConstraintGraph::build`] — pair-based construction from a slice
//!   of `(body_a, body_b)` tuples in the same order the adapter's
//!   `distance_constraints` Vec exposes them.
//! - [`ConstraintGraph::greedy_color`] — deterministic greedy coloring
//!   that assigns each constraint the smallest color not used by any
//!   earlier-indexed neighbour, returning colors as sorted buckets of
//!   constraint indices.
//!
//! # Determinism
//!
//! Both `build` and `greedy_color` walk constraints in ascending index
//! order and use only [`Vec`] — no `HashMap`, no unordered iteration.
//! The result is bit-identical across platforms, threads, and rustc
//! versions, matching the determinism contract already established for
//! the v1.4.x GPU kernels.
//!
//! # v1.5.0 scope
//!
//! This release adds the algorithm and its tests only. The adapter
//! wire-up (batched rigid kernel with `workgroup_id`-indexed constraint
//! selection) lands in v1.5.1.

/// Conflict graph for a set of distance constraints.
///
/// Each node in the graph corresponds to one constraint (identified by
/// its index in the input slice). Two nodes are connected by an edge
/// iff the corresponding constraints share at least one body — which
/// means they cannot safely be solved in parallel because both would
/// write to the shared body's position slot.
///
/// Construction cost is O(N²) in the constraint count. For the target
/// workload sizes (ropes, chains, ragdoll ties: up to a few thousand
/// constraints), this is well below the per-frame budget and cost
/// scales trivially compared to the per-constraint sqrt+div dispatch.
///
/// Colored groups are recovered by [`Self::greedy_color`].
#[derive(Clone, Debug)]
pub struct ConstraintGraph {
    /// Number of constraints (nodes).
    n: usize,
    /// Adjacency list. `adjacency[i]` is the sorted list of node indices
    /// that share a body with constraint `i`.
    adjacency: Vec<Vec<usize>>,
}

impl ConstraintGraph {
    /// Build the conflict graph from a slice of `(body_a, body_b)`
    /// tuples. Constraint indices are the slice indices (so the caller
    /// preserves whatever push order it prefers).
    ///
    /// Runs in O(N²) time, O(N + E) memory. Both inputs and outputs are
    /// deterministic — no hash iteration, no thread-local state.
    #[must_use]
    pub fn build(pairs: &[(usize, usize)]) -> Self {
        let n = pairs.len();
        let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n];

        for i in 0..n {
            let (a_i, b_i) = pairs[i];
            for j in (i + 1)..n {
                let (a_j, b_j) = pairs[j];
                if a_i == a_j || a_i == b_j || b_i == a_j || b_i == b_j {
                    // Insertions are in ascending `j` order for `i`,
                    // and ascending `i` order for `j`, so no explicit
                    // sort is needed. `sort_unstable()` still runs as
                    // a cheap belt-and-braces determinism guard.
                    adjacency[i].push(j);
                    adjacency[j].push(i);
                }
            }
        }

        for adj in &mut adjacency {
            adj.sort_unstable();
        }

        Self { n, adjacency }
    }

    /// Number of constraints (nodes) in the graph.
    #[must_use]
    pub const fn constraint_count(&self) -> usize {
        self.n
    }

    /// Number of undirected edges in the graph. Each shared body between
    /// two constraints contributes exactly one edge, regardless of how
    /// many bodies they share.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        // Each edge is stored twice (once in each endpoint's adjacency
        // list); divide by 2 to recover the undirected count.
        let total: usize = self.adjacency.iter().map(Vec::len).sum();
        total / 2
    }

    /// Return the sorted list of neighbours of constraint `i`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= constraint_count()`.
    #[must_use]
    pub fn neighbours(&self, i: usize) -> &[usize] {
        &self.adjacency[i]
    }

    /// Greedy graph coloring in ascending constraint-index order.
    ///
    /// For each constraint `i` from `0` to `n - 1`, assigns the
    /// smallest color that no already-colored neighbour uses. Returns a
    /// `Vec<Vec<usize>>` where entry `c` holds the ascending-order list
    /// of constraint indices assigned to color `c`.
    ///
    /// # Determinism
    ///
    /// The algorithm iterates the adjacency list (which is
    /// sorted by [`Self::build`]) in ascending order, so the result is
    /// bit-identical on every platform. Not stashed inside a `HashMap`,
    /// not spawned across threads — deterministic by construction.
    ///
    /// # Optimality
    ///
    /// Greedy coloring is not optimal in general; the chromatic number
    /// may be lower for some graphs. The target workloads (ropes,
    /// chains, ragdoll ties) are typically bipartite or near-bipartite,
    /// where greedy already produces the optimal 2-3 colors.
    #[must_use]
    pub fn greedy_color(&self) -> Vec<Vec<usize>> {
        let mut color_of = vec![usize::MAX; self.n];
        let mut num_colors: usize = 0;

        for i in 0..self.n {
            // Find the smallest color not used by an already-colored
            // neighbour of `i`. Only earlier-indexed neighbours (`j < i`)
            // carry a color at this point, so the search space is
            // bounded by the highest color seen so far plus one.
            let mut c: usize = 0;
            loop {
                let clashes = self.adjacency[i].iter().any(|&j| j < i && color_of[j] == c);
                if !clashes {
                    break;
                }
                c += 1;
            }
            color_of[i] = c;
            if c + 1 > num_colors {
                num_colors = c + 1;
            }
        }

        let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); num_colors];
        for (i, &c) in color_of.iter().enumerate() {
            buckets[c].push(i);
        }
        buckets
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Empty graph → zero nodes, zero edges, zero colors.
    #[test]
    fn empty_graph_has_no_colors() {
        let graph = ConstraintGraph::build(&[]);
        assert_eq!(graph.constraint_count(), 0);
        assert_eq!(graph.edge_count(), 0);
        let colors = graph.greedy_color();
        assert!(colors.is_empty());
    }

    /// Single constraint → one node, zero edges, one color containing it.
    #[test]
    fn single_constraint_gets_one_color() {
        let graph = ConstraintGraph::build(&[(0, 1)]);
        assert_eq!(graph.constraint_count(), 1);
        assert_eq!(graph.edge_count(), 0);
        assert_eq!(graph.greedy_color(), vec![vec![0]]);
    }

    /// Two disjoint constraints (bodies never overlap) → both in the
    /// same color because they're safe to solve concurrently.
    #[test]
    fn disjoint_constraints_share_a_color() {
        // C0: (0, 1), C1: (2, 3) — no shared bodies.
        let graph = ConstraintGraph::build(&[(0, 1), (2, 3)]);
        assert_eq!(graph.constraint_count(), 2);
        assert_eq!(graph.edge_count(), 0);
        assert_eq!(graph.greedy_color(), vec![vec![0, 1]]);
    }

    /// Two constraints sharing a body → separate colors (K2).
    #[test]
    fn shared_body_forces_separate_colors() {
        // C0: (0, 1), C1: (1, 2) — share body 1.
        let graph = ConstraintGraph::build(&[(0, 1), (1, 2)]);
        assert_eq!(graph.constraint_count(), 2);
        assert_eq!(graph.edge_count(), 1);
        assert_eq!(graph.neighbours(0), &[1]);
        assert_eq!(graph.neighbours(1), &[0]);
        assert_eq!(graph.greedy_color(), vec![vec![0], vec![1]]);
    }

    /// Chain of five bodies (0-1-2-3-4) with four constraints →
    /// alternating 2 colors, bipartite optimum.
    #[test]
    fn chain_graph_uses_two_colors() {
        // Constraint list: (0,1), (1,2), (2,3), (3,4).
        let pairs = [(0, 1), (1, 2), (2, 3), (3, 4)];
        let graph = ConstraintGraph::build(&pairs);
        assert_eq!(graph.constraint_count(), 4);
        assert_eq!(graph.edge_count(), 3);
        // Neighbours: C0-C1, C1-C2, C2-C3.
        assert_eq!(graph.neighbours(0), &[1]);
        assert_eq!(graph.neighbours(1), &[0, 2]);
        assert_eq!(graph.neighbours(2), &[1, 3]);
        assert_eq!(graph.neighbours(3), &[2]);
        // Greedy coloring: C0=0, C1=1, C2=0, C3=1 → buckets [[0,2], [1,3]].
        assert_eq!(graph.greedy_color(), vec![vec![0, 2], vec![1, 3]]);
    }

    /// Triangle of three bodies with three constraints closing the loop
    /// → K3 → needs 3 colors.
    #[test]
    fn triangle_graph_needs_three_colors() {
        // Constraints: (0,1), (1,2), (0,2).
        let pairs = [(0, 1), (1, 2), (0, 2)];
        let graph = ConstraintGraph::build(&pairs);
        assert_eq!(graph.constraint_count(), 3);
        assert_eq!(graph.edge_count(), 3);
        // Every pair shares a body:
        //   C0-C1 share body 1, C1-C2 share body 2, C0-C2 share body 0.
        assert_eq!(graph.neighbours(0), &[1, 2]);
        assert_eq!(graph.neighbours(1), &[0, 2]);
        assert_eq!(graph.neighbours(2), &[0, 1]);
        assert_eq!(graph.greedy_color(), vec![vec![0], vec![1], vec![2]]);
    }

    /// Star: one central body shared by every constraint → K_N →
    /// needs N colors.
    #[test]
    fn star_graph_needs_n_colors() {
        // Body 0 is central. Constraints: (0,1), (0,2), (0,3), (0,4), (0,5).
        let pairs = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)];
        let graph = ConstraintGraph::build(&pairs);
        assert_eq!(graph.constraint_count(), 5);
        // Every pair shares body 0, so this is K5 with 10 edges.
        assert_eq!(graph.edge_count(), 10);
        for i in 0..5 {
            let expected: Vec<usize> = (0..5).filter(|&j| j != i).collect();
            assert_eq!(graph.neighbours(i), expected.as_slice());
        }
        assert_eq!(
            graph.greedy_color(),
            vec![vec![0], vec![1], vec![2], vec![3], vec![4]]
        );
    }

    /// Determinism check: the same input pairs produce byte-identical
    /// coloring across repeated calls, matching the wider Fix128 kernel
    /// determinism contract.
    #[test]
    fn coloring_is_deterministic_across_repeated_calls() {
        let pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (2, 5)];
        let graph = ConstraintGraph::build(&pairs);
        let a = graph.greedy_color();
        let b = graph.greedy_color();
        let c = ConstraintGraph::build(&pairs).greedy_color();
        assert_eq!(a, b);
        assert_eq!(a, c);
    }
}

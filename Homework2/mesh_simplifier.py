import sys
import argparse
import time
import numpy as np
import heapq
from collections import defaultdict
import os


def read_obj(filepath):
    """Read a triangle mesh from an OBJ file."""
    vertices = []
    faces = []
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "v":
                vertices.append([float(x) for x in parts[1:4]])
            elif parts[0] == "f":
                face = []
                for p in parts[1:]:
                    idx = int(p.split("/")[0])
                    face.append(idx - 1 if idx > 0 else len(vertices) + idx)
                if len(face) == 3:
                    faces.append(face)
    return np.array(vertices, dtype=np.float64), faces


def write_obj(filepath, vertices, faces):
    """Write a triangle mesh to an OBJ file."""
    with open(filepath, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write("f " + " ".join(str(idx + 1) for idx in face) + "\n")


class MeshSimplifier:
    def __init__(self, vertices, faces):
        self.vertices = vertices.copy()
        self.faces = [list(f) for f in faces]

        self.deleted_faces = set()
        self.deleted_vertices = set()

        # vertex -> set of face indices
        self.vertex_faces = defaultdict(set)

        self.face_normals = {}
        self.face_quadrics = {}
        self.vertex_quadrics = {}

        # Priority queue with lazy deletion
        self.heap = []
        self.edge_best_pos = {}
        self.valid_edges = set()
        self.counter = 0

        self._build_adjacency()
        self._compute_face_normals()
        self._init_quadrics()
        self._build_priority_queue()

    @staticmethod
    def _ek(v1, v2):
        """Canonical edge key (sorted pair)."""
        return (min(v1, v2), max(v1, v2))

    # ------------------------------------------------------------------ #
    #  Adjacency helpers                                                   #
    # ------------------------------------------------------------------ #

    def _build_adjacency(self):
        for fi, face in enumerate(self.faces):
            for v in face:
                self.vertex_faces[v].add(fi)

    def _get_neighbors(self, v):
        nbrs = set()
        for fi in self.vertex_faces[v]:
            if fi in self.deleted_faces:
                continue
            for fv in self.faces[fi]:
                if fv != v:
                    nbrs.add(fv)
        return nbrs

    def _get_edge_faces(self, v1, v2):
        """Active faces that contain both v1 and v2."""
        result = set()
        for fi in self.vertex_faces[v1]:
            if fi in self.deleted_faces:
                continue
            if v2 in self.faces[fi]:
                result.add(fi)
        return result

    def _get_edges_of_vertex(self, v):
        edges = set()
        for fi in self.vertex_faces[v]:
            if fi in self.deleted_faces:
                continue
            face = self.faces[fi]
            for i in range(3):
                edges.add(self._ek(face[i], face[(i + 1) % 3]))
        return edges

    # ------------------------------------------------------------------ #
    #  Geometry / quadrics                                                 #
    # ------------------------------------------------------------------ #

    def _compute_face_normal(self, fi):
        v0, v1, v2 = self.faces[fi]
        e1 = self.vertices[v1] - self.vertices[v0]
        e2 = self.vertices[v2] - self.vertices[v0]
        n = np.cross(e1, e2)
        norm = np.linalg.norm(n)
        return n / norm if norm > 1e-12 else np.zeros(3)

    def _compute_face_normals(self):
        for fi in range(len(self.faces)):
            self.face_normals[fi] = self._compute_face_normal(fi)

    def _face_quadric(self, fi):
        n = self.face_normals[fi]
        d = -n.dot(self.vertices[self.faces[fi][0]])
        nd = np.array([n[0], n[1], n[2], d])
        return np.outer(nd, nd)

    def _init_quadrics(self):
        for fi in range(len(self.faces)):
            self.face_quadrics[fi] = self._face_quadric(fi)
        for vi in range(len(self.vertices)):
            Q = np.zeros((4, 4))
            for fi in self.vertex_faces[vi]:
                Q += self.face_quadrics[fi]
            self.vertex_quadrics[vi] = Q
        print("Initialized vertex quadrics successfully.")

    def _compute_edge_cost(self, v1, v2):
        Q = self.vertex_quadrics[v1] + self.vertex_quadrics[v2]
        A = Q[:3, :3]
        b = Q[:3, 3]
        c = Q[3, 3]

        def _cost(p):
            return float(p @ A @ p + 2.0 * b @ p + c)

        p1, p2 = self.vertices[v1], self.vertices[v2]
        p_mid = 0.5 * (p1 + p2)

        # Tikhonov regularized solve: (A + λI)x = -b
        max_diag = max(A[0, 0], A[1, 1], A[2, 2])
        lam = 1e-6 * max(max_diag, 1e-12)
        x = np.linalg.solve(A + lam * np.eye(3), -b)

        # Fallback: if regularized solution is worse than v1, v2, or midpoint,
        # pick the candidate with the lowest cost.
        candidates = [(x, _cost(x)), (p1, _cost(p1)), (p2, _cost(p2)), (p_mid, _cost(p_mid))]
        x, cost = min(candidates, key=lambda t: t[1])

        ek = self._ek(v1, v2)
        self.edge_best_pos[ek] = x
        return cost

    # ------------------------------------------------------------------ #
    #  Priority queue                                                      #
    # ------------------------------------------------------------------ #

    def _push(self, ek, cost):
        self.counter += 1
        heapq.heappush(self.heap, (cost, self.counter, ek))
        self.valid_edges.add(ek)

    def _pop_min(self):
        while self.heap:
            cost, _, ek = heapq.heappop(self.heap)
            if ek in self.valid_edges:
                self.valid_edges.discard(ek)
                return cost, ek
        return None, None

    def _invalidate(self, ek):
        self.valid_edges.discard(ek)

    def _build_priority_queue(self):
        visited = set()
        for face in self.faces:
            for i in range(3):
                ek = self._ek(face[i], face[(i + 1) % 3])
                if ek not in visited:
                    visited.add(ek)
                    self._push(ek, self._compute_edge_cost(*ek))
        print("Build priority queue successfully.")

    # ------------------------------------------------------------------ #
    #  Collapse validity                                                   #
    # ------------------------------------------------------------------ #

    def _is_collapse_ok(self, v_keep, v_remove):
        """Link-condition check: common neighbours must equal the third
        vertices of the faces sharing the edge."""
        n_keep = self._get_neighbors(v_keep) - {v_remove}
        n_remove = self._get_neighbors(v_remove) - {v_keep}
        common = n_keep & n_remove

        edge_faces = self._get_edge_faces(v_keep, v_remove)
        third = set()
        for fi in edge_faces:
            for v in self.faces[fi]:
                if v != v_keep and v != v_remove:
                    third.add(v)

        if common != third:
            return False

        # No resulting face should have duplicate vertices
        for fi in self.vertex_faces[v_remove]:
            if fi in self.deleted_faces or fi in edge_faces:
                continue
            new_face = [v_keep if v == v_remove else v for v in self.faces[fi]]
            if len(set(new_face)) < 3:
                return False

        return True

    # ------------------------------------------------------------------ #
    #  Edge collapse                                                       #
    # ------------------------------------------------------------------ #

    def _collapse_edge(self, v1, v2):
        ek = self._ek(v1, v2)
        best_pos = self.edge_best_pos[ek]

        if self._is_collapse_ok(v1, v2):
            v_keep, v_remove = v1, v2
        elif self._is_collapse_ok(v2, v1):
            v_keep, v_remove = v2, v1
        else:
            return 0

        self._invalidate(ek)

        # Invalidate all edges incident to either endpoint
        for e in self._get_edges_of_vertex(v_remove) | self._get_edges_of_vertex(v_keep):
            self._invalidate(e)

        # Faces that share the edge — will be deleted
        edge_faces = self._get_edge_faces(v_keep, v_remove)

        # Move surviving vertex to optimal position
        self.vertices[v_keep] = best_pos

        # Delete shared faces
        removed = 0
        for fi in edge_faces:
            self.deleted_faces.add(fi)
            removed += 1
            for v in self.faces[fi]:
                self.vertex_faces[v].discard(fi)

        # Re-wire: replace v_remove with v_keep in remaining faces
        for fi in list(self.vertex_faces[v_remove]):
            if fi in self.deleted_faces:
                continue
            self.faces[fi] = [v_keep if v == v_remove else v for v in self.faces[fi]]
            self.vertex_faces[v_keep].add(fi)
            self.vertex_faces[v_remove].discard(fi)

        self.vertex_faces[v_remove].clear()
        self.deleted_vertices.add(v_remove)

        # --- Update normals & quadrics for affected faces --- #
        affected_faces = {
            fi for fi in self.vertex_faces[v_keep] if fi not in self.deleted_faces
        }
        for fi in affected_faces:
            self.face_normals[fi] = self._compute_face_normal(fi)
            self.face_quadrics[fi] = self._face_quadric(fi)

        # --- Update vertex quadrics --- #
        Q_combined = self.vertex_quadrics[v1] + self.vertex_quadrics[v2]
        self.vertex_quadrics[v_keep] = Q_combined

        affected_verts = set()
        for fi in affected_faces:
            affected_verts.update(self.faces[fi])

        for vi in affected_verts:
            if vi == v_keep:
                continue
            Q = np.zeros((4, 4))
            for fi in self.vertex_faces[vi]:
                if fi not in self.deleted_faces:
                    Q += self.face_quadrics[fi]
            self.vertex_quadrics[vi] = Q

        # --- Re-insert affected edges into the priority queue --- #
        visited = set()
        for vi in affected_verts:
            for fi in self.vertex_faces[vi]:
                if fi in self.deleted_faces:
                    continue
                face = self.faces[fi]
                for i in range(3):
                    e = self._ek(face[i], face[(i + 1) % 3])
                    if e in visited:
                        continue
                    visited.add(e)
                    a, b = e
                    if a in self.deleted_vertices or b in self.deleted_vertices:
                        continue
                    self._push(e, self._compute_edge_cost(a, b))

        return removed

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    @property
    def active_face_count(self):
        return len(self.faces) - len(self.deleted_faces)

    def simplify(self, target_face_count):
        current = self.active_face_count
        print(f"Simplifying mesh to {target_face_count} faces...")
        print("It takes some time, please wait...")

        while current > target_face_count and self.heap:
            _, ek = self._pop_min()
            if ek is None:
                break
            v1, v2 = ek
            if v1 in self.deleted_vertices or v2 in self.deleted_vertices:
                continue
            current -= self._collapse_edge(v1, v2)

        print(f"Simplification complete. {current} faces remaining.")


# ====================================================================== #
#  CLI entry point                                                         #
# ====================================================================== #


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mesh simplification using Quadric Error Metrics (QEM).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
        examples:
        python mesh_simplifier.py bunny.obj bunny_lo.obj 0.5
        python mesh_simplifier.py input.obj output.obj 0.3
        """,
    )
    parser.add_argument(
        "--input",
        metavar="INPUT",
        help="Path to the input mesh file (OBJ format)",
    )
    parser.add_argument(
        "--output",
        metavar="OUTPUT",
        help="Path to the output mesh file (OBJ format)",
    )
    parser.add_argument(
        "--scale",
        metavar="SCALE",
        type=float,
        help="Simplification ratio in (0, 1]. "
        "Target face count = original faces * scale.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_file, output_file, scale = args.input, args.output, args.scale

    if scale <= 0 or scale > 1:
        print("Error: scale must be in (0, 1].")
        sys.exit(1)

    start_time = time.time()

    vertices, faces = read_obj(input_file)
    edge_set = {
        (min(f[i], f[(i + 1) % 3]), max(f[i], f[(i + 1) % 3]))
        for f in faces
        for i in range(3)
    }
    print(
        f"Loaded mesh with {len(faces)} faces, {len(edge_set)} edges "
        f"and {len(vertices)} vertices."
    )
    print(f"Simplifying begin with scale: {scale}.")

    simplifier = MeshSimplifier(vertices, faces)
    simplifier.simplify(int(len(faces) * scale))

    # Collect surviving mesh
    vmap = {}
    out_verts = []
    for vi in range(len(simplifier.vertices)):
        if vi not in simplifier.deleted_vertices:
            vmap[vi] = len(out_verts)
            out_verts.append(simplifier.vertices[vi])

    out_faces = []
    for fi in range(len(simplifier.faces)):
        if fi not in simplifier.deleted_faces:
            out_faces.append([vmap[v] for v in simplifier.faces[fi]])

    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    write_obj(output_file, np.array(out_verts), out_faces)

    elapsed = time.time() - start_time
    print(f"Mesh simplified and saved to {output_file}")
    print(f"Program execution time: {elapsed:.4f} seconds.")


if __name__ == "__main__":
    main()

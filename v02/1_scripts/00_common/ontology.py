"""Cell Ontology parsing and tree distance.

Ontology distance is a first-class experimental axis (``CLAUDE.md`` §1), so
this is measurement code, not a utility. Poisoning macrophage -> dendritic cell
and macrophage -> type II pneumocyte are different attacks, and the number that
separates them comes from here.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from pathlib import Path

_MAX_DEPTH = 16
"""Search cap. The Cell Ontology is far shallower than this; the bound exists so
a malformed graph fails fast rather than walking forever."""


@dataclass
class Ontology:
    """Name lookup and undirected ``is_a`` distance over Cell Ontology terms."""

    name_to_ids: dict[str, set[str]] = field(default_factory=dict)
    id_to_name: dict[str, str] = field(default_factory=dict)
    parents: dict[str, set[str]] = field(default_factory=dict)
    adj: dict[str, set[str]] = field(default_factory=dict)

    @classmethod
    def from_obo(cls, path: str | Path) -> Ontology:
        """Parse an OBO file, keeping only non-obsolete ``CL:`` terms.

        Args:
            path: an ``.obo`` file. ``cl.obo`` also imports GO/UBERON/PR terms;
                those are skipped.

        Returns:
            A populated :class:`Ontology`.
        """
        o = cls(parents=collections.defaultdict(set), adj=collections.defaultdict(set))
        cur: dict = {}

        def flush(c: dict) -> None:
            if not c or not c.get("id", "").startswith("CL:") or c.get("obsolete"):
                return
            if "name" in c:
                o.name_to_ids.setdefault(c["name"].lower(), set()).add(c["id"])
                o.id_to_name[c["id"]] = c["name"]
            for s in c.get("synonyms", []):
                o.name_to_ids.setdefault(s.lower(), set()).add(c["id"])
            for p in c.get("is_a", []):
                o.parents[c["id"]].add(p)

        for line in Path(path).read_text().splitlines():
            if line == "[Term]":
                flush(cur)
                cur = {"is_a": [], "synonyms": []}
            elif line.startswith("id: "):
                cur["id"] = line[4:].strip()
            elif line.startswith("name: "):
                cur["name"] = line[6:].strip()
            elif line.startswith("is_obsolete: true"):
                cur["obsolete"] = True
            elif line.startswith("is_a: "):
                # Strip BOTH the "! comment" and any "{modifier}" annotation.
                # cl.obo writes `is_a: CL:0000542 {is_inferred="true"} ! ...`;
                # keeping the modifier makes a phantom node distinct from the
                # real term and inflates every distance through that edge.
                target = line[6:].split("!")[0].split("{")[0].strip()
                if target.startswith("CL:"):
                    cur["is_a"].append(target)
            elif line.startswith("synonym: ") and " EXACT " in line:
                quoted = line.split('"')
                if len(quoted) > 1:
                    cur["synonyms"].append(quoted[1])
        flush(cur)

        for child, ps in o.parents.items():
            for p in ps:
                o.adj[child].add(p)
                o.adj[p].add(child)
        return o

    def to_id(self, name: str) -> str | None:
        """Resolve a cell-type name to its CL id.

        Args:
            name: primary name or exact synonym; case-insensitive.

        Returns:
            The id, or ``None`` if unknown or ambiguous. Ambiguity returns
            ``None`` rather than a guess: silently picking one of two terms
            would attach cells to the wrong point in the tree.
        """
        ids = self.name_to_ids.get(name.lower())
        return next(iter(ids)) if ids and len(ids) == 1 else None

    def distance(self, a: str, b: str) -> int | None:
        """Shortest undirected ``is_a`` path length between two terms.

        Args:
            a: source CL id.
            b: target CL id.

        Returns:
            Hop count, 0 if identical, or ``None`` if unreachable.
        """
        if a == b:
            return 0
        seen, frontier = {a}, [a]
        for d in range(1, _MAX_DEPTH + 1):
            nxt = []
            for node in frontier:
                for m in self.adj.get(node, ()):
                    if m in seen:
                        continue
                    if m == b:
                        return d
                    seen.add(m)
                    nxt.append(m)
            if not nxt:
                return None
            frontier = nxt
        return None

    def is_ancestor_or_descendant(self, a: str, b: str) -> bool:
        """Whether either term lies on the other's ``is_a`` chain.

        Args:
            a: one CL id.
            b: the other.

        Returns:
            True if one generalises the other. Such a pair is not a viable
            attack: calling a T cell a "lymphocyte" is a vaguer truth, not a
            lie.
        """
        for x, y in ((a, b), (b, a)):
            seen, frontier = {x}, [x]
            while frontier:
                nxt = []
                for node in frontier:
                    for p in self.parents.get(node, ()):
                        if p == y:
                            return True
                        if p not in seen:
                            seen.add(p)
                            nxt.append(p)
                frontier = nxt
        return False

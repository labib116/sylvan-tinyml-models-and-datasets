from __future__ import annotations

from dataclasses import dataclass, field


def hamming_distance(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


@dataclass
class _Node:
    hash_value: str
    image_id: str
    children: dict[int, "_Node"] = field(default_factory=dict)


class PerceptualHashIndex:
    """BK-tree index for efficient Hamming-distance near-duplicate lookup."""

    def __init__(self) -> None:
        self.root: _Node | None = None

    def add(self, hash_value: str, image_id: str) -> None:
        if not hash_value:
            return
        node = _Node(hash_value, image_id)
        if self.root is None:
            self.root = node
            return
        current = self.root
        while True:
            distance = hamming_distance(hash_value, current.hash_value)
            child = current.children.get(distance)
            if child is None:
                current.children[distance] = node
                return
            current = child

    def nearest(self, hash_value: str, threshold: int) -> tuple[str, int] | None:
        if not hash_value or self.root is None:
            return None
        best: tuple[str, int] | None = None
        stack = [self.root]
        while stack:
            node = stack.pop()
            distance = hamming_distance(hash_value, node.hash_value)
            if distance <= threshold and (best is None or distance < best[1]):
                best = (node.image_id, distance)
                if distance == 0:
                    return best
            lower, upper = distance - threshold, distance + threshold
            stack.extend(child for edge, child in node.children.items() if lower <= edge <= upper)
        return best


class DuplicateIndex:
    def __init__(self) -> None:
        self.exact: dict[str, str] = {}
        self.perceptual = PerceptualHashIndex()

    def add(self, image_id: str, sha256: str, phash: str) -> None:
        if sha256:
            self.exact.setdefault(sha256, image_id)
        if phash:
            self.perceptual.add(phash, image_id)

    def exact_match(self, sha256: str) -> str | None:
        return self.exact.get(sha256)

    def near_match(self, phash: str, threshold: int) -> tuple[str, int] | None:
        return self.perceptual.nearest(phash, threshold)


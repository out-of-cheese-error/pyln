import typing as ty

import numpy as np

from .paths import Box, Hit, NoHit
from .shape import Shape


class Node:
    def __init__(self, shapes: ty.List[Shape]):
        self.axis = None
        self.point = 0
        self.shapes = shapes
        self.left = None
        self.right = None

    def intersect(
        self, ray_origin: np.ndarray, ray_direction: np.ndarray, tmin, tmax
    ) -> Hit:
        if self.axis is None:
            return self.intersect_shapes(ray_origin, ray_direction)
        tsplit = (self.point - ray_origin[self.axis]) / ray_direction[self.axis]
        left_first = (ray_origin[self.axis] < self.point) or (
            ray_origin[self.axis] == self.point
            and ray_direction[self.axis] <= 0
        )
        if left_first:
            first = self.left
            second = self.right
        else:
            first = self.right
            second = self.left
        if tsplit > tmax or tsplit <= 0:
            return first.intersect(ray_origin, ray_direction, tmin, tmax)
        elif tsplit < tmin:
            return second.intersect(ray_origin, ray_direction, tmin, tmax)
        else:
            h1 = first.intersect(ray_origin, ray_direction, tmin, tsplit)
            if h1.t <= tsplit:
                return h1
            h2 = second.intersect(
                ray_origin, ray_direction, tsplit, min(tmax, h1.t)
            )
            if h1.t <= h2.t:
                return h1
            else:
                return h2

    def intersect_shapes(
        self, ray_origin: np.ndarray, ray_direction: np.ndarray
    ) -> Hit:
        hit = NoHit
        for shape in self.shapes:
            h = shape.intersect(ray_origin, ray_direction)
            if h.t < hit.t:
                hit = h
        return hit

    def partition_score(self, axis: int, point) -> int:
        left, right = 0, 0
        for shape in self.shapes:
            box = shape.bounding_box()
            l, r = box.partition(axis, point)
            if l:
                left += 1
            if r:
                right += 1
        return max(left, right)

    def partition(self, axis: int, point: np.ndarray):
        left = []
        right = []
        for shape in self.shapes:
            box = shape.bounding_box()
            l, r = box.partition(axis, point)
            if l:
                left.append(shape)
            if r:
                right.append(shape)
        return left, right

    def split(self, depth: int):
        if len(self.shapes) < 8:
            return
        values = []
        for shape in self.shapes:
            box = shape.bounding_box()
            values += [box.min, box.max]
        medians = np.median(np.array(values), axis=0)

        best = int(len(self.shapes) * 0.85)
        best_axis = None
        best_point = 0.0

        for i in range(3):
            s = self.partition_score(i, medians[i])
            if s < best:
                best = s
                best_axis = i
                best_point = medians[i]

        if best_axis is None:
            return

        l, r = self.partition(best_axis, best_point)
        self.axis = best_axis
        self.point = best_point
        self.left = Node(l)
        self.right = Node(r)
        self.left.split(depth + 1)
        self.right.split(depth + 1)
        self.shapes = None  # only needed at leaf nodes

    def show_tree(self, level=0):
        s = " " * level + "Node:"

        if self.axis is None:
            s += f"Shapes ({len(self.shapes)})\n"
            for shape in self.shapes:
                if getattr(shape, "show_tree", None):
                    s += shape.show_tree(level + 1)
            return s

        s += f"{self.axis} {self.point:.2f}\n"
        s += self.left.show_tree(level + 1)
        s += self.right.show_tree(level + 1)
        return s


class Tree:
    def __init__(self, box: Box, node: Node):
        self.box: Box = box
        self.root: Node = node

    @staticmethod
    def from_shapes(shapes):
        box = Box.BoxForShapes(shapes)
        node = Node(shapes)
        node.split(0)
        return Tree(box, node)

    def intersect(self, ray_origin: np.ndarray, ray_direction: np.ndarray):
        tmin, tmax = self.box.intersect(ray_origin, ray_direction)
        if tmax < tmin or tmax <= 0:
            return NoHit
        return self.root.intersect(ray_origin, ray_direction, tmin, tmax)

    def show_tree(self, level=0):
        return " " * level + "Tree\n" + self.root.show_tree(level + 1)

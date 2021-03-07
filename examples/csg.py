import numpy as np
import pyln
from pyln import BooleanShape, Op


def main():
    shape = BooleanShape.from_shapes(
        Op.Difference,
        [
            BooleanShape.from_shapes(
                Op.Intersection,
                [pyln.Sphere(), pyln.Cube([-0.8, -0.8, -0.8], [0.8, 0.8, 0.8])],
            ),
            pyln.Cylinder(0.4, -2.0, 2.0),
            pyln.TransformedShape(
                pyln.Cylinder(0.4, -2.0, -2.0),
                pyln.utility.vector_rotate(
                    np.array([1.0, 0.0, 0.0]), np.deg2rad(90)
                ),
            ),
            pyln.TransformedShape(
                pyln.Cylinder(0.4, -2.0, 2.0),
                pyln.utility.vector_rotate(
                    np.array([0.0, 1.0, 0.0]), np.deg2rad(90)
                ),
            ),
        ],
    )
    for i in range(0, 90, 5):
        scene = pyln.Scene()
        matrix = pyln.utility.vector_rotate(
            np.array([0, 0, 1], dtype=np.float64), np.deg2rad(i)
        )
        scene.add(pyln.TransformedShape(shape, matrix))
        scene.add(shape)

        # define camera parameters
        eye = np.array([0, 6, 2], dtype=np.float64)  # camera position
        center = np.array([0, 0, 0], dtype=np.float64)  # camera looks at
        up = np.array([0, 0, 1], dtype=np.float64)  # up direction

        # define rendering parameters
        width = 750  # rendered width
        height = 750  # rendered height
        fovy = 20.0  # vertical field of view, degrees
        znear = 0.1  # near z plane
        zfar = 10.0  # far z plane
        step = 0.01  # how finely to chop the paths for visibility testing

        # compute 2D paths that depict the 3D scene
        paths = scene.render(
            eye, center, up, width, height, fovy, znear, zfar, step
        )
        print(i, len(paths.paths))

        # save results
        if len(paths.paths):
            paths.write_to_svg(f"examples/images/csg_{i}.svg", width, height)


if __name__ == "__main__":
    main()

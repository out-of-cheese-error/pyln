import numpy as np
import pyln


def main():
    pyln.utility.compile_numba()
    mesh = pyln.Mesh.from_obj("examples/suzanne.obj")
    mesh.fit_inside(
        pyln.Box([-1, -1, -1], [1, 1, 1]), np.array([0.5, 0.5, 0.5])
    )
    size = 1024
    for i in range(32):
        plane = pyln.Plane(
            np.array([0, 0, (i / (32 - 1)) * 2 - 1]), np.array([0, 0, 1])
        )
        paths = plane.intersect_mesh(mesh)
        paths = paths.transform(
            np.dot(
                pyln.utility.vector_translate(
                    np.array([size / 2, size / 2, 0])
                ),
                pyln.utility.vector_scale(np.array([size / 2, size / 2, 1])),
            )
        )
        paths.write_to_svg(f"examples/images/slice{i}.svg", size, size)


if __name__ == "__main__":
    main()

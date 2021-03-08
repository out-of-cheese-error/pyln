import pyln
from pyln.utility import Transform


def main():
    pyln.utility.compile_numba()
    mesh = pyln.Mesh.from_obj("examples/suzanne.obj")
    mesh.fit_inside(pyln.Box([-1, -1, -1], [1, 1, 1]), [0.5, 0.5, 0.5])
    size = 1024
    slices = 64
    images = []
    for i in range(slices):
        plane = pyln.Plane(
            [0, 0, (i / (slices - 1)) * 2 - 1],
            [0, 0, 1],
        )
        paths = plane.intersect_mesh(mesh)
        paths = paths.transform(
            pyln.utility.matrix_transform(
                [
                    (Transform.Translate, [size / 2, size / 2, 0]),
                    (Transform.Scale, [size / 2, size / 2, 1]),
                ],
            )
        )
        images.append(paths.to_image(size, size))

    # save results
    images[0].save(
        "examples/images/slicer.gif",
        save_all=True,
        append_images=images[1:],
        optimize=True,
        duration=100,
        loop=0,
    )


if __name__ == "__main__":
    main()

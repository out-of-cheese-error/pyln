import numpy as np
import pyln


def main():
    pyln.utility.compile_numba()
    # create a scene and add a single cube
    blocks = np.genfromtxt("examples/mountain.csv", delimiter=",")
    scene = pyln.Scene()
    size = np.zeros(3)
    size[:] = 0.5
    for v in blocks:
        scene.add(pyln.Cube(v - size, v + size))

    # define camera parameters
    eye = [90, -90, 70]  # camera position
    center = [0, 0, -15]  # camera looks at
    up = [0, 0, 1]  # up direction

    # define rendering parameters
    width = 1920  # rendered width
    height = 1080  # rendered height
    fovy = 50.0  # vertical field of view, degrees
    znear = 0.1  # near z plane
    zfar = 1000.0  # far z plane
    step = 0.1  # how finely to chop the paths for visibility testing

    # compute 2D paths that depict the 3D scene
    paths = scene.render(
        eye, center, up, width, height, fovy, znear, zfar, step
    )

    # save results
    paths.write_to_svg(
        "examples/images/mountain.svg", width, height, background_color="white"
    )


if __name__ == "__main__":
    main()

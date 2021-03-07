import numpy as np
import pyln


def main():
    # create a scene and add a single cube
    scene = pyln.Scene()
    scene.add(pyln.Cube([-1, -1, -1], [1, 1, 1]))

    # define camera parameters
    eye = np.array([4, 3, 2], dtype=np.float64)  # camera position
    center = np.array([0, 0, 0], dtype=np.float64)  # camera looks at
    up = np.array([0, 0, 1], dtype=np.float64)  # up direction

    # define rendering parameters
    width = 1024  # rendered width
    height = 1024  # rendered height
    fovy = 50.0  # vertical field of view, degrees
    znear = 0.1  # near z plane
    zfar = 10.0  # far z plane
    step = 0.01  # how finely to chop the paths for visibility testing

    # compute 2D paths that depict the 3D scene
    paths = scene.render(
        eye, center, up, width, height, fovy, znear, zfar, step
    )
    print(paths, len(paths.paths))
    # save results
    paths.write_to_svg("out.svg", width, height)
    paths.write_to_png("out.png", width, height)


if __name__ == "__main__":
    main()

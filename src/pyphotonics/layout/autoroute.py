import gdstk
import numpy as np
import nazca as nd
import nazca.geometries as ng
from nazca.interconnects import Interconnect
from tkinter import *
from pyphotonics.layout import utils, gui

tk = Tk()


class WaveguidePath:
    def __init__(self, points, width, r_min):
        self.points = points
        self.width = width
        self.r_min = r_min


def turn_port_route(port, r_min, target_angle, reverse=False):
    """
    Returns a 4-point bend that turns the given port to the desired angle

    Parameters:
        port (3-tuple):
            Port represented by (x, y, angle), where angle is degrees counter-clockwise from the horizontal
        r_min (double):
            The minimum radius for executing the turn
        target_angle (double):
            Angle in degrees counter-clockwise from the horizontal that the port should be turned to
        reverse (bool):
            Treat port as an output, returning the turn leading into the port from the target_angle

    Returns:
        bend (list of 4 tuples):
            4-point bend that turns the given port to the desired angle
    """
    bend_angle = (target_angle - port[2]) % 360
    if bend_angle > 180:
        bend_angle -= 360
    bend_angle_rad = np.radians(bend_angle)
    port_angle_rad = np.radians(port[2])

    # Define the curve to arrive at the target angle using 4 points
    start = utils.get_port_coords(port)

    if np.isclose(bend_angle, 0):
        return [start]

    l1 = np.abs(r_min * np.tan(bend_angle_rad / 4)) + 0.001
    inter1 = np.array([l1 * np.cos(port_angle_rad), l1 * np.sin(port_angle_rad)])

    l2 = 2 * l1
    theta2 = port_angle_rad + bend_angle_rad / 2
    inter2 = inter1 + np.array([l2 * np.cos(theta2), l2 * np.sin(theta2)])

    theta3 = theta2 + bend_angle_rad / 2
    inter3 = inter2 + np.array([l1 * np.cos(theta3), l1 * np.sin(theta3)])

    if reverse:
        return [-inter3 + start, -inter2 + start, -inter1 + start, start]

    return [start, start + inter1, start + inter2, start + inter3]


def direct_route(port1, port2, r_min):
    """
    Returns the a basic two segment route between two ports, adding segments as necessary to account for port angle.

    Parameters:
        inputs (list of 3-tuples):
            List of input ports represented by (x, y, angle), where angle is degrees counter-clockwise from the horizontal

        outputs (list of 3-tuples):
            List of output ports represented by (x, y, angle), where angle is degrees counter-clockwise from the horizontal

        width (double):
            Width of waveguides in GDS units

        r_min (double):
            Minimum bend radius in GDS units

        d_min (doubles):
            Minimum distance between waveguides in GDS units

        output_file (str):
            Path to output GDS file

    Returns:
        paths (list of WaveguidePaths):
            List of WaveguidePaths between their corresponding inputs and outputs
    """
    # Determine the best set of directions for the given ports to be bent to minimize total bend angle
    dirs = utils.get_perpendicular_directions(port1, port2)
    best_dir = dirs[
        np.argmin(
            list(
                map(
                    lambda x: abs((port1[2] - x[0]) % 360)
                    + abs((port2[2] - x[1]) % 360),
                    dirs,
                )
            )
        )
    ]

    # Turn ports to the desired angle
    points1 = turn_port_route(port1, r_min, best_dir[0])
    points2 = turn_port_route(
        port2, r_min, best_dir[1], reverse=True
    )  # Reverse for output

    # Add intermediate point for manhattan routing
    coords1 = points1[-1] if best_dir[0] == 90 or best_dir[0] == -90 else points2[0]
    coords2 = points2[0] if best_dir[0] == 90 or best_dir[0] == -90 else points1[-1]

    return points1 + [np.array([coords1[0], coords2[1]])] + points2


def user_route(
    inputs, outputs, width, r_min, d_min, initial_paths=None, current_gds=None
):
    if not initial_paths:
        initial_paths = get_rough_paths(inputs, outputs, current_gds)
        if initial_paths is None:
            raise TypeError("No initial paths specified")

    final_paths = []
    for i in range(len(inputs)):
        path = initial_paths[i]
        if len(path) <= 2:
            final_paths.append(WaveguidePath(path, width, r_min))
        curr_dir = utils.manhattan_angle(utils.horizontal_angle(path[1] - path[0]))
        manhattan_path = turn_port_route(inputs[i], r_min, np.degrees(curr_dir))
        for j in range(1, len(path) - 1):
            if j == len(path) - 2:
                final_dir = utils.manhattan_angle(
                    utils.horizontal_angle(path[j + 1] - path[j])
                )
                output_turn = turn_port_route(
                    outputs[i], r_min, np.degrees(final_dir), reverse=True
                )
                intermediate = utils.ray_intersect(
                    manhattan_path[-1],
                    curr_dir,
                    output_turn[0],
                    utils.reverse_angle(final_dir),
                )
                if intermediate is None:
                    raise TypeError("Provided path is not a valid Manhattan route")
                manhattan_path.extend([intermediate, *output_turn])
            else:
                manhattan_path.append(
                    utils.ray_project(manhattan_path[-1], curr_dir, path[j])
                )
                curr_dir = utils.manhattan_angle(
                    utils.horizontal_angle(path[j + 1] - path[j])
                )
        final_paths.append(WaveguidePath(manhattan_path, width, r_min))
    return final_paths


def staircase_route(inputs, outputs, width, r_min, d_min, initial_paths=None):
    pass


def get_rough_paths(inputs, outputs, current_gds=None):
    pathing_gui = gui.PathingGUI(tk, inputs, outputs, current_gds)
    tk.mainloop()
    if pathing_gui.autoroute_paths is None:
        print("Autoroute was not called, exiting...")
        return None
    return pathing_gui.autoroute_paths


def autoroute(
    inputs,
    outputs,
    width,
    r_min,
    d_min,
    method="direct",
    current_gds=None,
    initial_paths=None,
):
    """
    Generates a set of paths with waveguides connecting inputs ports to output ports.

    Parameters:
        inputs (list of 3-tuples):
            List of input ports represented by (x, y, angle), where angle is degrees counter-clockwise from the horizontal

        outputs (list of 3-tuples):
            List of output ports represented by (x, y, angle), where angle is degrees counter-clockwise from the horizontal

        width (double):
            Width of waveguides in GDS units

        r_min (double):
            Minimum bend radius in GDS units

        d_min (double):
            Minimum distance between waveguides in GDS units

        method (str):
            Minimum distance between waveguides in GDS units

    Returns:
        paths (list of WaveguidePaths):
            List of WaveguidePaths between their corresponding inputs and outputs
    """

    if len(inputs) != len(outputs):
        raise ValueError("Input and output port arrays must have the same length")

    if r_min <= 0:
        raise ValueError("Minimum radius must be positive")

    if method == "direct":
        paths = []
        for i in range(len(inputs)):
            port1, port2 = inputs[i], outputs[i]
            paths.append(WaveguidePath(direct_route(port1, port2), width, r_min))
        return paths
    elif method == "user":
        return user_route(
            inputs, outputs, width, r_min, d_min, initial_paths, current_gds
        )


def write_paths_to_gds(paths, output_file, layer=0, datatype=0, geometry="flexpath"):
    """
    Generates a set of paths with waveguides connecting inputs ports to output ports.

    Parameters:
        paths (list of WaveguidePaths):
            List of WaveguidePaths to be written

        output_file (str):
            Path to output GDS file

        geometry (str):
            One of "flexpath" or "bend".
                "flexpath" - Each path is written as a single GDS flex path
                "bend" - Each path is written as separate straight and bend polygons

    Returns:
        paths (list of WaveguidePaths):
            List of WaveguidePaths between their corresponding inputs and outputs
    """
    if geometry == "flexpath":
        lib = gdstk.Library()
        cell = lib.new_cell("AUTOROUTE")
        for path in paths:
            fp = gdstk.FlexPath(
                path.points,
                path.width,
                bend_radius=path.r_min,
                simple_path=True,
                layer=layer,
                datatype=datatype,
            )
            cell.add(fp)
        lib.write_gds(output_file)
    elif geometry == "bend":
        for path in paths:
            points = path.points
            N = len(points)
            prev_tangent_len = 0
            for i in range(N - 1):
                segment_len = utils.euclidean_distance(points[i], points[i + 1])
                if i == N - 2:
                    nd.strt(
                        length=segment_len - prev_tangent_len,
                        width=path.width,
                        layer=(layer, datatype),
                    ).put()
                else:
                    angle = utils.path_angle(points[i], points[i + 1], points[i + 2])
                    interior_angle = np.pi - np.abs(angle)
                    new_tangent_len = (
                        0
                        if np.isclose(interior_angle, 0.0)
                        else path.r_min / np.tan(interior_angle / 2)
                    )
                    if i == 0:
                        starting_angle = utils.path_angle(
                            (points[i][0] - 1, points[i][1]), points[i], points[i + 1]
                        )
                        nd.strt(
                            length=segment_len - prev_tangent_len - new_tangent_len,
                            width=path.width,
                            layer=(layer, datatype),
                        ).put(
                            float(points[i][0]),
                            float(points[i][1]),
                            np.degrees(starting_angle),
                        )
                    else:
                        nd.strt(
                            length=segment_len - prev_tangent_len - new_tangent_len,
                            width=path.width,
                            layer=(layer, datatype),
                        ).put()
                    nd.bend(
                        angle=np.degrees(angle),
                        radius=path.r_min,
                        width=path.width,
                        layer=(layer, datatype),
                    ).put()
                    prev_tangent_len = new_tangent_len
        nd.export_gds(filename=output_file)
    elif geometry == "rectilinear":
        lib = gdstk.Library()
        cell = lib.new_cell("AUTOROUTE")
        for path in paths:
            fp = gdstk.FlexPath(path.points, path.width, layer=layer, datatype=datatype)
            cell.add(fp)
        lib.write_gds(output_file)


if __name__ == "__main__":
    paths = autoroute(
        [
            (5429.736, 2832.87, 180),
            (5412.221, 2610.605, 180),
            (5363.33, 2500.071, 180),
            (5355.79, 2290.019, 180),
            (5376.872, 2276.988, 180),
            (5359.089, 2264.991, 180),
            (5372.219, 2253.115, 180),
            (5395.943, 2241.232, 180),
        ],
        [
            (-300, 3163.777, 180),
            (-300, 3153.777, 180),
            (-300, 3143.777, 180),
            (-300, 3125.5, 180),
            (-300, 3098.5, 180),
            (-300, 3071.5, 180),
            (-300, 2528.777, 180),
            (-300, 2508.777, 180),
        ],
        0.8,
        50,
        10,
        method="user",
        # initial_paths=[
        #     [
        #         np.array([5429.736, 2832.87]),
        #         np.array([2002.58084737, 2808.82035253]),
        #         np.array([1989.29847368, 3256.23289862]),
        #         np.array([-300.0, 3163.777]),
        #     ],
        #     [
        #         np.array([5412.221, 2610.605]),
        #         np.array([1874.18456842, 2667.06588249]),
        #         np.array([1852.04727895, 3207.50479954]),
        #         np.array([-300.0, 3153.777]),
        #     ],
        #     [
        #         np.array([5363.33, 2500.071]),
        #         np.array([1626.24692632, 2543.0307212]),
        #         np.array([1599.68217895, 3141.05739171]),
        #         np.array([-300.0, 3143.777]),
        #     ],
        #     [
        #         np.array([5355.79, 2290.019]),
        #         np.array([1334.03470526, 2294.96039862]),
        #         np.array([1307.46995789, 3065.75032949]),
        #         np.array([-300.0, 3125.5]),
        #     ],
        #     [
        #         np.array([5376.872, 2276.988]),
        #         np.array([1072.81468947, 2255.09195392]),
        #         np.array([408.69600526, 3074.60998387]),
        #         np.array([-300.0, 3098.5]),
        #     ],
        #     [
        #         np.array([5359.089, 2264.991]),
        #         np.array([780.60246842, 2228.51299078]),
        #         np.array([27.93462632, 3074.60998387]),
        #         np.array([-300.0, 3071.5]),
        #     ],
        #     [
        #         np.array([5372.219, 2253.115]),
        #         np.array([488.26726243, 2208.70181919]),
        #         np.array([475.96876827, 2547.09139612]),
        #         np.array([-300.0, 2528.777]),
        #     ],
        #     [
        #         np.array([5395.943, 2241.232]),
        #         np.array([306.86447368, 2190.2442059]),
        #         np.array([276.1182383, 2519.40497619]),
        #         np.array([-300.0, 2508.777]),
        #     ],
        # ],
    )
    # paths = autoroute(
    #     [(-4130, -3739, 0), (-4384, -3759, 0)],
    #     [(2036, -2502, 0), (2152, -2497, 180)],
    #     0.8,
    #     50,
    #     10,
    #     method="user",
    #     current_gds="/Users/rohan/Downloads/TOP_ALL_ASML.GDS",
    #     # initial_paths=[
    #     #     [
    #     #         np.array([-4130.0, -3739.0]),
    #     #         np.array([1714.95789474, -3648.85526316]),
    #     #         np.array([1804.2, -2498.01315789]),
    #     #         np.array([2036.0, -2502.0]),
    #     #     ],
    #     #     [
    #     #         np.array([-4384.0, -3759.0]),
    #     #         np.array([2384.27368421, -3748.06578947]),
    #     #         np.array([2389.23157895, -2468.25]),
    #     #         np.array([2152.0, -2497.0]),
    #     #     ],
    #     # ],
    # )
    write_paths_to_gds(
        paths, "examples/autoroute.gds", layer=1, datatype=1, geometry="flexpath"
    )
    write_paths_to_gds(
        paths, "examples/autoroute_bend.gds", layer=1, datatype=1, geometry="bend"
    )

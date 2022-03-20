import gdstk
import numpy as np
import nazca as nd
import nazca.geometries as ng
from nazca.interconnects import Interconnect

class WaveguidePath():
    def __init__(self, points, width, r_min):
        self.points = points
        self.width = width
        self.r_min = r_min

def get_port_coords(port):
    return np.array([port[0], port[1]])

def euclidean_distance(p1, p2):
    return np.linalg.norm(p2 - p1)

def path_angle(p1, p2, p3):
    v1 = p2-p1
    v2 = p3-p2

    acute_angle = np.arccos(np.clip(np.dot(v1, v2)/np.linalg.norm(v1)/np.linalg.norm(v2), -1, 1))
    if np.cross(v1, v2) < 0:
        return -acute_angle
    return acute_angle

def get_perpendicular_directions(port1, port2):
    dx = port2[0] - port1[0]
    dy = port2[1] - port1[1]
    if dx >= 0 and dy >= 0:
        return [(0, 90), (90, 0)]
    if dx < 0 and dy >= 0:
        return [(90, 180), (180, 90)]
    if dx < 0 and dy < 0:
        return [(180, -90), (-90, 180)]
    if dx >= 0 and dy < 0:
        return [(-90, 0), (0, -90)]

def turn_port_route(port, r_min, target_angle, reverse=False):
    bend_angle = (target_angle - port[2]) % 360
    if bend_angle > 180:
        bend_angle = bend_angle - 360
    bend_angle_rad = np.radians(bend_angle)
    port_angle_rad = np.radians(port[2])

    # Define the curve to arrive at the target angle using 4 points
    start = get_port_coords(port)

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
    dirs = get_perpendicular_directions(port1, port2)
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
    points1 = turn_port_route(port1, r_min, best_dir[0])
    points2 = turn_port_route(
            port2, r_min, best_dir[1], reverse=True
            )  # Reverse for output
    coords1 = points1[-1] if best_dir[0] == 90 or best_dir[0] == -90 else points2[0]
    coords2 = points2[0] if best_dir[0] == 90 or best_dir[0] == -90 else points1[-1]
    return points1 + [np.array([coords1[0], coords2[1]])] + points2


def autoroute(inputs, outputs, width, r_min, d_min):
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

        d_min (doubles):
            Minimum distance between waveguides in GDS units

    Returns:
        paths (list of WaveguidePaths):
            List of WaveguidePaths between their corresponding inputs and outputs
    """

    if len(inputs) != len(outputs):
        raise ValueError("Input and output port arrays must have the same length")

    if r_min <= 0:
        raise ValueError("Minimum radius must be positive")

    paths = []
    for i in range(len(inputs)):
        port1, port2 = inputs[i], outputs[i]
        paths.append(WaveguidePath(direct_route(port1, port2, r_min), width, r_min))

    return paths

def write_paths_to_gds(paths, output_file, geometry="flexpath"):
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
            fp = gdstk.FlexPath(path.points, path.width, bend_radius=path.r_min, simple_path=True)
            cell.add(fp)
        lib.write_gds(output_file)
    elif geometry == "bend":
        for path in paths:
            points = path.points
            N = len(points)
            prev_tangent_len = 0
            for i in range(N-1):
                segment_len = euclidean_distance(points[i], points[i+1])
                if i == N-2:
                    nd.strt(length=segment_len - prev_tangent_len, width=path.width).put()
                else:
                    angle = path_angle(points[i], points[i+1], points[i+2])
                    interior_angle = np.pi - np.abs(angle)
                    new_tangent_len = 0 if np.isclose(interior_angle, 0.0) else path.r_min / np.tan(interior_angle/2)
                    if i == 0:
                        starting_angle = path_angle((points[i][0]-1, points[i][1]), points[i], points[i+1])
                        nd.strt(length=segment_len - prev_tangent_len - new_tangent_len, width=path.width).put(float(points[i][0]), float(points[i][1]), np.degrees(starting_angle))
                    else:
                        nd.strt(length=segment_len - prev_tangent_len - new_tangent_len, width=path.width).put()
                    nd.bend(angle=np.degrees(angle), radius=path.r_min, width=path.width).put()
                    prev_tangent_len = new_tangent_len
        nd.export_gds(filename=output_file)
    elif geometry == "rectilinear":
        lib = gdstk.Library()
        cell = lib.new_cell("AUTOROUTE")
        for path in paths:
            fp = gdstk.FlexPath(path.points, path.width)
            cell.add(fp)
        lib.write_gds(output_file)

if __name__ == "__main__":
    paths = autoroute([(5429.736,2832.87,180), (5412.221,2610.605,180),(5363.33,2500.071,180), (5355.79,2290.019,180),
        (5376.872,2276.988,180),
        (5359.089,2264.991,180),
        (5372.219,2253.115,180),
        (5395.943,2241.232,180)],
        [(-300,3163.777,180),
            (-300,3153.777,180),
            (-300,3143.777,180),
            (-300,3125.5,180),
            (-300,3098.5,180),
            (-300,3071.5,180),
            (-300,2528.777,180),
            (-300,2508.777,180)],
        0.8,
        50,
        10,
        )
    write_paths_to_gds(paths, "examples/autoroute.gds", geometry="flexpath")
    write_paths_to_gds(paths, "examples/autoroute_rectilinear.gds", geometry="rectilinear")
    write_paths_to_gds(paths, "examples/autoroute_bend.gds", geometry="bend")

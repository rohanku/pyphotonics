import numpy as np
import bisect
from pyphotonics.layout import routing


def get_port_coords(port):
    """Returns the coordinates as a numpy array of a port represented as an (x, y, angle) 3-tuple"""
    return np.array([port.x, port.y])


def euclidean_distance(p1, p2):
    """Returns the euclidean distance between two points"""
    return np.linalg.norm(p2 - p1)


def normalize_angle(angle):
    """Return angle between -180 and 180 that is equivalent to given angle"""
    angle %= 2.0 * np.pi
    if angle > np.pi:
        angle -= 2.0 * np.pi
    return angle


def relative_angle(v1, v2):
    """Returns the smallest angle between the two vectors"""
    return np.arccos(
        np.clip(np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2), -1, 1)
    )


def path_angle(p1, p2, p3):
    """Returns the bend angle of a 3-point turn"""
    v1 = p2 - p1
    v2 = p3 - p2

    angle = relative_angle(v1, v2)
    if np.cross(v1, v2) < 0:
        return -angle
    return angle


def horizontal_angle(v):
    """Returns angle relative to horizontal"""
    return np.arctan2(v[1], v[0])


def reverse_angle(angle):
    """Flips an angle about the origin"""
    return normalize_angle(angle + np.pi)


def reverse_port(port):
    return routing.Port(
        port.x, port.y, reverse_angle(port.angle), geometry=port.geometry
    )


def angle_close(angle1, angle2):
    """Checks if two angles are close to one another"""
    return np.isclose(0, normalize_angle(angle1 - angle2))


def manhattan_angle(angle):
    """Returns the closest direction parallel to the x or y axes or xy line to the given angle"""
    angle = normalize_angle(angle)
    angles = [np.pi * (i / 4 - 1) for i in range(9)]
    boundary_angles = [
        np.pi * (i // 2 / 2 - 0.75 - 0.05 + 0.1 * (i % 2)) for i in range(8)
    ]
    return angles[bisect.bisect(boundary_angles, angle)]


def perp(v):
    """Rotates given vector by 90 degrees counter-clockwise"""
    b = np.empty_like(v)
    b[0] = -v[1]
    b[1] = v[0]
    return b


def line_intersect(p1, angle1, p2, angle2):
    """Returns the intersection point of two lines, or None if no such point exists"""
    angle1, angle2 = normalize_angle(angle1), normalize_angle(angle2)
    if np.isclose(angle1, angle2):
        return None
    v1 = np.array([np.cos(angle1), np.sin(angle1)])
    v2 = np.array([np.cos(angle2), np.sin(angle2)])
    dp = p1 - p2
    v1_perp = perp(v1)
    return (v1_perp @ dp) / (v1_perp @ v2) * v2 + p2


def ray_intersect(p1, angle1, p2, angle2):
    """Returns the intersection point of two rays, or None if no such point exists"""
    point = line_intersect(p1, angle1, p2, angle2)
    if not angle_close(horizontal_angle(point - p2), angle2):
        return None
    return point


def ray_project(p1, angle, p2):
    """Returns projection of p2 onto ray starting at p1 with the given angle from the horizontal"""
    v1 = np.array([np.cos(angle), np.sin(angle)])
    return v1 @ (p2 - p1) * v1 + p1


def get_perpendicular_directions(port1, port2):
    """Returns the appropriate port directions such that they face each other at a 90 degree angle. The first element gives an x axis direction for port 1."""
    dx = port2.x - port1.x
    dy = port2.y - port1.y
    if dx >= 0 and dy >= 0:
        return [(0, np.pi / 2), (np.pi / 2, 0)]
    if dy >= 0:
        return [(np.pi, np.pi / 2), (np.pi / 2, np.pi)]
    if dx < 0:
        return [(np.pi, -np.pi / 2), (-np.pi / 2, np.pi)]
    return [(0, -np.pi / 2), (-np.pi / 2, 0)]


def get_bounding_box(points, padding=0):
    """Returns the bounding box of a list of points with the given padding"""
    x_coords = list(map(lambda x: x[0], points))
    y_coords = list(map(lambda x: x[1], points))
    return (
        min(x_coords) - padding,
        min(y_coords) - padding,
        max(x_coords) + padding,
        max(y_coords) + padding,
    )


def bbox_overlap(bbox1, bbox2):
    return not (
        bbox1[2] <= bbox2[0]
        or bbox1[3] <= bbox2[1]
        or bbox1[0] >= bbox2[2]
        or bbox1[1] >= bbox2[3]
    )


def in_bbox(bbox, x, y):
    return bbox_overlap(bbox, (x, y, x, y))


def get_port_polygons(ports, l, w):
    """Returns polygon outlines for port markers used in the GUI"""
    polys = []
    for port in ports:
        poly = []
        if port.geometry is None or port.geometry.kind == "slab":
            poly.extend(
                [
                    port.x - w / 2 * np.sin(port.angle),
                    port.y - w / 2 * np.cos(port.angle),
                    port.x + w / 2 * np.sin(port.angle),
                    port.y + w / 2 * np.cos(port.angle),
                    port.x + w / 2 * np.sin(port.angle) + l * np.cos(port.angle),
                    port.y + w / 2 * np.cos(port.angle) - l * np.sin(port.angle),
                    port.x + 1.5 * l * np.cos(port.angle),
                    port.y - 1.5 * l * np.sin(port.angle),
                    port.x - w / 2 * np.sin(port.angle) + l * np.cos(port.angle),
                    port.y - w / 2 * np.cos(port.angle) - l * np.sin(port.angle),
                ]
            )
        elif port.geometry.kind == "ridge":
            poly.extend(
                [
                    port.x - w * np.sin(port.angle),
                    port.y - w * np.cos(port.angle),
                    port.x + w * np.sin(port.angle),
                    port.y + w * np.cos(port.angle),
                    port.x + w * np.sin(port.angle) + l * np.cos(port.angle),
                    port.y + w * np.cos(port.angle) - l * np.sin(port.angle),
                    port.x + w / 2 * np.sin(port.angle) + l * np.cos(port.angle),
                    port.y + w / 2 * np.cos(port.angle) - l * np.sin(port.angle),
                    port.x + 1.5 * l * np.cos(port.angle),
                    port.y - 1.5 * l * np.sin(port.angle),
                    port.x - w / 2 * np.sin(port.angle) + l * np.cos(port.angle),
                    port.y - w / 2 * np.cos(port.angle) - l * np.sin(port.angle),
                    port.x - w * np.sin(port.angle) + l * np.cos(port.angle),
                    port.y - w * np.cos(port.angle) - l * np.sin(port.angle),
                ]
            )
        polys.append(poly)
    return polys

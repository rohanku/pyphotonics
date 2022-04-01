# paths = autoroute(
#     [
#         (5429.736, 2832.87, 180),
#         (5412.221, 2610.605, 180),
#         (5363.33, 2500.071, 180),
#         (5355.79, 2290.019, 180),
#         (5376.872, 2276.988, 180),
#         (5359.089, 2264.991, 180),
#         (5372.219, 2253.115, 180),
#         (5395.943, 2241.232, 180),
#     ],
#     [
#         (-300, 3163.777, 180),
#         (-300, 3153.777, 180),
#         (-300, 3143.777, 180),
#         (-300, 3125.5, 180),
#         (-300, 3098.5, 180),
#         (-300, 3071.5, 180),
#         (-300, 2528.777, 180),
#         (-300, 2508.777, 180),
#     ],
#     0.8,
#     [50],
#     10,
#     method="user"
# )
paths = autoroute(
    [(-4130, -3739, 0), (-4384, -3759, 0)],
    [(2036, -2502, 0), (2152, -2497, 180)],
    0.8,
    [20, 50],
    10,
    method="user",
    route_file="/Users/rohan/Downloads/grating_coupler.route",
    current_gds="/Users/rohan/Downloads/TOP_ALL_ASML.GDS",
)
write_paths_to_gds(paths, "examples/autoroute.gds", layer=1, datatype=1)

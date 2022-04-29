import numpy as np
from pyphotonics.layout.routing import (
    user_route,
    write_paths_to_gds,
    Port,
    WaveguideGeometry,
)
from pyphotonics.config import PATH

inputs = [Port(-11545.453, 2182.5 - 127 * i, 0) for i in range(7)]
outputs = [Port(-9435.453, 2055.5, np.pi), Port(-10695.453, 2055.5, 0)] + [
    Port(-10463.728 + 254 * i, 1826.777, np.pi / 2) for i in range(5)
]

waveguides = user_route(
    WaveguideGeometry(0.8),
    [50, 62.75],
    inputs=inputs,
    outputs=outputs,
    current_gds=PATH.example_autoroute,
)

write_paths_to_gds(waveguides, 'demo.gds' , layer=1111)

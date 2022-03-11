import gdstk


def autoroute(inputs, outputs, width, r_min, d_min, output_file):
    """
    Generates a GDS file with waveguides connecting inputs ports to output ports.

    Parameters:
        inputs (list of 3-tuples):
            List of input ports represented by (x, y, angle), where angle is degrees counter-clockwise from the horizontal

        outputs (list of 3-tuples):
            List of output ports represented by (x, y, angle), where angle is degrees counter-clockwise from the horizontal

        width (double):
            Width of waveguides in meters

        r_min (double):
            Minimum bend radius in meters

        d_min (doubles):
            Minimum distance between waveguides in meters

        output_file (str):
            Path to output GDS file
            
    """

    if len(inputs) != len(outputs):
        raise ValueError("Input and output port arrays must have the same length")

    if r_min <= 0:
        raise ValueError("Minimum radius must be positive")

    lib = gdstk.Library()

    cell = lib.new_cell("AUTOROUTE")

    rect = gdstk.rectangle((0, 0), (2, 1))
    cell.add(rect)

    lib.write_gds("/Users/rohan/Downloads/autoroute.gds")

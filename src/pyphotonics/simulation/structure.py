class SOI:
    def __init__(
        self,
        substrate_s=1.5e-3,
        substrate_t=725e-6,
        box_t=2e-6,
        si_t=220e-9,
        tox_t=1e-6,
    ):
        """
        Initialize an SOI material stack with the given parameters.

        Parameters:
            substrate-s (double)
        """
        # SOI parameters
        self.substrate_s = substrate_s
        self.substrate_t = substrate_t
        self.box_t = box_t
        self.si_t = si_t
        self.tox_t = tox_t

    def setup(self, sim):
        """
        Set up SOI material stack in provided simulation and returns the created layers.

        Parameters:
            sim:
                Lumerical API simulation object

        Returns:
            substrate:
                Lumerical rect object representing silicon substrate

            box:
                Lumerical rect object representing bottom oxide

            si_etch:
                Lumerical rect object representing oxide filling in areas of etched silicon

            tox:
                Lumerical rect object representing top oxide
        """

        # Create substrate
        substrate = sim.addrect(
            name="substrate",
            x=0,
            x_span=self.substrate_s,
            y=0,
            y_span=self.substrate_s,
            z_min=-self.substrate_t - self.box_t,
            z_max=-self.box_t,
            material="Si (Silicon) - Palik",
        )

        # Create BOX
        box = sim.addrect(
            name="BOX",
            x=0,
            x_span=self.substrate_s,
            y=0,
            y_span=self.substrate_s,
            z_min=-self.box_t,
            z_max=0,
            material="SiO2 (Glass) - Palik",
        )

        # Create Si etch layer
        si_etch = sim.addrect(
            name="Si_etch",
            x=0,
            x_span=self.substrate_s,
            y=0,
            y_span=self.substrate_s,
            z_min=0,
            z_max=self.si_t,
            material="SiO2 (Glass) - Palik",
        )

        # Create TOX
        tox = sim.addrect(
            name="TOX",
            x=0,
            x_span=self.substrate_s,
            y=0,
            y_span=self.substrate_s,
            z_min=self.si_t,
            z_max=self.si_t + self.tox_t,
            material="SiO2 (Glass) - Palik",
        )

        return substrate, box, si_etch, tox

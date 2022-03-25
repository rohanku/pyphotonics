import numpy as np
import os
from pyphotonics.layout import utils
import random
import string
from tkinter import *
from tkinter import ttk, messagebox
from PIL import Image, ImageTk


class PathingGUI(ttk.Frame):
    def __init__(self, mainframe, inputs, outputs, current_gds=None, padding=200):
        # Find dimensions for routing frame in GDS and Tkinter canvas coordinates
        self.gds_bbox = utils.get_bounding_box(
            list(map(utils.get_port_coords, inputs))
            + list(map(utils.get_port_coords, outputs)),
            padding=500,
        )
        aspect_ratio = (self.gds_bbox[3] - self.gds_bbox[1]) / (
            self.gds_bbox[2] - self.gds_bbox[0]
        )
        if aspect_ratio <= 1920 / 1080:
            w = mainframe.winfo_screenwidth() - 2 * padding
            h = int(0.5 + w * aspect_ratio)
        else:
            h = mainframe.winfo_screenheight() - 2 * padding
            w = int(0.5 + h / aspect_ratio)

        path = None
        self.image = None
        if current_gds:
            # Retrieve screenshot of the provided GDS file
            tag = "".join(random.choice(string.ascii_letters) for i in range(10))
            path = f"/tmp/pyphotonics/{tag}.png"

            gds_to_png_abs_path = os.path.join(
                os.path.dirname(__file__), "../klayout/gds_to_png.py"
            )
            os.system(
                f"klayout -r {gds_to_png_abs_path} -rd current_gds={current_gds} -rd output_path={path} -rd bx1={self.gds_bbox[0]} -rd by1={self.gds_bbox[1]} -rd bx2={self.gds_bbox[2]} -rd by2={self.gds_bbox[3]} -rd w={w} -rd h={h}"
            )
            self.image = Image.open(path)

        ttk.Frame.__init__(self, master=mainframe)
        self.master.title("Pyphotonics Autoroute Selector")

        # Define relevant parameters
        self.width, self.height = w, h
        self.imscale = 1.0  # Scale for the canvas image
        self.delta = 1.2  # Zoom speed
        self.port_length = 6  # Length of port markers
        self.port_width = 6  # Width of port markers
        self.linewidth = 1  # Display line width of routes
        self.selected_color = "#1eff00"  # Color of selected ports
        self.deselected_color = "#00c8ff"  # Color of deselected ports
        self.N = len(inputs)  # Number of paths
        self.autoroute_paths = (
            None  # Determines whether program was quit or autoroute was run
        )

        # Menu bar and shortcuts
        menu_bar = Menu(self.master)

        # Basic file operations (e.g. save, open, close)
        file_menu = Menu(menu_bar, tearoff=0)
        file_menu.add_command(
            label="Close", command=self.master.quit, accelerator="Cmd+W"
        )
        file_menu.add_command(label="Save", command=self.save, accelerator="Cmd+S")
        self.master.bind_all("<Command-w>", lambda x: self.master.quit())
        self.master.bind_all("<Command-S>", self.save)
        menu_bar.add_cascade(label="File", menu=file_menu)

        # Basic edit operations (e.g. select path, undo, redo)
        # TODO: Add bindings and commands for menu items
        edit_menu = Menu(menu_bar, tearoff=0)
        edit_menu.add_command(label="Next Path", accelerator="D")
        edit_menu.add_command(label="Previous Path", accelerator="A")
        edit_menu.add_command(label="Undo", accelerator="Cmd-Z")
        edit_menu.add_command(label="Redo", accelerator="Cmd-Shift-Z")
        menu_bar.add_cascade(label="Edit", menu=edit_menu)

        self.master.config(menu=menu_bar)

        # Set up Tkinter canvas
        self.canvas = Canvas(self.master, width=w, height=h, highlightthickness=0)
        self.canvas.grid(row=1, column=0, sticky="nswe")
        self.canvas.update()

        # Make canvas expandable and allow for negative coordinates
        self.canvas.configure(scrollregion=self.canvas.bbox("ALL"))
        self.master.rowconfigure(0, weight=1)
        self.master.columnconfigure(0, weight=1)

        # Bind events to the Canvas
        self.canvas.bind("<Configure>", self.show_image)
        self.canvas.bind("<ButtonPress-3>", self.move_from)
        self.canvas.bind("<B3-Motion>", self.move_to)
        self.canvas.bind("<MouseWheel>", self.wheel)  # Scroll for Windows and MacOS
        self.canvas.bind("<Button-5>", self.wheel)  # Scroll down for Linux
        self.canvas.bind("<Button-4>", self.wheel)  # Scroll up for Linux

        # Put image into container rectangle and use it to set proper coordinates to the image
        self.container = self.canvas.create_rectangle(0, 0, w, h, width=0)

        # Convert input and output ports to PNG coordinates and draw their respective markers
        self.png_inputs = self.get_png_ports(inputs)
        self.input_rects = list(
            map(
                lambda x: self.canvas.create_polygon(*x, fill=self.deselected_color),
                utils.get_port_polygons(
                    self.png_inputs, self.port_length, self.port_width
                ),
            )
        )

        self.png_outputs = self.get_png_ports(outputs)
        self.output_rects = list(
            map(
                lambda x: self.canvas.create_polygon(*x, fill=self.deselected_color),
                utils.get_port_polygons(
                    self.png_outputs, -self.port_length, self.port_width
                ),
            )
        )

        # Color the selected ports appropriately
        self.canvas.itemconfig(self.input_rects[0], fill=self.selected_color)
        self.canvas.itemconfig(self.output_rects[0], fill=self.selected_color)

        # Toolbar
        toolbar = Frame(self.master, relief=RAISED)

        # Path selection variables
        self.selected_path_index = (
            0  # Index of selected option for addressing various arrays
        )
        self.current_paths = [
            [utils.get_port_coords(self.png_inputs[i])] for i in range(self.N)
        ]  # List of current coordinates for each path, defaults to just those of the input port
        self.path_lines = [
            None
        ] * self.N  # Tkinter Canvas object IDs of lines representing current paths
        self.path_terminated = [
            False
        ] * self.N  # Whether the current path connects the input and output ports, determines if user should be in drawing mode
        self.cursor_line = self.canvas.create_line(
            *self.get_zoom_coords(self.current_paths[self.selected_path_index][-1]),
            *self.get_zoom_coords(self.current_paths[self.selected_path_index][-1]),
            width=self.linewidth,
        )  # Line from last placed point to cursor

        # Path selection widgets
        self.paths = [
            f"Path {i}" for i in range(self.N)
        ]  # String options for Tkinter OptionMenu
        self.selected_path = StringVar(
            self.master, self.paths[self.selected_path_index]
        )  # String selected by OptionMenu
        path_select = OptionMenu(
            toolbar, self.selected_path, *self.paths, command=self.path_selected
        )  # OptionMenu for selecting paths
        redraw_button = Button(toolbar, text="Redraw", command=self.clear)
        clear_all_button = Button(toolbar, text="Clear All", command=self.clear_all)
        clear_all_button = Button(toolbar, text="Autoroute", command=self.autoroute)

        # Toolbar UI layout
        toolbar.grid(row=0, column=0, sticky="nwe")
        path_select.grid(row=0, column=0)
        redraw_button.grid(row=0, column=1)
        clear_all_button.grid(row=0, column=2)

        # Bind mouse movement and clicking for route placement
        self.canvas.bind("<Motion>", self.motion)
        self.canvas.bind("<ButtonRelease-1>", self.click)

        self.show_image()

    def move_from(self, event):
        """Remember previous coordinates for scrolling with the mouse"""
        self.canvas.scan_mark(event.x, event.y)

    def move_to(self, event):
        """Drag (move) canvas to the new position"""
        self.canvas.scan_dragto(event.x, event.y, gain=1)
        self.show_image()  # redraw the image

    def wheel(self, event):
        """Zoom with mouse wheel"""
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        bbox = self.canvas.bbox(self.container)  # Get image area

        # Allow zoom only if mouse is within the image
        if bbox[0] < x < bbox[2] and bbox[1] < y < bbox[3]:
            pass
        else:
            return

        scale = 1.0
        # Respond to Linux (event.num) or Windows/Mac (event.delta) wheel event
        if event.num == 5 or event.delta < 0:  # Scroll down
            i = min(self.width, self.height)
            if int(i * self.imscale) < 30:
                return  # Image is less than 30 pixels
            self.imscale /= self.delta
            scale /= self.delta
        elif event.num == 4 or event.delta > 0:  # Scroll up
            i = min(self.canvas.winfo_width(), self.canvas.winfo_height())
            if i < self.imscale:
                return  # 1 pixel is bigger than the visible area
            self.imscale *= self.delta
            scale *= self.delta
        self.canvas.scale("all", x, y, scale, scale)  # Rescale all canvas objects
        self.show_image()

    def show_image(self, event=None):
        """Show image and other relevant geometry on the Canvas"""
        # Draw all paths, including the one being currently edited
        for i in range(self.N):
            if len(self.current_paths[i]) > 1:
                if self.path_lines[i]:
                    self.canvas.coords(
                        self.path_lines[i],
                        *np.ndarray.flatten(
                            np.array(
                                list(map(self.get_zoom_coords, self.current_paths[i]))
                            )
                        ),
                    )
                else:
                    self.path_lines[i] = self.canvas.create_line(
                        *np.ndarray.flatten(
                            np.array(
                                list(map(self.get_zoom_coords, self.current_paths[i]))
                            )
                        ),
                        width=self.linewidth,
                    )

        bbox1 = self.canvas.bbox(self.container)  # Get image area

        # Remove 1 pixel shift at the sides of the bbox1
        bbox1 = (bbox1[0] + 1, bbox1[1] + 1, bbox1[2] - 1, bbox1[3] - 1)
        bbox2 = (
            self.canvas.canvasx(0),
            self.canvas.canvasy(0),
            self.canvas.canvasx(self.canvas.winfo_width()),
            self.canvas.canvasy(self.canvas.winfo_height()),
        )  # Get visible area of the canvas
        bbox = [
            min(bbox1[0] - self.width, bbox2[0]),
            min(bbox1[1] - self.height, bbox2[1]),
            max(bbox1[2] + self.width, bbox2[2]),
            max(bbox1[3] + self.height, bbox2[3]),
        ]  # Get scroll region box
        if (
            bbox[0] == bbox2[0] and bbox[2] == bbox2[2]
        ):  # Whole image in the visible area
            bbox[0] = bbox1[0]
            bbox[2] = bbox1[2]
        if (
            bbox[1] == bbox2[1] and bbox[3] == bbox2[3]
        ):  # Whole image in the visible area
            bbox[1] = bbox1[1]
            bbox[3] = bbox1[3]
        x1 = max(
            bbox2[0] - bbox1[0], 0
        )  # Get coordinates (x1,y1,x2,y2) of the image tile
        y1 = max(bbox2[1] - bbox1[1], 0)
        x2 = min(bbox2[2], bbox1[2]) - bbox1[0]
        y2 = min(bbox2[3], bbox1[3]) - bbox1[1]
        if self.image and (
            int(x2 - x1) > 0 and int(y2 - y1) > 0
        ):  # Show image if it in the visible area
            x = min(int(x2 / self.imscale), self.width)
            y = min(int(y2 / self.imscale), self.height)
            image = self.image.crop(
                (int(x1 / self.imscale), int(y1 / self.imscale), x, y)
            )
            imagetk = ImageTk.PhotoImage(image.resize((int(x2 - x1), int(y2 - y1))))
            imageid = self.canvas.create_image(
                max(bbox2[0], bbox1[0]),
                max(bbox2[1], bbox1[1]),
                anchor="nw",
                image=imagetk,
            )
            self.canvas.lower(imageid)  # Set image into background
            self.canvas.imagetk = (
                imagetk  # Keep an extra reference to prevent garbage-collection
            )

    def path_selected(self, *args):
        """Handler for changing the selected path"""
        # Clear paths that have not been completed
        if not self.path_terminated[self.selected_path_index]:
            self.clear_by_index(self.selected_path_index)

        # Set the appropriate marker colors and update the current index
        for i in range(self.N):
            if self.paths[i] == self.selected_path.get():
                self.selected_path_index = i
                self.canvas.itemconfig(self.input_rects[i], fill=self.selected_color)
                self.canvas.itemconfig(self.output_rects[i], fill=self.selected_color)
            else:
                self.canvas.itemconfig(self.input_rects[i], fill=self.deselected_color)
                self.canvas.itemconfig(self.output_rects[i], fill=self.deselected_color)

    def motion(self, event):
        """Handler for mouse movement"""
        # If the path is not terminated, show the drawing cursor line
        if not self.path_terminated[self.selected_path_index]:
            event_coords = np.array([event.x, event.y])
            coords = self.get_zoom_coords(
                self.current_paths[self.selected_path_index][-1]
            )
            self.canvas.itemconfigure(self.cursor_line, state="normal")
            self.canvas.coords(
                self.cursor_line, *coords, *self.get_canvas_coords(event_coords)
            )
        else:
            self.canvas.itemconfigure(self.cursor_line, state="hidden")

    def click(self, event):
        """Handler for mouse click"""
        # If the path is not terminated, add to the current path
        if not self.path_terminated[self.selected_path_index]:
            event_coords = np.array([event.x, event.y])
            coords1 = self.get_canvas_coords(event_coords)
            coords2 = self.get_zoom_coords(
                self.current_paths[self.selected_path_index][-1]
            )
            overlapping_objects = self.canvas.find_overlapping(*coords1, *coords2)
            # If the new endpoint causes the path to enter the output port, terminate the path
            if self.output_rects[self.selected_path_index] in overlapping_objects:
                self.path_terminated[self.selected_path_index] = True
                self.current_paths[self.selected_path_index].append(
                    utils.get_port_coords(self.png_outputs[self.selected_path_index])
                )
            else:
                self.current_paths[self.selected_path_index].append(
                    self.get_real_coords(event_coords)
                )
        self.show_image()

    def clear_by_index(self, index):
        """Clear the current path for the given index"""
        self.current_paths[index] = [
            utils.get_port_coords(self.png_inputs[self.selected_path_index])
        ]
        self.canvas.delete(self.path_lines[index])
        self.path_lines[index] = None
        self.path_terminated[index] = False

    def clear(self):
        """Clear the currently selected path"""
        self.clear_by_index(self.selected_path_index)

    def clear_all(self):
        """Clear all paths"""
        for i in range(self.N):
            self.clear_by_index(i)

    def save(self):
        pass

    def autoroute(self):
        if not all(self.path_terminated):
            messagebox.showerror("error", "Not all paths have been specified!")
            return
        self.autoroute_paths = list(
            map(lambda x: list(map(self.get_gds_coords, x)), self.current_paths)
        )
        self.master.quit()

    def get_zoom_coords(self, coords):
        """Convert 100% zoom canvas coordinates to canvas coordinates for the current zoom"""
        bbox1 = self.canvas.bbox(self.container)
        return np.array(
            [coords[0] * self.imscale + bbox1[0], coords[1] * self.imscale + bbox1[1]]
        )

    def get_canvas_coords(self, coords):
        """Convert from absolute screen coordinates to zoomed canvas coordinates"""
        return np.array(
            [self.canvas.canvasx(coords[0]), self.canvas.canvasy(coords[1])]
        )

    def get_real_coords(self, coords):
        """Convert from absolute screen coordinates to 100% zoom canvas coordinates"""
        bbox1 = self.canvas.bbox(self.container)
        return np.array(
            [
                (self.canvas.canvasx(coords[0]) - bbox1[0]) / self.imscale,
                (self.canvas.canvasy(coords[1]) - bbox1[1]) / self.imscale,
            ]
        )

    def get_png_coords(self, coords):
        """Convert from GDS coordinates to coordinates on the canvas"""
        x = (
            self.width
            * (coords[0] - self.gds_bbox[0])
            / (self.gds_bbox[2] - self.gds_bbox[0])
        )
        y = self.height * (
            1 - (coords[1] - self.gds_bbox[1]) / (self.gds_bbox[3] - self.gds_bbox[1])
        )
        return np.array([x, y])

    def get_png_port(self, port):
        """Convert a single port's GDS coordinates to coordinates on the canvas"""
        png_coords = self.get_png_coords(utils.get_port_coords(port))
        return (png_coords[0], png_coords[1], port[2])

    def get_png_ports(self, ports):
        """Convert port GDS coordinates to coordinates on the canvas"""
        return list(map(self.get_png_port, ports))

    def get_gds_coords(self, coords):
        """Convert from coordinates on the canvas to GDS coordinates"""
        x = (
            coords[0] * (self.gds_bbox[2] - self.gds_bbox[0]) / self.width
            + self.gds_bbox[0]
        )
        y = (1 - coords[1] / self.height) * (
            self.gds_bbox[3] - self.gds_bbox[1]
        ) + self.gds_bbox[1]
        return np.array([x, y])

import numpy as np
import os
from pyphotonics.layout import utils, routing
import random
import string
from tkinter import *
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk


class PathingGUI(ttk.Frame):
    def __init__(
        self, mainframe, inputs, outputs, gds_bbox, potential_ports, current_gds=None
    ):
        # Find dimensions for routing frame in GDS and Tkinter canvas coordinates
        self.gds_bbox = gds_bbox
        padding = 200
        aspect_ratio = (self.gds_bbox[3] - self.gds_bbox[1]) / (
            self.gds_bbox[2] - self.gds_bbox[0]
        )
        if (
            aspect_ratio
            <= mainframe.winfo_screenwidth() / mainframe.winfo_screenheight()
        ):
            w = mainframe.winfo_screenwidth() - 2 * padding
            h = int(0.5 + w * aspect_ratio)
        else:
            h = mainframe.winfo_screenheight() - 2 * padding
            w = int(0.5 + h / aspect_ratio)

        self.image = None
        if current_gds:
            # Retrieve screenshot of the provided GDS file
            tag = "".join(random.choice(string.ascii_letters) for _ in range(10))
            os.makedirs("/tmp/pyphotonics/autoroute", exist_ok=True)
            path = f"/tmp/pyphotonics/autoroute/{tag}.png"

            gds_to_png_abs_path = os.path.join(
                os.path.dirname(__file__), "../klayout/gds_to_png.py"
            )
            rc = os.system(
                f"klayout -r {gds_to_png_abs_path} -rd current_gds={current_gds} -rd output_path={path} -rd bx1={self.gds_bbox[0]} -rd by1={self.gds_bbox[1]} -rd bx2={self.gds_bbox[2]} -rd by2={self.gds_bbox[3]} -rd w={w} -rd h={h}"
            )
            if rc == 0:
                try:
                    self.image = Image.open(path)
                except:
                    print(
                        "Failed to load image of GDS, continuing without background image"
                    )
            else:
                print(
                    "Failed to run klayout binary, continuing without backgroudn image. Make sure klayout has been added to path in order to view an image of the GDS in the waveguide editor"
                )

        ttk.Frame.__init__(self, master=mainframe)
        self.master.title("Pyphotonics Autoroute Selector")

        # Define relevant parameters
        self.width, self.height = w, h  # Width and height of Tkinter canvas
        self.imscale = 1.0  # Scale for the canvas image
        self.delta = 1.2  # Zoom speed
        self.port_length = 6  # Length of port markers
        self.port_width = 6  # Width of port markers
        self.linewidth = 1  # Display line width of routes
        self.selected_color = "#1eff00"  # Color of selected ports
        self.deselected_color = "#00c8ff"  # Color of deselected ports
        self.potential_color = "#ff006f"  # Color of deselected ports
        self.N = len(inputs)  # Number of paths
        self.inputs = inputs  # List of input ports
        self.outputs = outputs  # List of output ports
        self.potential_ports = potential_ports  # List of potential ports
        self.autoroute_paths = (
            None  # Determines whether program was quit or autoroute was run
        )

        # Menu bar and shortcuts
        menu_bar = Menu(self.master)

        # Basic file operations (e.g. save, open, close)
        self.current_file = None
        file_menu = Menu(menu_bar, tearoff=0)
        # TODO: Fix saving and restoring paths
        # file_menu.add_command(label="Open", command=self.open, accelerator="Cmd+O")
        file_menu.add_command(
            label="Close", command=self.master.quit, accelerator="Cmd+W"
        )
        # file_menu.add_command(label="Save", command=self.save, accelerator="Cmd+S")
        # file_menu.add_command(
        #     label="Save As", command=self.save_as, accelerator="Cmd+Shift+S"
        # )
        # self.master.bind_all("<Command-o>", self.open)
        # self.master.bind_all("<Command-O>", self.open)
        self.master.bind_all("<Command-w>", lambda x: self.master.quit())
        self.master.bind_all("<Command-W>", lambda x: self.master.quit())
        # self.master.bind_all("<Command-s>", self.save)
        # self.master.bind_all("<Command-S>", self.save)
        # self.master.bind_all("<Command-Shift-s>", self.save_as)
        # self.master.bind_all("<Command-Shift-S>", self.save_as)
        menu_bar.add_cascade(label="File", menu=file_menu)

        # Basic edit operations (e.g. select path, undo, redo)
        edit_menu = Menu(menu_bar, tearoff=0)
        edit_menu.add_command(
            label="Next Path", command=self.next_path, accelerator="E"
        )
        edit_menu.add_command(
            label="Previous Path", command=self.prev_path, accelerator="Q"
        )
        edit_menu.add_command(
            label="Add Path", command=self.add_path, accelerator="Shift+A"
        )
        edit_menu.add_command(label="Redraw", command=self.clear, accelerator="Cmd+D")
        edit_menu.add_command(
            label="Clear All", command=self.clear_all, accelerator="Cmd+L"
        )
        edit_menu.add_command(
            label="Autoroute", command=self.autoroute, accelerator="Cmd+R"
        )
        edit_menu.add_command(label="Undo", command=self.undo, accelerator="Cmd+Z")
        edit_menu.add_command(
            label="Redo", command=self.redo, accelerator="Cmd+Shift+Z"
        )
        self.master.bind_all("<KeyPress>", self.key_press)
        self.master.bind_all("<Command-Shift-Z>", self.redo)
        self.master.bind_all("<Command-d>", self.clear)
        self.master.bind_all("<Command-D>", self.clear)
        self.master.bind_all("<Command-l>", self.clear_all)
        self.master.bind_all("<Command-L>", self.clear_all)
        self.master.bind_all("<Command-r>", self.autoroute)
        self.master.bind_all("<Command-R>", self.autoroute)
        self.master.bind_all("<Command-z>", self.undo)
        self.master.bind_all("<Command-Z>", self.undo)
        self.master.bind_all("<Command-Shift-z>", self.redo)
        self.master.bind_all("<Command-Shift-Z>", self.redo)
        menu_bar.add_cascade(label="Edit", menu=edit_menu)

        self.master.config(menu=menu_bar)

        # Set up Tkinter canvas
        self.canvas = Canvas(self.master, width=w, height=h, highlightthickness=0)
        self.canvas.grid(row=1, column=0, sticky="nswe")
        self.canvas.update()

        # Make canvas expandable and allow for negative coordinates
        self.canvas.configure(scrollregion=self.canvas.bbox("ALL"))
        self.master.rowconfigure(1, weight=1)
        self.master.columnconfigure(0, weight=1)

        # Bind events to the Canvas
        self.canvas.bind("<Configure>", self.show_image)
        self.canvas.bind("<ButtonPress-3>", self.move_from) # Panning
        self.canvas.bind("<B3-Motion>", self.move_to) # Panning
        self.canvas.bind("<MouseWheel>", self.wheel)  # Scroll for Windows and MacOS
        self.canvas.bind("<Button-5>", self.wheel)  # Scroll down for Linux
        self.canvas.bind("<Button-4>", self.wheel)  # Scroll up for Linux

        # Put image into container rectangle and use it to set proper coordinates to the image
        self.container = self.canvas.create_rectangle(0, 0, w, h, width=0)

        # Convert input, output, and potential ports to PNG coordinates and draw their respective markers
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

        self.png_potential_ports = self.get_png_ports(potential_ports)
        self.potential_port_markers = list(
            map(
                lambda x: self.canvas.create_polygon(*x, fill=self.potential_color),
                utils.get_port_polygons(
                    self.png_potential_ports, self.port_length, self.port_width
                ),
            )
        )
        # If we have provided paths, no need to start in Add Path mode
        if self.N != 0:
            for marker in self.potential_port_markers:
                self.canvas.itemconfigure(marker, state="hidden")

        # Color the selected ports appropriately
        if self.N > 0:
            self.canvas.itemconfig(self.input_rects[0], fill=self.selected_color)
            self.canvas.itemconfig(self.output_rects[0], fill=self.selected_color)

        # Toolbar
        toolbar = Frame(self.master, relief=RAISED)

        # Path selection variables
        self.selected_path_index = (
            0  # Index of selected option for addressing various arrays, equal to self.N if in Add Path mode
        )
        self.current_paths = [
            [utils.get_port_coords(self.png_inputs[i])] for i in range(self.N)
        ] + [
            []
        ]  # List of current coordinates for each path, defaults to just those of the input port. Starts as an empty list for index self.N, corresponding to Add Path mode

        self.max_undos = 10
        self.undo_paths = [] # Stored paths for undoing
        self.undo_terminated = [] # Stored information about whether paths were terminated
        self.redo_paths = [] # Stored paths for redoing
        self.redo_terminated = [] # Stored information about whether paths were terminated
        self.path_lines = [None] * (
            self.N + 1
        )  # Tkinter Canvas object IDs of lines representing current paths
        self.path_terminated = [False] * (
            self.N + 1
        )  # Whether the current path connects the input and output ports, determines if user should be in drawing mode
        self.cursor_line = self.canvas.create_line(
            0,
            0,
            0,
            0,
            width=self.linewidth,
        )  # Line from last placed point to cursor

        self.new_input = -1  # Index of potential input port of path being added

        # Path selection widgets
        self.next_path_index = self.N + 1
        self.paths = [f"Path {i+1}" for i in range(self.N)] + [
            "Add Path"
        ]  # String options for Tkinter OptionMenu
        self.selected_path = StringVar(
            self.master, self.paths[self.selected_path_index]
        )  # String selected by OptionMenu
        self.path_select = OptionMenu(
            toolbar, self.selected_path, *self.paths, command=self.path_selected
        )  # OptionMenu for selecting paths
        redraw_button = Button(toolbar, text="Redraw", command=self.clear)
        clear_all_button = Button(toolbar, text="Clear All", command=self.clear_all)
        autoroute_button = Button(toolbar, text="Autoroute", command=self.autoroute)

        # Toolbar UI layout
        toolbar.grid(row=0, column=0, sticky="nwe")
        self.path_select.grid(row=0, column=0)
        redraw_button.grid(row=0, column=1)
        clear_all_button.grid(row=0, column=2)
        autoroute_button.grid(row=0, column=3)

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
        if not bbox[0] < x < bbox[2] or not bbox[1] < y < bbox[3]:
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
        if (
            not self.path_terminated[self.selected_path_index]
            and len(self.current_paths[self.selected_path_index]) > 0
        ):
            self.canvas.itemconfigure(self.cursor_line, state="normal")
        else:
            self.canvas.itemconfigure(self.cursor_line, state="hidden")
        if len(self.current_paths[self.selected_path_index]) > 0:
            cur_coords = self.canvas.coords(self.cursor_line)
            coords = self.get_zoom_coords(
                self.current_paths[self.selected_path_index][-1]
            )
            self.canvas.coords(self.cursor_line, *coords, cur_coords[2], cur_coords[3])
        for i in range(self.N + 1):
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

    def select_path(self, value):
        self.selected_path.set(value)
        self.path_selected()

    def path_selected(self, *args):
        """Handler for changing the selected path"""
        # Set the appropriate marker colors and update the current index
        for i in range(self.N + 1):
            if self.paths[i] == self.selected_path.get():
                self.selected_path_index = i
                if i < self.N:
                    self.canvas.itemconfig(
                        self.input_rects[i], fill=self.selected_color
                    )
                    self.canvas.itemconfig(
                        self.output_rects[i], fill=self.selected_color
                    )
                else:
                    for marker in self.potential_port_markers:
                        self.canvas.itemconfigure(marker, state="normal")
            elif i < self.N:
                self.canvas.itemconfig(
                    self.input_rects[i], fill=self.deselected_color
                )
                self.canvas.itemconfig(
                    self.output_rects[i], fill=self.deselected_color
                )
            else:
                for marker in self.potential_port_markers:
                    self.canvas.itemconfigure(marker, state="hidden")
        self.show_image()

    def motion(self, event):
        """Handler for mouse movement"""
        # If the path is not terminated, show the drawing cursor line
        event_coords = self.get_canvas_coords(np.array([event.x, event.y]))
        if len(self.current_paths[self.selected_path_index]) > 0:
            coords = self.get_zoom_coords(
                self.current_paths[self.selected_path_index][-1]
            )
            self.canvas.coords(self.cursor_line, *coords, *event_coords)
        else:
            self.canvas.coords(self.cursor_line, *event_coords, *event_coords)

    def click(self, event):
        """Handler for mouse click"""
        # If the path is not terminated, add to the current path
        if not self.path_terminated[self.selected_path_index]:
            self.push_state()
            event_coords = np.array([event.x, event.y])
            coords = self.get_canvas_coords(event_coords)
            buffer = np.array([5, 5])
            coords1 = coords - buffer
            coords2 = coords + buffer
            overlapping_objects = self.canvas.find_overlapping(*coords1, *coords2)
            # If the new endpoint causes the path to enter the output port, terminate the path
            if self.selected_path_index < self.N:
                if self.output_rects[self.selected_path_index] in overlapping_objects:
                    self.path_terminated[self.selected_path_index] = True
                    self.current_paths[self.selected_path_index].append(
                        utils.get_port_coords(
                            self.png_outputs[self.selected_path_index]
                        )
                    )
                else:
                    self.current_paths[self.selected_path_index].append(
                        self.get_real_coords(event_coords)
                    )
            else:
                found = False
                for i in range(len(self.potential_port_markers)):
                    marker = self.potential_port_markers[i]
                    port_coords = utils.get_port_coords(self.png_potential_ports[i])
                    if marker in overlapping_objects and all(
                        map(
                            lambda x: not np.allclose(port_coords, x),
                            self.current_paths[self.selected_path_index],
                        )
                    ):
                        if len(self.current_paths[self.selected_path_index]) != 0:
                            self.path_terminated[self.selected_path_index] = True
                            self.path_terminated.append(False)
                            self.current_paths.append([])
                            self.add_input_port(self.potential_ports[self.new_input])
                            self.add_output_port(
                                utils.reverse_port(self.potential_ports[i])
                            )
                            self.remove_potential_port(max(self.new_input, i))
                            self.remove_potential_port(min(self.new_input, i))
                            path_name = f"Path {self.next_path_index}"
                            self.next_path_index += 1
                            self.paths.insert(self.selected_path_index, path_name)
                            self.path_select["menu"].delete(self.N, "end")
                            self.path_select["menu"].add_command(
                                label=path_name,
                                command=lambda value=string: self.select_path(
                                    path_name
                                ),
                            )
                            self.path_select["menu"].add_command(
                                label="Add Path",
                                command=lambda value=string: self.select_path(
                                    "Add Path"
                                ),
                            )
                            self.path_lines.append(None)
                            self.new_input = -1
                            self.N += 1
                            self.current_paths[self.selected_path_index].append(
                                port_coords
                            )
                            self.select_path(path_name)
                        else:
                            self.new_input = i
                            self.canvas.itemconfig(
                                self.potential_port_markers[i], fill=self.selected_color
                            )
                            self.current_paths[self.selected_path_index].append(
                                port_coords
                            )
                        found = True
                        break
                if not found and len(self.current_paths[self.selected_path_index]) > 0:
                    self.current_paths[self.selected_path_index].append(
                        self.get_real_coords(event_coords)
                    )

        self.show_image()

    def add_input_port(self, port):
        self.inputs.append(port)
        self.png_inputs.extend(self.get_png_ports([port]))
        self.input_rects.extend(
            list(
                map(
                    lambda x: self.canvas.create_polygon(
                        *x, fill=self.deselected_color
                    ),
                    utils.get_port_polygons(
                        [self.png_inputs[-1]], self.port_length, self.port_width
                    ),
                )
            )
        )

    def remove_input_port(self, i):
        del self.inputs[i]
        del self.png_inputs[i]
        self.canvas.delete(self.input_rects[i])
        del self.input_rects[i]

    def add_output_port(self, port):
        self.outputs.append(port)
        self.png_outputs.extend(self.get_png_ports([port]))
        self.output_rects.extend(
            list(
                map(
                    lambda x: self.canvas.create_polygon(
                        *x, fill=self.deselected_color
                    ),
                    utils.get_port_polygons(
                        [self.png_outputs[-1]], -self.port_length, self.port_width
                    ),
                )
            )
        )

    def remove_output_port(self, i):
        del self.outputs[i]
        del self.png_outputs[i]
        self.canvas.delete(self.output_rects[i])
        del self.output_rects[i]

    def add_potential_port(self, port):
        self.potential_ports.append(port)
        self.png_potential_ports.extend(self.get_png_ports([port]))
        self.potential_port_markers.extend(
            list(
                map(
                    lambda x: self.canvas.create_polygon(
                        *x, fill=self.potential_color
                    ),
                    utils.get_port_polygons(
                        [self.potential_ports[-1]], -self.port_length, self.port_width
                    ),
                )
            )
        )

    def remove_potential_port(self, i):
        del self.potential_ports[i]
        del self.png_potential_ports[i]
        self.canvas.delete(self.potential_port_markers[i])
        del self.potential_port_markers[i]

    def push_state(self):
        if len(self.undo_paths) == self.max_undos:
            self.undo_paths.pop(0)
            self.undo_terminated.pop(0)
        self.undo_paths.append(list(map(lambda x: x[:], self.current_paths)))
        self.undo_terminated.append(self.path_terminated[:])
        self.redo_paths = []
        self.redo_terminated = []

    def add_path(self, *args):
        self.select_path("Add Path")

    def delete_path(self, *args):
        if self.selected_path_index == self.N:
            return

        self.select_path(self.paths(max(0, self.selected_path-1)))

    def clear_by_index(self, index):
        """Clear the current path for the given index"""
        if index < self.N:
            self.current_paths[index] = [
                utils.get_port_coords(self.png_inputs[self.selected_path_index])
            ]
            self.canvas.delete(self.path_lines[index])
            self.path_lines[index] = None
            self.path_terminated[index] = False
        else:
            self.current_paths[index] = []
            self.canvas.delete(self.path_lines[index])
            self.path_lines[index] = None
            self.path_terminated[index] = False
            self.canvas.itemconfig(
                self.potential_port_markers[self.new_input], fill=self.potential_color
            )
            self.new_input = -1

    def clear(self, *args):
        """Clear the currently selected path"""
        self.push_state()
        self.clear_by_index(self.selected_path_index)
        self.show_image()

    def clear_all(self, *args):
        """Clear all paths"""
        self.push_state()
        for i in range(self.N + 1):
            self.clear_by_index(i)
        self.show_image()

    def set_route_file(self, f):
        lines = f.read().strip().split("\n")
        if len(lines) < 1:
            raise TypeError("No data found")
        N = int(lines.pop(0))
        if len(lines) != 3 * N:
            raise TypeError("Incorrect number of lines")
        inputs = []
        for i in range(N):
            data = lines.pop(0).strip().split()
            if len(data) != 3:
                raise TypeError("Invalid port specification")
        outputs = []
        for i in range(N):
            data = lines.pop(0).strip().split()
            if len(data) != 3:
                raise TypeError("Invalid port specification")
            outputs.append(tuple(map(float, data)))
        for i in range(N):
            if inputs[i] != self.inputs[i] or outputs[i] != self.outputs[i]:
                messagebox.showerror("Ports do not correspond!")
                f.close()
                return
        paths = []
        path_terminated = []
        for i in range(N):
            data = lines.pop(0).strip().split()
            cur_terminated = data.pop(0) == "*"
            data = list(map(float, data))
            if len(data) % 2 != 0:
                raise TypeError("Invalid path specification")
            path = []
            for j in range(len(data) // 2):
                path.append(np.array([data[2 * j], data[2 * j + 1]]))
            path[0] = utils.get_port_coords(self.png_inputs[i])
            if cur_terminated:
                path[-1] = utils.get_port_coords(self.png_outputs[i])
            paths.append(path)
            path_terminated.append(cur_terminated)
        self.current_paths = paths
        self.path_terminated = path_terminated
        self.show_image()

    def open(self, *args):
        f = filedialog.askopenfile()
        if f is None:
            return
        try:
            self.set_route_file(f)
        except Exception as e:
            print(e)
            messagebox.showerror("error", "File is malformed!")
        f.close()

    def save_paths(self):
        with open(self.current_file, "w") as f:
            f.write(f"{self.N} {self.gds_bbox}\n")
            for i in range(self.N):
                f.write(f"{str(self.inputs[i])}\n")
            for i in range(self.N):
                f.write(f"{str(self.outputs[i])}\n")
            for i in range(self.N):
                f.write(
                    f"{'*' if self.path_terminated[i] else '-'} {' '.join(map(lambda x: f'{x[0]} {x[1]}', self.current_paths[i]))}\n"
                )
            for i in range(self.N):
                f.write(f"{str(self.potential_ports[i])}\n")

    def save(self, *args):
        if self.current_file is None:
            self.save_as()
        else:
            self.save_paths()

    def save_as(self, *args):
        self.current_file = filedialog.asksaveasfilename(
            defaultextension=".route",
            filetypes=(("Autoroute File", "*.route"), ("All Files", "*.*")),
        )
        if self.current_file is None:
            return
        self.save_paths()

    def key_press(self, event):
        if event.char == "e":
            self.next_path()
        if event.char == "q":
            self.prev_path()
        if event.char == "A":
            self.add_path()

    def next_path(self, *args):
        if self.N == 0:
            return
        if self.selected_path_index == self.N:
            self.selected_path_index -= 1
        self.select_path(self.paths[(self.selected_path_index + 1) % self.N])

    def prev_path(self, *args):
        if self.N == 0:
            return
        self.select_path(self.paths[(self.selected_path_index - 1) % self.N])

    def undo(self, *args):
        if len(self.undo_paths) == 0:
            messagebox.showerror("error", "Nothing to undo!")
            return
        self.redo_paths.append(list(map(lambda x: x[:], self.current_paths)))
        self.redo_terminated.append(self.path_terminated[:])
        for i in range(self.N):
            self.clear_by_index(i)
        self.current_paths = self.undo_paths.pop()
        self.path_terminated = self.undo_terminated.pop()
        self.show_image()

    def redo(self, *args):
        if len(self.redo_paths) == 0:
            messagebox.showerror("error", "Nothing to redo!")
            return
        self.undo_paths.append(list(map(lambda x: x[:], self.current_paths)))
        self.undo_terminated.append(self.path_terminated[:])
        self.clear_all()
        self.current_paths = self.redo_paths.pop()
        self.path_terminated = self.redo_terminated.pop()
        self.show_image()

    def autoroute(self, *args):
        if not all(self.path_terminated[: self.N]):
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
        return routing.Port(png_coords[0], png_coords[1], port.angle)

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

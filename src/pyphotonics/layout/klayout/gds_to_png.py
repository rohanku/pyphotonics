import pya

app = pya.Application.instance()
mw = app.main_window()
mw.load_layout(current_gds, 0)
lv = pya.LayoutView.current()
ly = pya.CellView.active().layout()

# top cell bounding box in micrometer units
bbox = pya.DBox(float(bx1), float(by1), float(bx2), float(by2))
lv.save_image_with_options(output_path, int(w), int(h), 0, 0, 0, bbox, False)

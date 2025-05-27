import math
import logging
from itertools import cycle
from typing import List, Tuple, Optional, Any
from PIL import Image, ImageDraw, ImageColor, ImageFont

import flet as ft
import flet.canvas as cv

# --- Configuration ---
DEFAULT_FONT_PATH = "arial.ttf"
BEZIER_APPROXIMATION_STEPS = 20

# --- Logging Setup ---
logger = logging.getLogger("flet_canvas2img")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.WARNING) # Change to INFO or DEBUG for more verbosity

# --- Pillow Lanczos filter compatibility ---
try:
    LANCZOS_FILTER = Image.Resampling.LANCZOS
except AttributeError:
    LANCZOS_FILTER = Image.LANCZOS

# --- Bezier Curve Helper Functions ---
def _approximate_quadratic_bezier(p0: Tuple[float, float], p1: Tuple[float, float], p2: Tuple[float, float], 
                                  steps: int = BEZIER_APPROXIMATION_STEPS) -> List[Tuple[float, float]]:
    """Returns a list of points approximating a quadratic Bezier curve."""
    if steps <= 0: return [p0, p2]
    return [
        (
            (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0],
            (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1]
        )
        for t in (i / steps for i in range(steps + 1))
    ]

def _approximate_cubic_bezier(p0: Tuple[float, float], p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float], 
                              steps: int = BEZIER_APPROXIMATION_STEPS) -> List[Tuple[float, float]]:
    """Returns a list of points approximating a cubic Bezier curve."""
    if steps <= 0: return [p0, p3]
    return [
        (
            (1 - t) ** 3 * p0[0] + 3 * (1 - t) ** 2 * t * p1[0] + 3 * (1 - t) * t ** 2 * p2[0] + t ** 3 * p3[0],
            (1 - t) ** 3 * p0[1] + 3 * (1 - t) ** 2 * t * p1[1] + 3 * (1 - t) * t ** 2 * p2[1] + t ** 3 * p3[1]
        )
        for t in (i / steps for i in range(steps + 1))
    ]

# --- Dashed Line Helper Function ---
def _draw_dashed_line(draw_context: ImageDraw.ImageDraw, 
                      points_list: List[Tuple[float, float]], 
                      dash_pattern_list: List[float], 
                      fill_color: Tuple[int, int, int, int], 
                      stroke_width: int):
    """Draws a dashed line or polyline. The dash pattern resets for each segment of the polyline."""
    if not points_list or len(points_list) < 2 or stroke_width <= 0:
        return # Nothing to draw or invisible
    if not dash_pattern_list or len(dash_pattern_list) < 2:
        # Draw as solid if no valid dash pattern
        draw_context.line(points_list, fill=fill_color, width=stroke_width)
        return
    for i in range(len(points_list) - 1):
        p_start, p_end = points_list[i], points_list[i+1]
        dx, dy = p_end[0] - p_start[0], p_end[1] - p_start[1]
        segment_length = math.hypot(dx, dy)
        if segment_length == 0: continue
        norm_dx, norm_dy = dx / segment_length, dy / segment_length
        current_pos_on_segment = 0.0
        current_pattern_iter = cycle(dash_pattern_list)
        while current_pos_on_segment < segment_length:
            dash_on_len = next(current_pattern_iter, 0.0)
            dash_off_len = next(current_pattern_iter, 0.0)
            if dash_on_len <= 0 and dash_off_len <= 0:
                logger.debug(f"Dash pattern {dash_pattern_list} resulted in zero advancement. Breaking dash for segment.")
                break
            if dash_on_len > 0:
                actual_draw_len = min(dash_on_len, segment_length - current_pos_on_segment)
                if actual_draw_len > 0:
                    start_x = p_start[0] + norm_dx * current_pos_on_segment
                    start_y = p_start[1] + norm_dy * current_pos_on_segment
                    end_x = p_start[0] + norm_dx * (current_pos_on_segment + actual_draw_len)
                    end_y = p_start[1] + norm_dy * (current_pos_on_segment + actual_draw_len)
                    draw_context.line([(start_x, start_y), (end_x, end_y)], fill=fill_color, width=stroke_width)
            current_pos_on_segment += dash_on_len + dash_off_len

# --- Flet Paint Attribute Extraction Helper ---
def get_flet_paint_attributes(shape_paint: Optional[ft.Paint] = None) -> \
    Tuple[Tuple[int,int,int,int], float, ft.PaintingStyle, ft.StrokeCap, ft.StrokeJoin, Optional[List[float]]]:
    """
    Extracts Flet paint attributes with sensible defaults.
    Returns: tuple (pil_color_rgba, stroke_width, paint_style, stroke_cap, stroke_join, dash_pattern)
    """
    color_str: str = "black"
    stroke_width: float = 1.0
    paint_style: ft.PaintingStyle = ft.PaintingStyle.FILL
    stroke_cap: ft.StrokeCap = ft.StrokeCap.BUTT
    stroke_join: ft.StrokeJoin = ft.StrokeJoin.MITER
    dash_pattern: Optional[List[float]] = None
    if shape_paint:
        if shape_paint.color is not None: color_str = shape_paint.color
        if shape_paint.stroke_width is not None: stroke_width = float(shape_paint.stroke_width)
        if shape_paint.style is not None: paint_style = shape_paint.style
        if shape_paint.stroke_cap is not None: stroke_cap = shape_paint.stroke_cap
        if shape_paint.stroke_join is not None: stroke_join = shape_paint.stroke_join
        flet_dash_attr = getattr(shape_paint, 'dash_pattern', None)
        if isinstance(flet_dash_attr, list) and len(flet_dash_attr) >= 2:
            try:
                processed_dash = [float(d) for d in flet_dash_attr]
                if all(d >= 0 for d in processed_dash):
                    dash_pattern = processed_dash
                else:
                    logger.warning(f"Dash pattern contains negative values: {flet_dash_attr}. Using solid line.")
            except ValueError:
                logger.warning(f"Dash pattern contains non-numeric values: {flet_dash_attr}. Using solid line.")
        elif flet_dash_attr is not None:
            logger.warning(f"Invalid dash_pattern format: {flet_dash_attr}. Expected list of numbers. Using solid line.")
    try:
        pil_color_rgba: Tuple[int,int,int,int] = ImageColor.getcolor(color_str, "RGBA")
    except ValueError:
        if str(color_str).lower() == "transparent":
            pil_color_rgba = (0, 0, 0, 0)
        else:
            logger.warning(f"Could not parse color '{color_str}'. Defaulting to opaque black.")
            pil_color_rgba = ImageColor.getcolor("black", "RGBA")
    return pil_color_rgba, stroke_width, paint_style, stroke_cap, stroke_join, dash_pattern

def canvas2img(
    shapes: List[Any],
    width: int = 770,
    height: int = 640,
    bgcolor: Tuple[int, int, int, int] = (255, 255, 255, 255),
    can_save: bool = True,
    save_path: str = "output.png",
    supersampling_factor: float = 1.0,
) -> Optional[Image.Image]:
    """
    Renders a list of Flet canvas shapes to a PIL (Pillow) Image object.
    Supports: Line, Circle, Ellipse, Arc, Rect (including border_radius), 
    Polygon, Polyline, Image, Path (with LineTo, QuadraticTo, CubicTo, Close), 
    and basic Text rendering.
    Features include anti-aliasing via supersampling and dashed line patterns.

    :param shapes: A list of Flet canvas shape objects (e.g., `cv.Line`, `cv.Circle`).
    :param width: The target width of the final output image in pixels. Must be > 0.
    :param height: The target height of the final output image in pixels. Must be > 0.
    :param bgcolor: Background color of the image as an (R, G, B, A) tuple.
                   Defaults to opaque white. For transparent, use (R,G,B,0).
    :param can_save: If True, the generated image will be saved to `save_path`.
    :param save_path: The file path where the image will be saved if `can_save` is True.
    :param supersampling_factor: The factor for supersampling (e.g., 2.0 for 2x resolution).
                                 Values <= 1.0 result in no supersampling. Higher values produce smoother edges but increase processing time and memory.
    :return: A `PIL.Image.Image` object, or `None` if a critical error occurs (e.g., invalid dimensions).
    """
    if not isinstance(shapes, list):
        logger.error("`shapes` argument must be a list. Cannot generate image.")
        return None
    if width <= 0 or height <= 0:
        logger.error(f"Target width ({width}) and height ({height}) must be positive. Cannot generate image.")
        return None

    scale = float(supersampling_factor) if supersampling_factor > 1.0 else 1.0
    render_width = max(1, int(round(width * scale)))
    render_height = max(1, int(round(height * scale)))
    try:
        img = Image.new("RGBA", (render_width, render_height), bgcolor)
    except Exception as e:
        logger.error(f"Failed to create new PIL Image ({render_width}x{render_height}): {e}")
        return None
    draw = ImageDraw.Draw(img, "RGBA")

    # --- Shape Processing Loop ---
    for shape_idx, shape_obj in enumerate(shapes):
        if shape_obj is None:
            continue
        paint_attributes = getattr(shape_obj, 'paint', None)
        pil_color, stroke_w_orig, style, cap, join, dash_pattern_orig = get_flet_paint_attributes(paint_attributes)
        stroke_w_scaled = stroke_w_orig * scale
        pil_draw_stroke_width = max(1, int(round(stroke_w_scaled))) if stroke_w_scaled > 0 else 0
        dash_pattern_scaled = [max(0.5, d_val * scale) if d_val > 0 else 0 for d_val in dash_pattern_orig] if dash_pattern_orig else None
        pil_join_type_arg = "curve" if join == ft.StrokeJoin.ROUND else None
        # Note: ft.StrokeJoin.BEVEL is not directly supported by PIL's line `joint` parameter. Miter is the default behavior if `joint` is None or not "curve".

        # --- Line Shape ---
        if isinstance(shape_obj, cv.Line):
            x1s, y1s = float(shape_obj.x1) * scale, float(shape_obj.y1) * scale
            x2s, y2s = float(shape_obj.x2) * scale, float(shape_obj.y2) * scale
            if pil_draw_stroke_width > 0:
                if dash_pattern_scaled:
                    _draw_dashed_line(draw, [(x1s, y1s), (x2s, y2s)], dash_pattern_scaled, pil_color, pil_draw_stroke_width)
                else:
                    draw.line([(x1s, y1s), (x2s, y2s)], fill=pil_color, width=pil_draw_stroke_width)
                if cap == ft.StrokeCap.ROUND and stroke_w_scaled > 0.1: # Check scaled width for meaningful cap
                    r_cap = stroke_w_scaled / 2.0
                    draw.ellipse((x1s - r_cap, y1s - r_cap, x1s + r_cap, y1s + r_cap), fill=pil_color, outline=None)
                    draw.ellipse((x2s - r_cap, y2s - r_cap, x2s + r_cap, y2s + r_cap), fill=pil_color, outline=None)

        # --- Circle & Ellipse Shapes (Common logic factored) ---
        elif isinstance(shape_obj, (cv.Circle, cv.Ellipse)):
            bbox_coords: List[float]
            if isinstance(shape_obj, cv.Circle):
                center_x, center_y = float(shape_obj.x) * scale, float(shape_obj.y) * scale
                radius = float(shape_obj.radius) * scale
                if radius < 0: radius = 0 # Ensure non-negative radius
                bbox_coords = [center_x - radius, center_y - radius, center_x + radius, center_y + radius]
            else: # cv.Ellipse
                x_el, y_el = float(shape_obj.x) * scale, float(shape_obj.y) * scale
                w_el, h_el = float(shape_obj.width) * scale, float(shape_obj.height) * scale
                if w_el < 0: w_el = 0; # Ensure non-negative dimensions
                if h_el < 0: h_el = 0;
                bbox_coords = [x_el, y_el, x_el + w_el, y_el + h_el]
            fill_arg, outline_arg, width_arg_el = None, None, 0
            if style == ft.PaintingStyle.FILL:
                fill_arg = pil_color
            elif style == ft.PaintingStyle.STROKE:
                if pil_draw_stroke_width > 0:
                    outline_arg = pil_color
                    width_arg_el = pil_draw_stroke_width
            elif style == ft.PaintingStyle.STROKE_AND_FILL:
                fill_arg = pil_color
                if pil_draw_stroke_width > 0:
                    outline_arg = pil_color 
                    width_arg_el = pil_draw_stroke_width
            draw.ellipse(bbox_coords, fill=fill_arg, outline=outline_arg, width=width_arg_el)

        # --- Arc Shape ---
        elif isinstance(shape_obj, cv.Arc):
            x_arc, y_arc = float(shape_obj.x) * scale, float(shape_obj.y) * scale
            w_arc, h_arc = float(shape_obj.width) * scale, float(shape_obj.height) * scale
            if w_arc <0: w_arc = 0
            if h_arc <0: h_arc = 0
            bbox_arc = [x_arc, y_arc, x_arc + w_arc, y_arc + h_arc]
            start_angle_deg = math.degrees(shape_obj.start_angle)
            sweep_angle_deg = math.degrees(shape_obj.sweep_angle)
            end_angle_deg = start_angle_deg + sweep_angle_deg
            if style == ft.PaintingStyle.FILL or style == ft.PaintingStyle.STROKE_AND_FILL:
                draw.pieslice(bbox_arc, start=start_angle_deg, end=end_angle_deg, fill=pil_color, outline=None)
            if style == ft.PaintingStyle.STROKE or style == ft.PaintingStyle.STROKE_AND_FILL:
                if pil_draw_stroke_width > 0:
                    draw.arc(bbox_arc, start=start_angle_deg, end=end_angle_deg, fill=pil_color, width=pil_draw_stroke_width)
        
        # --- Rectangle Shape ---
        elif isinstance(shape_obj, cv.Rect):
            x_r, y_r = float(shape_obj.x) * scale, float(shape_obj.y) * scale
            w_r, h_r = float(shape_obj.width) * scale, float(shape_obj.height) * scale
            if w_r < 0: w_r = 0
            if h_r < 0: h_r = 0
            bbox_r = [x_r, y_r, x_r + w_r, y_r + h_r]
            fill_arg_r, outline_arg_r, width_arg_r = None, None, 0
            if style == ft.PaintingStyle.FILL: fill_arg_r = pil_color
            elif style == ft.PaintingStyle.STROKE:
                if pil_draw_stroke_width > 0: outline_arg_r, width_arg_r = pil_color, pil_draw_stroke_width
            elif style == ft.PaintingStyle.STROKE_AND_FILL:
                fill_arg_r = pil_color
                if pil_draw_stroke_width > 0: outline_arg_r, width_arg_r = pil_color, pil_draw_stroke_width
            br_flet = getattr(shape_obj, "border_radius", None)
            radius_pil = 0.0
            if isinstance(br_flet, ft.BorderRadius):
                # PIL's rounded_rectangle supports only a single uniform radius.
                # We'll use top_left and warn if others are different.
                tl_scaled = float(br_flet.top_left) * scale
                if all(abs(getattr(br_flet, corner, 0.0) * scale - tl_scaled) < 1e-6 for corner in ["top_right", "bottom_right", "bottom_left"]):
                    radius_pil = tl_scaled
                else:
                    logger.warning("Per-corner border_radius with different values provided. PIL uses a single uniform radius. Using top-left.")
                    radius_pil = tl_scaled
            elif isinstance(br_flet, (int, float)) and br_flet > 0:
                radius_pil = float(br_flet) * scale
            if radius_pil > 0.1: # Draw rounded if radius is significant
                draw.rounded_rectangle(bbox_r, radius=radius_pil, fill=fill_arg_r, outline=outline_arg_r, width=width_arg_r)
            else: # Draw sharp rectangle
                draw.rectangle(bbox_r, fill=fill_arg_r, outline=outline_arg_r, width=width_arg_r)

        # --- Polygon Shape ---
        elif isinstance(shape_obj, cv.Polygon):
            points_poly = [(float(pt_x) * scale, float(pt_y) * scale) for pt_x, pt_y in shape_obj.points]
            if len(points_poly) < 2: continue
            if style == ft.PaintingStyle.FILL:
                if len(points_poly) >= 3: draw.polygon(points_poly, fill=pil_color, outline=None)
            elif style == ft.PaintingStyle.STROKE:
                if pil_draw_stroke_width > 0:
                    closed_pts = points_poly + [points_poly[0]] if len(points_poly) >=2 else points_poly
                    draw.line(closed_pts, fill=pil_color, width=pil_draw_stroke_width, joint=pil_join_type_arg)
            elif style == ft.PaintingStyle.STROKE_AND_FILL:
                if len(points_poly) >= 3: draw.polygon(points_poly, fill=pil_color, outline=None)
                if pil_draw_stroke_width > 0:
                    closed_pts = points_poly + [points_poly[0]] if len(points_poly) >=2 else points_poly
                    draw.line(closed_pts, fill=pil_color, width=pil_draw_stroke_width, joint=pil_join_type_arg)

        # --- Polyline Shape ---
        elif isinstance(shape_obj, cv.Polyline):
            points_pline = [(float(pt_x) * scale, float(pt_y) * scale) for pt_x, pt_y in shape_obj.points]
            if len(points_pline) < 2 or pil_draw_stroke_width <= 0: continue
            if dash_pattern_scaled:
                _draw_dashed_line(draw, points_pline, dash_pattern_scaled, pil_color, pil_draw_stroke_width)
            else:
                draw.line(points_pline, fill=pil_color, width=pil_draw_stroke_width, joint=pil_join_type_arg)
            if cap == ft.StrokeCap.ROUND and stroke_w_scaled > 0.1:
                r_cap_pl = stroke_w_scaled / 2.0
                draw.ellipse((points_pline[0][0]-r_cap_pl, points_pline[0][1]-r_cap_pl, points_pline[0][0]+r_cap_pl, points_pline[0][1]+r_cap_pl), fill=pil_color, outline=None)
                if len(points_pline) > 1 and points_pline[-1] != points_pline[0]: # Avoid double cap on closed-loop polyline
                    draw.ellipse((points_pline[-1][0]-r_cap_pl, points_pline[-1][1]-r_cap_pl, points_pline[-1][0]+r_cap_pl, points_pline[-1][1]+r_cap_pl), fill=pil_color, outline=None)

        # --- Image Shape ---
        elif isinstance(shape_obj, cv.Image):
            try:
                img_pil_src: Optional[Image.Image] = None
                if isinstance(shape_obj.src, Image.Image): img_pil_src = shape_obj.src.copy()
                elif isinstance(shape_obj.src, str) and shape_obj.src: img_pil_src = Image.open(shape_obj.src)
                else:
                    logger.warning(f"cv.Image 'src' attribute (type: {type(shape_obj.src)}) is not a PIL Image or valid file path. Skipping.")
                    continue
                x_img, y_img = float(shape_obj.x) * scale, float(shape_obj.y) * scale
                w_img, h_img = float(shape_obj.width) * scale, float(shape_obj.height) * scale
                if w_img <= 0 or h_img <= 0: continue # Skip zero-dimension images
                img_pil_src = img_pil_src.resize((int(round(w_img)), int(round(h_img))), LANCZOS_FILTER)
                if img_pil_src.mode != 'RGBA': img_pil_src = img_pil_src.convert('RGBA')
                img.paste(img_pil_src, (int(round(x_img)), int(round(y_img))), mask=img_pil_src)
            except FileNotFoundError:
                logger.warning(f"cv.Image: File not found at path '{shape_obj.src}'. Skipping.")
            except Exception as e:
                logger.warning(f"Could not process or paste cv.Image (src: {shape_obj.src}): {e}. Skipping.")

        # --- Path Shape ---
        elif isinstance(shape_obj, cv.Path):
            subpaths_data: List[dict] = []
            current_path_pts: List[Tuple[float,float]] = []
            current_path_start_pt: Optional[Tuple[float,float]] = None
            last_known_pen_pos: Optional[Tuple[float,float]] = None
            for path_elem in shape_obj.elements:
                if isinstance(path_elem, cv.Path.MoveTo):
                    if current_path_pts: subpaths_data.append({"points": list(current_path_pts), "closed": False})
                    pt_m = (float(path_elem.x) * scale, float(path_elem.y) * scale)
                    current_path_pts = [pt_m]
                    current_path_start_pt = pt_m
                    last_known_pen_pos = pt_m
                elif not last_known_pen_pos: # Other elements require a current pen position
                    logger.debug("Path element (LineTo, etc.) found without preceding MoveTo. Skipping element.")
                    continue 
                elif isinstance(path_elem, cv.Path.LineTo):
                    pt_l = (float(path_elem.x) * scale, float(path_elem.y) * scale)
                    current_path_pts.append(pt_l)
                    last_known_pen_pos = pt_l
                elif isinstance(path_elem, cv.Path.QuadraticTo):
                    ctrl_q = (float(path_elem.x1)*scale, float(path_elem.y1)*scale)
                    end_q = (float(path_elem.x2)*scale, float(path_elem.y2)*scale)
                    current_path_pts.extend(_approximate_quadratic_bezier(last_known_pen_pos, ctrl_q, end_q)[1:])
                    last_known_pen_pos = end_q
                elif isinstance(path_elem, cv.Path.CubicTo):
                    ctrl1=(float(path_elem.x1)*scale, float(path_elem.y1)*scale)
                    ctrl2=(float(path_elem.x2)*scale, float(path_elem.y2)*scale)
                    end_c=(float(path_elem.x3)*scale, float(path_elem.y3)*scale)
                    current_path_pts.extend(_approximate_cubic_bezier(last_known_pen_pos, ctrl1, ctrl2, end_c)[1:])
                    last_known_pen_pos = end_c
                # cv.Path.ArcTo is complex, requires SVG arc to Bezier/lines. Not implemented.
                elif isinstance(path_elem, cv.Path.Close):
                    if current_path_pts and current_path_start_pt:
                        # Explicitly close by adding start point if not already there
                        if current_path_pts[-1] != current_path_start_pt:
                            current_path_pts.append(current_path_start_pt)
                        subpaths_data.append({"points": list(current_path_pts), "closed": True})
                    # Reset for a new subpath (which must start with MoveTo)
                    current_path_pts, current_path_start_pt, last_known_pen_pos = [], None, None
            if current_path_pts: # Add any final unclosed subpath
                subpaths_data.append({"points": list(current_path_pts), "closed": False})

            # Draw the processed path
            if style in (ft.PaintingStyle.FILL, ft.PaintingStyle.STROKE_AND_FILL):
                for sub_info in subpaths_data:
                    if len(sub_info["points"]) >= 3: # Polygon fill needs at least 3 points
                        draw.polygon(sub_info["points"], fill=pil_color, outline=None)
            
            if style in (ft.PaintingStyle.STROKE, ft.PaintingStyle.STROKE_AND_FILL) and pil_draw_stroke_width > 0:
                for sub_info in subpaths_data:
                    pts_list, is_closed = sub_info["points"], sub_info["closed"]
                    if len(pts_list) >= 2: # Line stroke needs at least 2 points
                        if dash_pattern_scaled:
                            _draw_dashed_line(draw, pts_list, dash_pattern_scaled, pil_color, pil_draw_stroke_width)
                        else:
                            draw.line(pts_list, fill=pil_color, width=pil_draw_stroke_width, joint=pil_join_type_arg)
                        
                        # Apply caps only to visually open ends of non-explicitly-closed subpaths
                        if cap == ft.StrokeCap.ROUND and not is_closed and stroke_w_scaled > 0.1:
                            r_cap_path = stroke_w_scaled / 2.0
                            # Cap at the start of the subpath's line segments
                            draw.ellipse((pts_list[0][0]-r_cap_path, pts_list[0][1]-r_cap_path, 
                                          pts_list[0][0]+r_cap_path, pts_list[0][1]+r_cap_path), fill=pil_color, outline=None)
                            # Cap at the end, only if it's a different point (meaning path has length)
                            if len(pts_list) > 1 and pts_list[-1] != pts_list[0]:
                                draw.ellipse((pts_list[-1][0]-r_cap_path, pts_list[-1][1]-r_cap_path, 
                                              pts_list[-1][0]+r_cap_path, pts_list[-1][1]+r_cap_path), fill=pil_color, outline=None)
        
        # --- Text Shape (Basic Implementation) ---
        elif isinstance(shape_obj, cv.Text):
            try:
                font_size = int(max(1, (shape_obj.size if shape_obj.size is not None else 10) * scale))
                pil_font = ImageFont.load_default() # Default fallback
                try:
                    # Attempt to load specified/default font
                    font_family = getattr(shape_obj, 'font_family', DEFAULT_FONT_PATH) # Use font_family if available
                    if not font_family : font_family = DEFAULT_FONT_PATH # Ensure a path if font_family is empty string
                    # PIL/Pillow font selection by weight/style from family name is complex
                    pil_font = ImageFont.truetype(font_family, font_size)
                except IOError:
                    logger.debug(f"Font '{font_family}' not found or invalid. Using PIL default for Text.")
                except Exception as e_font: # Catch other font loading issues
                    logger.warning(f"Error loading font '{font_family}' for Text: {e_font}. Using PIL default.")
                text_x_s, text_y_s = float(shape_obj.x) * scale, float(shape_obj.y) * scale
                
                # Basic Flet TextAlign to PIL anchor mapping
                # Note: Flet START/END depend on LTR/RTL, not handled here. JUSTIFY not supported by draw.text.
                flet_align = getattr(shape_obj, "text_align", ft.TextAlign.LEFT)
                pil_anchor = "lt" # default: left-top
                if flet_align == ft.TextAlign.CENTER: pil_anchor = "mt" # middle-top
                elif flet_align == ft.TextAlign.RIGHT: pil_anchor = "rt" # right-top
                # More precise alignment needs font metrics (getbbox/getlength) for 'middle' of text block.
                draw.text((text_x_s, text_y_s), shape_obj.text, font=pil_font, fill=pil_color, anchor=pil_anchor)
            except Exception as e_render_text:
                logger.warning(f"Error rendering cv.Text ('{getattr(shape_obj, 'text', '')}'): {e_render_text}")

        # --- Unhandled Paint Features (e.g., Shaders/Gradients) ---
        elif paint_attributes and getattr(paint_attributes, "shader", None):
            logger.warning(f"Gradient paint (shader) on shape type {type(shape_obj).__name__} is not supported by this PIL-based renderer.")
        
        # --- Unknown or Unimplemented Shape Type ---
        elif not isinstance(shape_obj, (cv.Line, cv.Circle, cv.Ellipse, cv.Arc, cv.Rect, cv.Polygon, cv.Polyline, cv.Image, cv.Path, cv.Text)):
            shape_type_str = getattr(shape_obj, 'type', type(shape_obj).__name__)
            logger.warning(f"Flet canvas shape type '{shape_type_str}' is not implemented. Skipping shape at index {shape_idx}.")

    # --- Final Downscaling for Supersampling ---
    if scale > 1.0 and (render_width != width or render_height != height):
        final_target_width, final_target_height = int(width), int(height)
        if final_target_width > 0 and final_target_height > 0:
            try:
                img = img.resize((final_target_width, final_target_height), LANCZOS_FILTER)
            except Exception as e_resize:
                logger.error(f"Error during final image resize: {e_resize}")
        else:
            logger.warning("Target width/height for resize is zero or negative. Skipping final resize.")

    if can_save:
        try:
            img.save(save_path, "PNG")
        except Exception as e:
            logger.error(f"Failed to save image to '{save_path}': {e}")

    return img
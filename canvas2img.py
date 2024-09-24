import flet.canvas as cv
from PIL import Image, ImageDraw
import ast
import math

class Canvas2Img:
    def __init__(self, canvas: cv.Canvas):
        self.canvas = canvas
        self.shapes = canvas.shapes


def generate_image_from_shapes(shapes):
   

    # Create an empty white image
    img = Image.new("RGB", (1920, 1080), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    for i, _ in enumerate(shapes):
        shape_name, shape_props = str(shapes[i]).split(" ", 1)
        shape_properties = ast.literal_eval(shape_props)

        if shape_name == "line":
            color = ast.literal_eval(shape_properties["paint"])["color"]
           
            start_point = (shape_properties["x1"], shape_properties["y1"])
            end_point = (shape_properties["x2"], shape_properties["y2"])
            
            draw.line([start_point, end_point], fill=color, width=3)
        elif shape_name == "circle":
            x = shape_properties["x"]
            y = shape_properties["y"]
            radius = shape_properties["radius"]
            draw.ellipse(
                [x - radius, y - radius, x + radius, y + radius], outline="black"
            )
        elif shape_name == "arc":
            bounding_box = [
                shape_properties["x"],
                shape_properties["y"],
                shape_properties["width"] + shape_properties["x"],
                shape_properties["height"] + shape_properties["y"],
            ]
            start_angle = math.degrees(shape_properties["startangle"])
            end_angle = math.degrees(
                shape_properties["sweepangle"] + shape_properties["startangle"]
            )
            draw.arc(bounding_box, start=start_angle, end=end_angle, fill="black")
    # Save the image as PNG or JPG
    img.save("output.png", "PNG")
    return img

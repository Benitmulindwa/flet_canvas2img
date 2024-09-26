from PIL import Image, ImageDraw
import ast


def generate_image_from_shapes(shapes):
    # Create an empty white image
    img = Image.new("RGB", (1920, 1080), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    for i, _ in enumerate(shapes):
        shape_name, shape_props = str(shapes[i]).split(" ", 1)
        shape_properties = ast.literal_eval(shape_props)

        paint = shape_properties.get("paint", None)

        # does paint property exist?
        if paint:
            paint = ast.literal_eval(shape_properties["paint"])
            color = paint.get("color", "black")
            stroke_width = paint.get("stroke_width", 2)
            style = paint.get("style", None)
            style = color if style == "fill" else None
        else:
            color = "black"
            stroke_width = 2
            style = None

        # handle line shape
        if shape_name == "line":
            start_point = (shape_properties["x1"], shape_properties["y1"])
            end_point = (shape_properties["x2"], shape_properties["y2"])
            draw.line([start_point, end_point], fill=color, width=stroke_width)

        # handle circle shape
        elif shape_name == "circle":
            print(shape_properties)
            x = shape_properties["x"]
            y = shape_properties["y"]
            radius = shape_properties["radius"]
            draw.ellipse(
                [x - radius, y - radius, x + radius, y + radius],
                outline=color,
                width=stroke_width,
                fill=style,
            )
        # handle arc shape
        elif shape_name == "arc":

            x = shape_properties["x"]
            y = shape_properties["y"]
            width = shape_properties["width"]
            height = shape_properties["height"]

            bounding_box = [
                x,
                y,
                width + x,
                height + y,
            ]
            # convert the angles to degree
            start_angle = math.degrees(shape_properties["startangle"])
            end_angle = math.degrees(
                shape_properties["sweepangle"] + shape_properties["startangle"]
            )

            draw.arc(
                bounding_box,
                start=start_angle,
                end=end_angle,
                fill=color,
                width=stroke_width,
            )
        # handle Rectangle shape
        elif shape_name == "rect":
            x = shape_properties["x"]
            y = shape_properties["y"]
            width = shape_properties["width"]
            height = shape_properties["height"]
            draw.rectangle(
                [
                    x,
                    y,
                    width + x,
                    height + y,
                ],
                width=stroke_width,
                fill=style,
                outline=color
            )
    # Save the image as PNG or JPG
    img.save("output.png", "PNG")
    return img

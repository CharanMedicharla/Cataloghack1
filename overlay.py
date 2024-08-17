from PIL import Image

def overlay_image(background, overlay, x_offset, y_offset):
    """
    Overlays a transparent image onto the background at the specified offset.
    
    Parameters:
        background (PIL.Image): The background image.
        overlay (PIL.Image): The image to be overlaid.
        x_offset (int): The x-coordinate for the top-left corner of the overlay.
        y_offset (int): The y-coordinate for the top-left corner of the overlay.
    """
    background.paste(overlay, (x_offset, y_offset), overlay)
    return background

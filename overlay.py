from PIL import Image

def overlay_image(background, overlay, x_offset, y_offset):
    
    background.paste(overlay, (x_offset, y_offset), overlay)
    return background

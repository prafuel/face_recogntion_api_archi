import cv2

def draw_text_with_background(frame, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, 
                              font_scale=0.9,
                              text_color=(255, 255, 255),
                              bg_color=(0, 0, 0),
                              thickness=1,
                              padding=5):
    """
    Draw text with a background rectangle for better visibility.
    """
    x, y = position
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_w, text_h = text_size[0], text_size[1]

    # Draw filled rectangle as background
    cv2.rectangle(frame, (x - padding, y - text_h - padding), (x + text_w + padding, y + padding), bg_color, -1)

    # Put text on top of the rectangle
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness)

    return frame
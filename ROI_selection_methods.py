import cv2
import numpy as np
def select_pixels_in_polygon(image_array, polygon_vertices):
    # Create a mask with the same dimensions as the image, initialized to zeros
    mask = np.zeros(image_array.shape[:2], dtype=np.uint8)
    
    # Extract x and y coordinates from polygon vertices
    x_coords, y_coords = zip(*polygon_vertices)
    
    # Get the coordinates of the polygon
    rr, cc = polygon(y_coords, x_coords, mask.shape)
    
    # Fill the polygon on the mask
    mask[rr, cc] = 1
    
    # Use the mask to select the pixels within the polygon
    selected_pixels = image_array[mask == 1]
    
    return selected_pixels


def mask_out_face(image,landmarks):
        points = [(int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])) for landmark in landmarks.landmark]
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        # Create a convex hull around the face landmarks
        hull = cv2.convexHull(np.array(points))
        
        # Fill the convex hull on the mask
        cv2.fillConvexPoly(mask, hull, 255)

# Create a new image where the non-face regions are blacked out
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        return masked_image
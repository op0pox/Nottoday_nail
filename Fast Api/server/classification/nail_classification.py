import cv2
import numpy as np
import json
from scipy.spatial.distance import cdist

def normalize_contour(contour):
    contour = contour.astype(np.float32)
    x, y, w, h = cv2.boundingRect(contour)
    
    cx = x + w / 2.0
    cy = y + h / 2.0
    contour[:, 0, 0] -= cx
    contour[:, 0, 1] -= cy
    
    max_dim = max(w, h)
    if max_dim > 0:
        contour /= max_dim
        
    return contour

def calculate_chamfer_distance(contour1, contour2):
    pts1 = contour1.reshape(-1, 2)
    pts2 = contour2.reshape(-1, 2)
    
    dist_matrix = cdist(pts1, pts2, metric='euclidean')
    dist1 = np.mean(np.min(dist_matrix, axis=1))
    dist2 = np.mean(np.min(dist_matrix, axis=0))
    
    return dist1 + dist2

def classify_nail_shape(contour, template_json_path):
    with open(template_json_path, 'r', encoding='utf-8') as f:
        templates = json.load(f)
        
    norm_input = normalize_contour(contour)
    
    best_shape = None
    min_dist = float('inf')
    
    for shape_name, points in templates.items():
        template_contour = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        norm_template = normalize_contour(template_contour)
        
        dist = calculate_chamfer_distance(norm_input, norm_template)
        
        if dist < min_dist:
            min_dist = dist
            best_shape = shape_name
            
    return best_shape, min_dist
"""Image processing utilities."""

import os
from typing import List, Dict
from PIL import Image


class ImageUtils:
    """Image processing utilities"""
    
    @staticmethod
    def save_image_segments(image: Image.Image, layout_predictions, 
                           page_num: int, output_dir: str) -> List[Dict]:
        """Extract and save image segments."""
        image_segments = []
        image_labels = {'Picture', 'Figure', 'Formula', 'Table'}
        bboxes = getattr(layout_predictions, 'bboxes', [])
        
        for i, bbox_info in enumerate(bboxes):
            label = getattr(bbox_info, 'label', '')
            if label not in image_labels:
                continue
                
            bbox = getattr(bbox_info, 'bbox', [])
            if len(bbox) < 4:
                continue
                
            x1, y1, x2, y2 = bbox[:4]
            cropped_image = image.crop((x1, y1, x2, y2))
            filename = f"page_{page_num+1}_image_{i}_{label.lower()}.jpg"
            filepath = os.path.join(output_dir, filename)
            cropped_image.save(filepath, 'JPEG', quality=95)
            
            image_segments.append({
                'filename': filename,
                'filepath': filepath,
                'label': label,
                'bbox': bbox,
                'polygon': getattr(bbox_info, 'polygon', []),
                'position': getattr(bbox_info, 'position', 0),
                'confidence': getattr(bbox_info, 'confidence', 0.0)
            })
        
        return image_segments
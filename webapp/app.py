import os
import io
import base64
import json
import numpy as np
import requests
from PIL import Image, ImageDraw
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2

app = Flask(__name__)
CORS(app)

# Configuration - can be overridden by environment variables
MODEL_SERVER_URL = os.getenv('MODEL_SERVER_URL', 'http://localhost:8000')
MODEL_NAME = os.getenv('MODEL_NAME', 'airplane-detection-model')
MODEL_VERSION = os.getenv('MODEL_VERSION', '1')

# Tiled inference configuration - MUST match training parameters
TILE_WIDTH = 512
TILE_HEIGHT = 512
TILE_OVERLAP = 128
CONF_THRESHOLD = 0.15
NMS_THRESHOLD = 0.8

# Class names for airplane detection
CLASS_NAMES = {
    0: 'Aircraft'
}

def preprocess_tile(tile_image):
    """Preprocess a single tile for model inference"""
    # Convert PIL image to numpy array
    img_array = np.array(tile_image)

    # Ensure RGB format
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        # Normalize pixel values to [0, 1]
        normalized = img_array.astype(np.float32) / 255.0

        # Add batch dimension and transpose to NCHW format (channels first)
        batched = np.expand_dims(normalized, axis=0)
        batched = np.transpose(batched, (0, 3, 1, 2))

        return batched
    else:
        raise ValueError("Invalid image format")

def sliding_window_inference(image, model_server_url, model_name, model_version):
    """
    Perform inference on full-size image using sliding window approach
    matching the notebook's tiled inference implementation
    """
    img_width, img_height = image.size
    image_np = np.array(image)

    print(f"Processing image size: {img_width}x{img_height}")

    all_detections = []

    # Calculate number of tiles needed
    x_tiles = (img_width + TILE_WIDTH - TILE_OVERLAP - 1) // (TILE_WIDTH - TILE_OVERLAP)
    y_tiles = (img_height + TILE_HEIGHT - TILE_OVERLAP - 1) // (TILE_HEIGHT - TILE_OVERLAP)

    print(f"Creating {x_tiles}x{y_tiles} = {x_tiles * y_tiles} tiles")

    for y in range(y_tiles):
        for x in range(x_tiles):
            # Calculate tile boundaries
            x_start = x * (TILE_WIDTH - TILE_OVERLAP)
            y_start = y * (TILE_HEIGHT - TILE_OVERLAP)
            x_end = min(x_start + TILE_WIDTH, img_width)
            y_end = min(y_start + TILE_HEIGHT, img_height)

            # Extract tile
            tile = image_np[y_start:y_end, x_start:x_end]

            # Pad tile if necessary to maintain consistent input size
            if tile.shape[0] != TILE_HEIGHT or tile.shape[1] != TILE_WIDTH:
                padded_tile = np.zeros((TILE_HEIGHT, TILE_WIDTH, 3), dtype=np.uint8)
                padded_tile[:tile.shape[0], :tile.shape[1]] = tile
                tile = padded_tile

            # Convert to PIL Image and preprocess
            tile_pil = Image.fromarray(tile)
            processed_tile = preprocess_tile(tile_pil)

            # Prepare request for OpenVINO Model Server
            model_url = f"{model_server_url}/v2/models/{model_name}/versions/{model_version}/infer"

            input_data = {
                "inputs": [{
                    "name": "images",  # Common YOLO input name
                    "shape": list(processed_tile.shape),
                    "datatype": "FP32",
                    "data": processed_tile.flatten().tolist()
                }]
            }

            # Send request to model server
            headers = {'Content-Type': 'application/json'}
            response = requests.post(model_url, json=input_data, headers=headers, timeout=30)

            if response.status_code == 200:
                model_output = response.json()

                # Process tile detections
                tile_detections = process_tile_predictions(
                    model_output, x_start, y_start, x, y, x_tiles, y_tiles
                )
                all_detections.extend(tile_detections)

    print(f"Found {len(all_detections)} raw detections before NMS")

    # Apply Non-Maximum Suppression
    if len(all_detections) > 0:
        final_detections = apply_nms(all_detections)
        print(f"Final detections after NMS: {len(final_detections)}")
        return final_detections

    return []

def process_tile_predictions(model_output, x_start, y_start, tile_x, tile_y, x_tiles, y_tiles):
    """Process predictions from a single tile and convert to global coordinates"""
    detections = []

    # Extract predictions from model output
    outputs = model_output.get('outputs', [])
    if not outputs:
        return detections

    # Get the main output (should be "output0" with shape [1, 5, 5376])
    predictions = outputs[0].get('data', [])
    if not predictions:
        return detections

    # Convert to numpy array and reshape to expected format
    predictions = np.array(predictions)

    # Debug info for first tile
    if tile_x == 0 and tile_y == 0:
        print(f"Raw predictions shape: {predictions.shape}")
        print(f"Raw predictions length: {len(predictions)}")
        print(f"Expected shape: [1, 5, 5376] = {1 * 5 * 5376} elements")

    # Reshape to [1, 5, 5376] format
    try:
        predictions = predictions.reshape(1, 5, 5376)
        if tile_x == 0 and tile_y == 0:
            print(f"Reshaped predictions to: {predictions.shape}")
    except ValueError as e:
        print(f"Error reshaping predictions: {e}")
        return detections

    # Remove batch dimension and transpose to get [5376, 5] format
    # Original: [1, 5, 5376] -> [5, 5376] -> [5376, 5]
    predictions = predictions[0]  # Remove batch dim: [5, 5376]
    predictions = predictions.T   # Transpose: [5376, 5]

    if tile_x == 0 and tile_y == 0:
        print(f"Final predictions shape: {predictions.shape}")
        print(f"Sample detection: {predictions[0]}")
        print(f"Sample confidence values: {predictions[:5, 4]}")  # First 5 confidence values

    # Process each detection
    for i, detection in enumerate(predictions):
        # YOLO format: [x_center, y_center, width, height, confidence]
        x_center, y_center, width, height, confidence = detection

        # Apply sigmoid to confidence if it appears to be raw logits
        if confidence > 1.0:
            confidence = 1.0 / (1.0 + np.exp(-confidence))  # Sigmoid activation

        # Skip low confidence detections
        if confidence < CONF_THRESHOLD:
            continue

        if tile_x == 0 and tile_y == 0 and len(detections) < 3:
            print(f"Processing detection {i}: conf={confidence:.3f}, coords=({x_center:.3f}, {y_center:.3f}, {width:.3f}, {height:.3f})")

        # Convert from normalized coordinates (0-1) to pixel coordinates
        x_center_px = x_center * TILE_WIDTH
        y_center_px = y_center * TILE_HEIGHT
        width_px = width * TILE_WIDTH
        height_px = height * TILE_HEIGHT

        # Convert to corner coordinates
        x1 = x_center_px - width_px / 2
        y1 = y_center_px - height_px / 2
        x2 = x_center_px + width_px / 2
        y2 = y_center_px + height_px / 2

        # Ensure coordinates are within tile bounds
        x1 = max(0, min(x1, TILE_WIDTH))
        y1 = max(0, min(y1, TILE_HEIGHT))
        x2 = max(0, min(x2, TILE_WIDTH))
        y2 = max(0, min(y2, TILE_HEIGHT))

        # Skip invalid boxes
        if x2 <= x1 or y2 <= y1:
            continue

        # Apply tile filtering similar to notebook
        margin = 10
        is_center_tile = (tile_x > 0 and tile_x < x_tiles-1 and 
                        tile_y > 0 and tile_y < y_tiles-1)

        if is_center_tile:
            # Center tiles - no margin needed
            should_include = True
        else:
            # Edge tiles - apply margin filter
            should_include = (x1 > margin and y1 > margin and 
                            x2 < TILE_WIDTH - margin and y2 < TILE_HEIGHT - margin)

        if should_include:
            # Convert to global image coordinates
            global_x1 = x1 + x_start
            global_y1 = y1 + y_start
            global_x2 = x2 + x_start
            global_y2 = y2 + y_start

            # Single class model - always class 0 (Aircraft)
            class_id = 0

            detections.append([global_x1, global_y1, global_x2, global_y2, 
                             float(confidence), class_id])

            if tile_x == 0 and tile_y == 0 and len(detections) <= 3:
                print(f"Added detection {len(detections)}: conf={confidence:.3f}, bbox=({global_x1:.1f},{global_y1:.1f},{global_x2:.1f},{global_y2:.1f})")

    return detections

def apply_nms(detections, nms_threshold=NMS_THRESHOLD, conf_threshold=CONF_THRESHOLD):
    """Apply Non-Maximum Suppression to remove duplicate detections"""
    if len(detections) == 0:
        return []

    detections_array = np.array(detections)

    # Extract boxes and scores
    boxes = detections_array[:, :4].tolist()
    scores = detections_array[:, 4].tolist()

    # Apply NMS using OpenCV
    indices = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        conf_threshold,
        nms_threshold
    )

    if len(indices) > 0:
        final_detections = []
        for i in indices.flatten():
            detection = detections[i]
            final_detections.append({
                'bbox': [float(detection[0]), float(detection[1]), 
                        float(detection[2]), float(detection[3])],
                'confidence': float(detection[4]),
                'class': CLASS_NAMES.get(int(detection[5]), 'Aircraft'),
                'class_id': int(detection[5])
            })
        return final_detections

    return []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400

        # Load and process image
        image = Image.open(file.stream)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        original_size = image.size

        print(f"Processing image: {original_size}")

        # Use sliding window inference matching the notebook approach
        detections = sliding_window_inference(
            image, MODEL_SERVER_URL, MODEL_NAME, MODEL_VERSION
        )

        # Convert image to base64 for frontend display
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='JPEG')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()

        # Calculate confidence level breakdown
        high_conf = sum(1 for det in detections if det['confidence'] >= 0.7)
        med_conf = sum(1 for det in detections if 0.5 <= det['confidence'] < 0.7)
        low_conf = sum(1 for det in detections if 0.25 <= det['confidence'] < 0.5)

        return jsonify({
            'success': True,
            'image': f'data:image/jpeg;base64,{img_base64}',
            'detections': detections,
            'image_size': original_size,
            'inference_method': 'tiled',
            'tile_config': {
                'tile_size': f'{TILE_WIDTH}x{TILE_HEIGHT}',
                'overlap': TILE_OVERLAP,
                'confidence_threshold': CONF_THRESHOLD,
                'nms_threshold': NMS_THRESHOLD
            },
            'confidence_breakdown': {
                'high': high_conf,
                'medium': med_conf,
                'low': low_conf,
                'total': len(detections)
            }
        })

    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Failed to connect to model server: {str(e)}'}), 500
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    try:
        # Test connection to model server
        health_url = f"{MODEL_SERVER_URL}/v2/models/{MODEL_NAME}/versions/{MODEL_VERSION}/ready"
        response = requests.get(health_url, timeout=5)
        model_ready = response.status_code == 200

        return jsonify({
            'status': 'healthy',
            'model_server_url': MODEL_SERVER_URL,
            'model_ready': model_ready
        })
    except:
        return jsonify({
            'status': 'unhealthy',
            'model_server_url': MODEL_SERVER_URL,
            'model_ready': False
        }), 503

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)

from flask import Flask, request, jsonify, send_from_directory, render_template_string
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image, ExifTags
import io
import base64
import os
import pandas as pd
import logging
import json
import uuid
from datetime import datetime
from matplotlib import cm

UPLOAD_DIR = 'uploads'
HISTORY_FILE = 'analysis_history.json'

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
model = None
age_dataset = None
species_heuristic = None


def load_model():
    """Load the trained model"""
    global model
    model_path = 'bagtikan_classifier.h5'

    if not os.path.exists(model_path):
        logger.error(f"Model file {model_path} not found!")
        return False

    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        logger.info("Model loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False


def load_age_dataset():
    """Load the Parashorea malaanona age-DBH dataset"""
    global age_dataset
    dataset_path = 'Parashorea_malaanona_age_dbh_dataset.xlsx'

    if not os.path.exists(dataset_path):
        logger.warning(f"Age dataset {dataset_path} not found! Using fallback classification.")
        return False

    try:
        age_dataset = pd.read_excel(dataset_path)
        logger.info(f"Age dataset loaded successfully! {len(age_dataset)} records found.")
        return True
    except Exception as e:
        logger.error(f"Error loading age dataset: {e}")
        return False


def get_decimal_from_dms(dms, ref):
    """Convert DMS (Degrees, Minutes, Seconds) to decimal degrees"""
    try:
        degrees = dms[0]
        minutes = dms[1]
        seconds = dms[2]
        
        decimal = float(degrees) + (float(minutes) / 60.0) + (float(seconds) / 3600.0)
        
        if ref in ['S', 'W']:
            decimal = -decimal
            
        return decimal
    except Exception as e:
        logger.warning(f"Error converting DMS to decimal: {e}")
        return None

def get_exif_location(image):
    """Extract GPS coordinates from image EXIF data"""
    try:
        exif = image._getexif()
        if not exif:
            return None

        exif_data = {}
        for tag_id, value in exif.items():
            tag = ExifTags.TAGS.get(tag_id, tag_id)
            if tag == "GPSInfo":
                gps_data = {}
                for t in value:
                    sub_tag = ExifTags.GPSTAGS.get(t, t)
                    gps_data[sub_tag] = value[t]
                
                if 'GPSLatitude' in gps_data and 'GPSLongitude' in gps_data:
                    lat = get_decimal_from_dms(gps_data['GPSLatitude'], gps_data['GPSLatitudeRef'])
                    lng = get_decimal_from_dms(gps_data['GPSLongitude'], gps_data['GPSLongitudeRef'])
                    
                    if lat is not None and lng is not None:
                        return {'lat': lat, 'lng': lng}
    except Exception as e:
        logger.warning(f"Error extracting GPS: {e}")
        return None
    return None


def preprocess_image(image_data):
    """Preprocess the uploaded image for prediction"""
    try:
        # Decode base64 image
        image_data = image_data.split(',')[1]  # Remove data:image/jpeg;base64, prefix
        image_bytes = base64.b64decode(image_data)

        # Open image with PIL
        image = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize to model input size
        image = image.resize((224, 224))

        # Convert to numpy array and normalize
        image_array = np.array(image, dtype=np.float32)
        image_array = image_array / 255.0

        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)

        return image_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return None


def estimate_age_from_image(image_data):
    try:
        raw = image_data.split(',')[1]
        image_bytes = base64.b64decode(raw)
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize((224, 224))
        arr = np.array(image, dtype=np.float32) / 255.0

        gray = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]).astype(np.float32)
        from scipy import ndimage
        gx = ndimage.sobel(gray, axis=0, mode='reflect')
        gy = ndimage.sobel(gray, axis=1, mode='reflect')
        mag = np.hypot(gx, gy)

        # Visual feature scores
        edge_score = float(np.clip(mag.mean(), 0.0, 1.0))
        green_mask = (arr[:, :, 1] > arr[:, :, 0]) & (arr[:, :, 1] > arr[:, :, 2]) & (arr[:, :, 1] > 0.24)
        green_ratio = float(green_mask.mean())

        # Combined score: texture increases age, foliage decreases age
        combined = 0.7 * edge_score + 0.3 * (1.0 - green_ratio)
        combined = float(np.clip(combined, 0.0, 1.0))

        # Category thresholds from combined score
        # 0.00–0.25: Seedling, 0.25–0.45: Young, 0.45–0.75: Middle age, 0.75–1.00: Mature
        if combined < 0.25:
            category = 'Seedling'
            cat_lo, cat_hi = 0.00, 0.25
        elif combined < 0.45:
            category = 'Young'
            cat_lo, cat_hi = 0.25, 0.45
        elif combined < 0.75:
            category = 'Middle age'
            cat_lo, cat_hi = 0.45, 0.75
        else:
            category = 'Mature'
            cat_lo, cat_hi = 0.75, 1.00

        # Resolve age range from dataset if available, otherwise defaults
        default_ranges = {
            'Seedling': (5, 10),
            'Young': (11, 24),
            'Middle age': (25, 60),
            'Mature': (61, 120)
        }

        if age_dataset is not None and 'Age Category' in age_dataset and 'Estimated Age (years)' in age_dataset:
            cat_data = age_dataset[age_dataset['Age Category'] == category]
            if len(cat_data) > 0:
                amin = int(cat_data['Estimated Age (years)'].min())
                amax = int(cat_data['Estimated Age (years)'].max())
            else:
                amin, amax = default_ranges[category]
        else:
            amin, amax = default_ranges[category]

        # Normalize within category band and map to age range
        t = 0.5 if cat_hi == cat_lo else (combined - cat_lo) / (cat_hi - cat_lo)
        t = float(np.clip(t, 0.0, 1.0))
        estimated_age = int(round(amin + t * (amax - amin)))
        age_range = f"{amin}-{amax} years"

        return {
            'category': category,
            'age_range': age_range,
            'estimated_age': estimated_age,
            'data_source': 'Image-only analysis',
            'visual_features': {
                'edge_score': round(edge_score, 4),
                'green_ratio': round(green_ratio, 4),
                'combined_score': round(combined, 4)
            }
        }
    except Exception as e:
        logger.error(f"Image-only age estimation failed: {e}")
        return None

def image_features_from_base64(image_data):
    try:
        raw = image_data.split(',')[1]
        image_bytes = base64.b64decode(raw)
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize((224, 224))
        arr = np.array(image, dtype=np.float32) / 255.0
        gray = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]).astype(np.float32)
        from scipy import ndimage
        gx = ndimage.sobel(gray, axis=0, mode='reflect')
        gy = ndimage.sobel(gray, axis=1, mode='reflect')
        mag = np.hypot(gx, gy)
        edge_score = float(np.clip(mag.mean(), 0.0, 1.0))
        green_mask = (arr[:, :, 1] > arr[:, :, 0]) & (arr[:, :, 1] > arr[:, :, 2]) & (arr[:, :, 1] > 0.24)
        green_ratio = float(green_mask.mean())
        return edge_score, green_ratio
    except Exception:
        return None

def last_conv_layer(model):
    try:
        for layer in reversed(model.layers):
            try:
                shp = layer.output_shape
            except Exception:
                shp = None
            if hasattr(layer, 'name') and ('conv' in layer.name.lower()) and shp and len(shp) == 4:
                return layer
        for layer in reversed(model.layers):
            try:
                shp = layer.output_shape
            except Exception:
                shp = None
            if shp and len(shp) == 4:
                return layer
    except Exception:
        pass
    return None

def gradcam_heatmap(image_array):
    try:
        if model is None:
            return None
        layer = last_conv_layer(model)
        if layer is None:
            return None
        grad_model = tf.keras.models.Model([model.inputs], [layer.output, model.output])
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image_array)
            p = predictions[:, 0]
            bag_prob = 1.0 - p
            target = tf.where(p > 0.5, p, bag_prob)
        grads = tape.gradient(target, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(1, 2))
        conv_outputs = conv_outputs[0]
        pooled_grads = pooled_grads[0]
        conv_outputs = conv_outputs.numpy()
        pooled_grads = pooled_grads.numpy()
        for i in range(pooled_grads.shape[-1]):
            conv_outputs[:, :, i] *= pooled_grads[i]
        heatmap = np.mean(conv_outputs, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / (np.max(heatmap) + 1e-8)
        return heatmap
    except Exception:
        return None

def overlay_heatmap_on_image(original_image, heatmap, alpha=0.5):
    try:
        if heatmap is None:
            return None
        hm = np.uint8(255 * heatmap)
        hm_img = Image.fromarray(hm).resize(original_image.size)
        colormap = cm.get_cmap('jet')
        colored = np.uint8(colormap(np.array(hm_img) / 255.0) * 255)
        colored_img = Image.fromarray(colored).convert('RGBA')
        base = original_image.convert('RGBA')
        blended = Image.blend(base, colored_img, alpha)
        buf = io.BytesIO()
        blended.save(buf, format='PNG')
        b64 = base64.b64encode(buf.getvalue()).decode('ascii')
        return 'data:image/png;base64,' + b64
    except Exception:
        return None

def generate_xai(image_data, species_result, saved_filename=None):
    try:
        raw = image_data.split(',')[1]
        image_bytes = base64.b64decode(raw)
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_resized = img.resize((224, 224))
        arr = np.array(img_resized, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)
        heatmap = gradcam_heatmap(arr)
        if heatmap is None:
            try:
                from scipy import ndimage
                gray = (0.299 * arr[0, :, :, 0] + 0.587 * arr[0, :, :, 1] + 0.114 * arr[0, :, :, 2]).astype(np.float32)
                gx = ndimage.sobel(gray, axis=0, mode='reflect')
                gy = ndimage.sobel(gray, axis=1, mode='reflect')
                mag = np.hypot(gx, gy)
                mag = mag - mag.min()
                mag = mag / (mag.max() + 1e-8)
                heatmap = mag
            except Exception:
                heatmap = None
        overlay = overlay_heatmap_on_image(img_resized, heatmap, alpha=0.5) if heatmap is not None else None
        conf = species_result.get('confidence')
        if isinstance(conf, (int, float)):
            if conf >= 0.8:
                level = 'High confidence'
            elif conf >= 0.5:
                level = 'Medium confidence'
            else:
                level = 'Low confidence'
        else:
            level = 'Confidence unavailable'
        explanation = 'Prediction is based on bark roughness and trunk texture' if species_result.get('is_bagtikan') else 'Model focused on color consistency and foliage patterns'
        return {'heatmap': overlay, 'explanation': explanation, 'confidence_level': level}
    except Exception:
        return {'heatmap': None, 'explanation': 'Explanation unavailable', 'confidence_level': 'Confidence unavailable'}

def compute_species_heuristic():
    try:
        base = 'img_data'
        bdir = os.path.join(base, 'bagtikan')
        odir = os.path.join(base, 'others')
        if not os.path.isdir(bdir) or not os.path.isdir(odir):
            return None
        def collect(dirpath, limit=160):
            feats = []
            names = [n for n in os.listdir(dirpath) if n.lower().endswith(('.jpg','.jpeg','.png'))]
            for name in names[:limit]:
                p = os.path.join(dirpath, name)
                try:
                    with open(p, 'rb') as f:
                        b64 = base64.b64encode(f.read()).decode('ascii')
                    data_url = 'data:image/jpeg;base64,' + b64
                    fv = image_features_from_base64(data_url)
                    if fv:
                        feats.append(fv)
                except Exception:
                    pass
            return np.array(feats, dtype=np.float32)
        A = collect(bdir)
        B = collect(odir)
        if len(A) < 2 or len(B) < 2:
            return None
        ma = A.mean(axis=0)
        mb = B.mean(axis=0)
        covA = np.cov(A.T)
        covB = np.cov(B.T)
        pooled = ((len(A)-1)*covA + (len(B)-1)*covB) / max(1, (len(A)+len(B)-2))
        invS = np.linalg.pinv(pooled)
        delta = ma - mb
        w = invS @ delta
        t = 0.5 * float((ma + mb) @ w)
        d = float(np.sqrt(delta @ invS @ delta))
        beta = max(3.0, 3.0 * d)
        return {'w': w.astype(np.float32), 't': float(t), 'beta': float(beta), 'ma': ma.astype(np.float32), 'mb': mb.astype(np.float32)}
    except Exception:
        return None

def predict_species_heuristic(image_data):
    global species_heuristic
    if species_heuristic is None:
        species_heuristic = compute_species_heuristic()
    try:
        fv = image_features_from_base64(image_data)
        if not fv:
            return {
                'is_bagtikan': None,
                'species': 'Unknown Species',
                'confidence': None,
                'raw_prediction': None
            }
        edge_score, green_ratio = fv
        x = np.array([edge_score, green_ratio], dtype=np.float32)
        if species_heuristic is None:
            s = 0.7 * edge_score + 0.3 * (1.0 - green_ratio)
            p = 1.0 / (1.0 + np.exp(-8.0 * (s - 0.5)))
        else:
            w = species_heuristic['w']
            t = species_heuristic['t']
            beta = species_heuristic.get('beta', 5.0)
            z = float(w[0] * x[0] + w[1] * x[1] - t)
            p = 1.0 / (1.0 + np.exp(-beta * z))
        is_bagtikan = p >= 0.5
        conf = p if is_bagtikan else (1.0 - p)
        return {
            'is_bagtikan': bool(is_bagtikan),
            'species': 'Bagtikan (Parashorea malaanona)' if is_bagtikan else 'Other Tree Species',
            'confidence': float(conf),
            'raw_prediction': float(p)
        }
    except Exception:
        return {
            'is_bagtikan': None,
            'species': 'Unknown Species',
            'confidence': None,
            'raw_prediction': None
        }


def ensure_storage():
    try:
        if not os.path.exists(UPLOAD_DIR):
            os.makedirs(UPLOAD_DIR, exist_ok=True)
        if not os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
                f.write('[]')
        return True
    except Exception:
        return False

def load_history():
    try:
        if not os.path.exists(HISTORY_FILE):
            return []
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []

def save_history_entry(entry):
    try:
        data = load_history()
        data.insert(0, entry)
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False

def delete_history_entry(analysis_id):
    try:
        data = load_history()
        remaining = []
        deleted = None
        for x in data:
            if x.get('id') == analysis_id and deleted is None:
                deleted = x
            else:
                remaining.append(x)
        if deleted is None:
            return False
        try:
            fn = deleted.get('file_name')
            if fn:
                p = os.path.join(UPLOAD_DIR, fn)
                if os.path.exists(p):
                    os.remove(p)
        except Exception:
            pass
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(remaining, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False

def update_history_location(analysis_id, lat, lng, source=None, accuracy=None):
    try:
        data = load_history()
        found = False
        for x in data:
            if x.get('id') == analysis_id:
                x['coords'] = {'lat': float(lat), 'lng': float(lng)}
                if source is not None:
                    x['loc_source'] = str(source)
                if accuracy is not None:
                    try:
                        x['loc_accuracy_m'] = float(accuracy)
                    except Exception:
                        x['loc_accuracy_m'] = accuracy
                found = True
                break
        if not found:
            return False
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False

def save_base64_image(image_data, analysis_id):
    try:
        prefix, b64 = image_data.split(',', 1)
        ext = 'png'
        if 'image/' in prefix:
            p = prefix.split('image/')[1]
            p = p.split(';')[0]
            if p in ['png', 'jpeg', 'jpg', 'webp']:
                ext = 'jpg' if p == 'jpeg' else p
        ensure_storage()
        filename = f"{analysis_id}.{ext}"
        path = os.path.join(UPLOAD_DIR, filename)
        with open(path, 'wb') as f:
            f.write(base64.b64decode(b64))
        return filename
    except Exception:
        return None

def classify_age_from_dataset(circumference):
    """
    Classify tree age based on circumference using the actual Parashorea malaanona dataset
    """
    global age_dataset

    if age_dataset is None:
        # Fallback to original classification if dataset not available
        return classify_age_fallback(circumference)

    try:
        # Convert circumference to DBH (diameter = circumference / π)
        dbh = circumference / np.pi

        # Based on dataset analysis, optimal boundaries are:
        # Seedling: < 9.08 inches circumference
        # Young: 9.08 - 18.32 inches circumference
        # Middle age: 18.32 - 47.60 inches circumference
        # Mature: > 47.60 inches circumference

        if circumference < 9.08:
            category = 'Seedling'
            age_range = '5-10 years'
            estimated_age = 8
            description = f'Seedling tree (circumference: {circumference:.1f} inches, estimated age: ~{estimated_age} years)'
        elif circumference <= 18.32:
            category = 'Young'
            age_range = '11-24 years'
            estimated_age = 18
            description = f'Young tree (circumference: {circumference:.1f} inches, estimated age: ~{estimated_age} years)'
        elif circumference <= 47.60:
            category = 'Middle age'
            age_range = '25-60 years'
            estimated_age = 43
            description = f'Middle-aged tree (circumference: {circumference:.1f} inches, estimated age: ~{estimated_age} years)'
        else:
            category = 'Mature'
            age_range = '61-120 years'
            estimated_age = 91
            description = f'Mature tree (circumference: {circumference:.1f} inches, estimated age: ~{estimated_age} years)'

        # Get more precise age estimation using linear interpolation within category
        refined_age = estimate_age_linear_interpolation(circumference, category)

        return {
            'category': category,
            'age_range': age_range,
            'estimated_age': refined_age,
            'description': description,
            'dbh_inches': round(dbh, 2),
            'circumference_inches': round(circumference, 2),
            'data_source': 'Parashorea malaanona dataset'
        }

    except Exception as e:
        logger.error(f"Error in dataset-based age classification: {e}")
        return classify_age_fallback(circumference)


def estimate_age_linear_interpolation(circumference, category):
    """
    Provide more precise age estimation using linear interpolation within category ranges
    """
    global age_dataset

    if age_dataset is None:
        return None

    try:
        # Filter dataset by category
        category_data = age_dataset[age_dataset['Age Category'] == category].copy()

        if len(category_data) == 0:
            return None

        # Calculate circumference for each record
        category_data['Circumference'] = category_data['DBH (inch)'] * np.pi

        # Sort by circumference
        category_data = category_data.sort_values('Circumference')

        # Find the closest records for interpolation
        lower_bound = category_data[category_data['Circumference'] <= circumference]
        upper_bound = category_data[category_data['Circumference'] >= circumference]

        if len(lower_bound) == 0:
            # Input circumference is below the minimum in this category
            return int(category_data['Estimated Age (years)'].min())
        elif len(upper_bound) == 0:
            # Input circumference is above the maximum in this category
            return int(category_data['Estimated Age (years)'].max())
        else:
            # Interpolate between closest points
            lower_record = lower_bound.iloc[-1]  # Highest circumference <= input
            upper_record = upper_bound.iloc[0]  # Lowest circumference >= input

            if lower_record['Circumference'] == upper_record['Circumference']:
                return int(lower_record['Estimated Age (years)'])

            # Linear interpolation
            x1, y1 = lower_record['Circumference'], lower_record['Estimated Age (years)']
            x2, y2 = upper_record['Circumference'], upper_record['Estimated Age (years)']

            interpolated_age = y1 + (y2 - y1) * (circumference - x1) / (x2 - x1)
            return int(round(interpolated_age))

    except Exception as e:
        logger.error(f"Error in linear interpolation: {e}")
        return None


def classify_age_fallback(circumference):
    """Fallback classification if dataset is not available (original method)"""
    if circumference < 4:
        return {
            'category': 'Young',
            'description': f'Young tree (< 4 inches circumference)',
            'data_source': 'fallback classification'
        }
    elif circumference <= 20.5:
        return {
            'category': 'Middle Age',
            'description': f'Middle-aged tree (4-20.5 inches circumference)',
            'data_source': 'fallback classification'
        }
    else:
        return {
            'category': 'Old',
            'description': f'Old tree (> 20.5 inches circumference)',
            'data_source': 'fallback classification'
        }


def get_dataset_statistics():
    """Get statistics about the loaded dataset"""
    global age_dataset

    if age_dataset is None:
        return None

    stats = {}
    categories = age_dataset['Age Category'].unique()

    for category in categories:
        category_data = age_dataset[age_dataset['Age Category'] == category]
        circumferences = category_data['DBH (inch)'] * np.pi

        stats[category] = {
            'count': len(category_data),
            'age_range': [
                int(category_data['Estimated Age (years)'].min()),
                int(category_data['Estimated Age (years)'].max())
            ],
            'circumference_range': [
                round(circumferences.min(), 2),
                round(circumferences.max(), 2)
            ],
            'avg_age': round(category_data['Estimated Age (years)'].mean(), 1),
            'avg_circumference': round(circumferences.mean(), 2)
        }

    return stats


@app.route('/', methods=['GET'])
def home():
    """Serve the main HTML page with enhanced age estimation information"""
    html_content = r'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced Bagtikan Tree Detector</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Poppins:wght@400;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
      tailwind.config = {
        theme: {
          extend: {
            colors: {
              forest: { 700: '#2e7d32', 500: '#4caf50', 300: '#a5d6a7' }
            },
            fontFamily: { sans: ['Inter','Poppins','ui-sans-serif','system-ui'] }
          }
        }
      }
    </script>
</head>
<body class="min-h-screen bg-gradient-to-br from-green-50 to-emerald-100 font-sans">
  <header class="sticky top-0 z-40 bg-gradient-to-r from-[#2e7d32] to-[#4caf50] text-white shadow-lg">
    <div class="max-w-6xl mx-auto px-4">
      <div class="h-14 flex items-center justify-between">
        <a href="/" class="flex items-center gap-2">
          <span class="inline-grid place-items-center h-9 w-9 rounded-lg bg-white/10">
            <svg class="h-5 w-5" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 2C7 2 4 7 4 11C4 16 8 20 12 22C16 20 20 16 20 11C20 7 17 2 12 2Z" fill="white" fill-opacity="0.9"/></svg>
          </span>
          <span class="text-sm sm:text-base font-semibold tracking-wide">Enhanced Bagtikan Tree Detector</span>
        </a>
        <button id="navToggle" class="sm:hidden inline-flex items-center rounded-md px-3 py-2 bg-white/10 hover:bg-white/20 transition">Menu</button>
        <nav id="navMenu" class="hidden sm:flex items-center gap-2">
          <a href="/" class="px-3 py-2 rounded-lg bg-white/15 hover:bg-white/25 transition text-sm font-medium ring-1 ring-white/10">Analyze Tree</a>
          <a href="/dashboard" class="px-3 py-2 rounded-lg hover:bg-white/20 transition text-sm font-medium ring-1 ring-white/10">Dashboard</a>
          <a href="/map" class="px-3 py-2 rounded-lg hover:bg-white/20 transition text-sm font-medium ring-1 ring-white/10">Tree Map</a>
          <a href="/training" class="px-3 py-2 rounded-lg hover:bg-white/20 transition text-sm font-medium ring-1 ring-white/10">Training</a>
          <a href="/about" class="px-3 py-2 rounded-lg hover:bg-white/20 transition text-sm font-medium ring-1 ring-white/10">About</a>
        </nav>
      </div>
    </div>
  </header>
  <script>
    const t = document.getElementById('navToggle'); const m = document.getElementById('navMenu');
    if (t) t.addEventListener('click', () => { m.classList.toggle('hidden') })
  </script>
  <div class="max-w-6xl mx-auto px-4 py-10">
    <div class="rounded-2xl bg-gradient-to-r from-[#2e7d32] to-[#4caf50] text-white shadow-lg p-8 flex items-start gap-4">
      <div class="flex-shrink-0 h-12 w-12 rounded-xl bg-white/10 grid place-items-center">
        <svg class="h-6 w-6" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M12 2C7 2 4 7 4 11C4 16 8 20 12 22C16 20 20 16 20 11C20 7 17 2 12 2Z" fill="white" fill-opacity="0.9"/>
        </svg>
      </div>
      <div>
        <h1 class="text-2xl sm:text-3xl font-bold tracking-tight">Enhanced Bagtikan Tree Detector</h1>
        <p class="mt-2 text-white/90">Advanced age estimation using Parashorea malaanonan research dataset</p>
      </div>
    </div>

    <div class="mt-6 rounded-2xl bg-[#a5d6a7]/30 border border-[#a5d6a7] p-6 shadow-md">
      <div class="flex items-start gap-3">
        <div class="h-9 w-9 rounded-lg bg-[#a5d6a7] text-white grid place-items-center">📊</div>
        <div class="text-gray-800">
          <p class="leading-relaxed">Image-only estimation using computer vision. Upload a Bagtikan (Parashorea malaanonan) tree image and get species detection, confidence, and age category with a visual-cue based range.</p>
        </div>
      </div>
    </div>

    <div id="uploadCard" class="mt-8 rounded-2xl border-2 border-dashed border-gray-300 bg-white p-10 shadow-md transition duration-200">
      <input id="fileInput" type="file" accept="image/*" class="hidden">
      <div id="uploadContent" class="text-center">
        <div class="mx-auto h-20 w-20 rounded-2xl bg-gray-100 grid place-items-center text-gray-600">
          <svg class="h-10 w-10" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none"><path d="M4 7a3 3 0 013-3h10a3 3 0 013 3v10a3 3 0 01-3 3H7a3 3 0 01-3-3V7z" stroke="currentColor" stroke-width="1.5"/><path d="M9 9h6M8 13h8M9 17h6" stroke="currentColor" stroke-width="1.5"/></svg>
        </div>
        <p class="mt-4 text-gray-700">Drag & drop your Bagtikan tree image here</p>
        <button id="chooseBtn" class="mt-5 inline-flex items-center rounded-2xl bg-[#2e7d32] px-6 py-3 text-white font-semibold shadow-lg hover:bg-[#256b2b] hover:shadow-xl transition">Choose Image</button>
      </div>
      <div id="previewContainer" class="hidden">
        <img id="previewImage" class="w-full max-h-[70vh] object-contain bg-gray-100 rounded-2xl shadow-lg" alt="Preview">
        <div class="mt-3 flex items-center justify-between">
          <p id="imageInfo" class="text-sm text-gray-600"></p>
          <button id="changeImageBtn" class="inline-flex items-center rounded-lg bg-gray-100 px-3 py-2 text-gray-700 hover:bg-gray-200 transition">Change Image</button>
        </div>
      </div>
    </div>

    <div class="mt-8 rounded-2xl bg-white p-6 shadow-md">
      <h3 class="text-lg font-semibold text-gray-800">Analysis</h3>
      <p class="mt-2 text-sm text-gray-600">Run image-only species detection and age estimation.</p>
      
      <div class="mt-4 mb-4">
        <label class="block text-sm font-medium text-gray-700 mb-1">Tree Tracking</label>
        <select id="treeSelect" class="w-full rounded-lg border-gray-300 shadow-sm focus:border-forest-500 focus:ring-forest-500 text-sm p-2.5 bg-gray-50 border">
            <option value="">New Tree (Start Tracking)</option>
        </select>
        <p class="mt-1 text-xs text-gray-500">Select an existing tree to add this analysis to its history.</p>
      </div>

      <button id="analyzeBtn" class="mt-5 inline-flex items-center justify-center rounded-2xl bg-[#2e7d32] px-6 py-3 text-white font-semibold shadow-lg hover:bg-[#256b2b] hover:shadow-xl transition">Analyze Tree Image</button>
    </div>

    <div id="errorMessage" class="mt-4 hidden rounded-xl border border-red-200 bg-red-50 p-4 text-red-700"></div>

    <div id="loadingSection" class="mt-4 hidden text-center p-6">
      <div class="mx-auto h-10 w-10 rounded-full border-4 border-green-200 border-t-[#2e7d32] animate-spin"></div>
      <p class="mt-3 text-gray-700">Analyzing your tree image and estimating age...</p>
    </div>

    <div id="resultsSection" class="mt-6 hidden grid gap-4">
      <div class="rounded-2xl bg-white p-6 shadow-md">
        <h4 class="text-base font-semibold text-gray-800">Species Detection</h4>
        <div id="speciesResult" class="mt-2 text-xl font-bold"></div>
        <div id="confidenceResult" class="mt-1 text-sm text-gray-600"></div>
      </div>
      <div id="ageCard" class="rounded-2xl bg-white p-6 shadow-md">
        <h4 class="text-base font-semibold text-gray-800">Age Estimation</h4>
        <div id="ageResult" class="mt-2 text-xl font-bold"></div>
        <div id="ageDetails" class="mt-3 text-sm text-gray-700"></div>
      </div>
      <div id="xaiCard" class="rounded-2xl bg-white p-6 shadow-md">
        <h4 class="text-base font-semibold text-gray-800">Model Explanation</h4>
        <div class="mt-2 text-sm text-gray-600">These explanations reflect model-based visual cues and are not absolute ground truth.</div>
        <div class="mt-3 flex gap-2">
          <button id="showOriginalBtn" class="inline-flex items-center rounded-lg bg-gray-100 px-3 py-2 text-gray-700 hover:bg-gray-200">Original</button>
          <button id="showHeatmapBtn" class="inline-flex items-center rounded-lg bg-forest-700 px-3 py-2 text-white hover:bg-[#256b2b]">Heatmap</button>
        </div>
        <div class="mt-3">
          <img id="xaiImageOriginal" class="w-full max-h-[40vh] object-contain rounded-lg border border-gray-200 hidden" alt="Original">
          <img id="xaiImageHeatmap" class="w-full max-h-[40vh] object-contain rounded-lg border border-gray-200 hidden" alt="Heatmap">
        </div>
        <div id="xaiText" class="mt-3 text-sm text-gray-800"></div>
        <div id="xaiConf" class="mt-1 text-xs text-gray-600"></div>
      </div>
    </div>
  </div>

  <script>
    let selectedFile = null
    const uploadCard = document.getElementById('uploadCard')
    const uploadContent = document.getElementById('uploadContent')
    const previewContainer = document.getElementById('previewContainer')
    const previewImage = document.getElementById('previewImage')
    const imageInfo = document.getElementById('imageInfo')
    const chooseBtn = document.getElementById('chooseBtn')
    const fileInput = document.getElementById('fileInput')
    const changeImageBtn = document.getElementById('changeImageBtn')

    chooseBtn.addEventListener('click', () => fileInput.click())
    changeImageBtn.addEventListener('click', () => fileInput.click())
    fileInput.addEventListener('change', handleFileSelect)

    uploadCard.addEventListener('dragover', (e) => {
      e.preventDefault()
      uploadCard.classList.add('ring-2','ring-[#4caf50]','bg-emerald-50')
    })
    uploadCard.addEventListener('dragleave', () => {
      uploadCard.classList.remove('ring-2','ring-[#4caf50]','bg-emerald-50')
    })
    uploadCard.addEventListener('drop', (e) => {
      e.preventDefault()
      uploadCard.classList.remove('ring-2','ring-[#4caf50]','bg-emerald-50')
      const files = e.dataTransfer.files
      if (files.length > 0) handleFile(files[0])
    })

    async function loadTrees() {
      try {
        const res = await fetch('/api/tracked-trees')
        const data = await res.json()
        if (data.trees) {
          const sel = document.getElementById('treeSelect')
          data.trees.forEach(t => {
            const opt = document.createElement('option')
            opt.value = t.tree_id
            opt.textContent = `Tree ${t.tree_id.substring(0,8)}... (${t.species}, ${t.count} records)`
            sel.appendChild(opt)
          })
        }
      } catch (e) { console.error('Failed to load trees', e) }
    }
    loadTrees()

    function handleFileSelect(event) {
      const file = event.target.files[0]
      if (file) handleFile(file)
    }

    function handleFile(file) {
      if (!file.type.startsWith('image/')) { showError('Please select a valid image file.') ; return }
      selectedFile = file
      const reader = new FileReader()
      reader.onload = function(e) {
        previewImage.src = e.target.result
        uploadContent.classList.add('hidden')
        previewContainer.classList.remove('hidden')
        const fileSize = (file.size / 1024 / 1024).toFixed(2)
        imageInfo.textContent = `${file.name} (${fileSize} MB)`
      }
      reader.readAsDataURL(file)
      document.getElementById('resultsSection').classList.add('hidden')
      document.getElementById('errorMessage').classList.add('hidden')
    }

    async function callModelAPI(imageData) {
      const treeId = document.getElementById('treeSelect').value
      const response = await fetch('/predict', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: imageData, tree_id: treeId })
      })
      if (!response.ok) { const e = await response.json() ; throw new Error(e.error || `HTTP ${response.status}`) }
      return await response.json()
    }

    function getImageAsBase64(file) {
      return new Promise((resolve, reject) => { const r = new FileReader() ; r.onload = () => resolve(r.result) ; r.onerror = reject ; r.readAsDataURL(file) })
    }

    async function analyzeTree() {
      if (!selectedFile) { showError('Please select an image first.') ; return }
      document.getElementById('loadingSection').classList.remove('hidden')
      document.getElementById('resultsSection').classList.add('hidden')
      document.getElementById('errorMessage').classList.add('hidden')
      try {
        const imageBase64 = await getImageAsBase64(selectedFile)
        const result = await callModelAPI(imageBase64)
        if (result.success) { displayResults(result) ; document.getElementById('loadingSection').classList.add('hidden') ; document.getElementById('resultsSection').classList.remove('hidden') }
        else { throw new Error('Prediction failed') }
      } catch (error) {
        document.getElementById('loadingSection').classList.add('hidden')
        showError(`Analysis failed: ${error.message}`)
      }
    }

    function getAgeClassName(category) { const n = category.toLowerCase().replace(/\s+/g,'-') ; return `age-${n}` }

    function displayResults(result) {
      const speciesElement = document.getElementById('speciesResult')
      const confidenceElement = document.getElementById('confidenceResult')
      speciesElement.textContent = result.species.species
      speciesElement.className = 'mt-2 text-xl font-bold ' + (result.species.is_bagtikan ? 'text-[#2e7d32]' : 'text-gray-800')
      const conf = (typeof result.species.confidence === 'number') ? `${(result.species.confidence * 100).toFixed(1)}%` : '—'
      confidenceElement.textContent = `Confidence: ${conf}`
      const ageCard = document.getElementById('ageCard')
      const ageElement = document.getElementById('ageResult')
      const ageDetailsElement = document.getElementById('ageDetails')
      if (result.age) {
        ageCard.classList.remove('hidden')
        if (result.age.estimated_age) { ageElement.textContent = `~${result.age.estimated_age} years (${result.age.category})` } else { ageElement.textContent = result.age.category }
        ageElement.className = 'mt-2 text-xl font-bold ' + (result.age.category === 'Seedling' ? 'text-teal-600' : result.age.category === 'Young' ? 'text-[#2e7d32]' : result.age.category === 'Middle age' ? 'text-amber-600' : 'text-purple-600')
        ageDetailsElement.innerHTML = `
          <div class="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-3">
            <div class="bg-white rounded-lg border-l-4 border-[#4caf50] p-3">
              <div class="text-sm font-semibold text-[#2e7d32]">Estimated Age</div>
              <div class="text-lg font-bold text-gray-800">${result.age.estimated_age} years</div>
            </div>
            <div class="bg-white rounded-lg border-l-4 border-teal-500 p-3">
              <div class="text-sm font-semibold text-teal-600">Age Category</div>
              <div class="text-gray-800">${result.age.category}</div>
            </div>
          </div>
          <div class="rounded-lg bg-gray-50 p-3">
            <div class="text-gray-700"><span class="font-semibold">Age Range:</span> ${result.age.age_range || 'Not specified'}</div>
            <div class="text-gray-700"><span class="font-semibold">Visual Features:</span> edge=${result.age.visual_features?.edge_score}, green=${result.age.visual_features?.green_ratio}</div>
          </div>
          <div class="mt-2 text-xs text-gray-500">${result.age.data_source || 'Scientific dataset analysis'}</div>
          <div class="mt-3">
            <a href="${result.detail_url}" class="inline-flex items-center rounded-lg bg-[#2e7d32] px-3 py-2 text-white font-semibold shadow hover:bg-[#256b2b]">View Details</a>
            <a href="/dashboard" class="ml-2 inline-flex items-center rounded-lg bg-gray-100 px-3 py-2 text-gray-700 hover:bg-gray-200">Open Dashboard</a>
            <button onclick="addLocation('${result.analysis_id}', '${result.map_url}')" class="ml-2 inline-flex items-center rounded-lg bg-emerald-600 px-3 py-2 text-white font-semibold shadow hover:bg-emerald-700">Add Location</button>
          </div>
        `
      } else if (result.species.is_bagtikan) {
        ageCard.classList.remove('hidden')
        ageElement.textContent = 'Age estimation unavailable for this image'
        ageElement.className = 'mt-2 text-xl font-bold text-gray-800'
        ageDetailsElement.innerHTML = '<em class="text-gray-600">Upload a clear tree image to estimate age</em>'
      } else {
        ageCard.classList.add('hidden')
      }
      const xaiCard = document.getElementById('xaiCard')
      const imgOrig = document.getElementById('xaiImageOriginal')
      const imgHeat = document.getElementById('xaiImageHeatmap')
      const xaiText = document.getElementById('xaiText')
      const xaiConf = document.getElementById('xaiConf')
      imgOrig.src = result.file_url || ''
      const hm = result.xai?.heatmap || null
      if (hm) { imgHeat.src = hm }
      imgOrig.classList.remove('hidden')
      imgHeat.classList.add('hidden')
      xaiText.textContent = result.xai?.explanation || ''
      xaiConf.textContent = result.xai?.confidence_level || ''
      const showOriginalBtn = document.getElementById('showOriginalBtn')
      const showHeatmapBtn = document.getElementById('showHeatmapBtn')
      showOriginalBtn.onclick = () => { imgOrig.classList.remove('hidden'); imgHeat.classList.add('hidden') }
      showHeatmapBtn.onclick = () => { imgHeat.classList.remove('hidden'); imgOrig.classList.add('hidden') }
    }

    async function addLocation(id, mapUrl) {
      // Redirect to map for precise pin-drop; geolocation will only help position
      window.location.href = mapUrl
    }

    function showError(message) {
      const e = document.getElementById('errorMessage')
      e.textContent = message
      e.classList.remove('hidden')
      setTimeout(() => { e.classList.add('hidden') }, 5000)
    }

    document.getElementById('analyzeBtn').addEventListener('click', analyzeTree)
  </script>
</body>
</html>'''
    return html_content


@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route('/dashboard', methods=['GET'])
def dashboard():
    ensure_storage()
    items = load_history()
    html = r'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Analysis Dashboard</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Poppins:wght@400;600;700&display=swap" rel="stylesheet">
  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    tailwind.config = {
      theme: {
        extend: {
          colors: { forest: { 700: '#2e7d32', 500: '#4caf50', 300: '#a5d6a7' } },
          fontFamily: { sans: ['Inter','Poppins','ui-sans-serif','system-ui'] }
        }
      }
    }
  </script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.css"/>
  <link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.Default.css"/>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://unpkg.com/leaflet.heat@0.2.0/dist/leaflet-heat.js"></script>
  <script src="https://unpkg.com/leaflet.markercluster@1.5.3/dist/leaflet.markercluster.js"></script>
</head>
<body class="min-h-screen bg-gradient-to-br from-green-50 to-emerald-100 font-sans">
  <header class="sticky top-0 z-50 bg-gradient-to-r from-[#1b5e20] to-[#2e7d32] text-white shadow-lg border-b border-white/10 backdrop-blur-sm bg-opacity-95">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="h-16 flex items-center justify-between">
        <a href="/" class="flex items-center gap-3 group">
          <span class="inline-grid place-items-center h-10 w-10 rounded-xl bg-white/10 group-hover:bg-white/20 transition duration-300">
            <svg class="h-6 w-6" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 2C7 2 4 7 4 11C4 16 8 20 12 22C16 20 20 16 20 11C20 7 17 2 12 2Z" fill="white" fill-opacity="0.9"/></svg>
          </span>
          <div class="flex flex-col">
            <span class="text-base font-bold tracking-wide leading-tight">BagtikanAI</span>
            <span class="text-[10px] text-emerald-100 uppercase tracking-wider font-medium">Research Dashboard v2.0</span>
          </div>
        </a>

        <div class="hidden md:flex items-center gap-1 bg-white/10 rounded-xl p-1 backdrop-blur-md">
          <a href="/" class="px-4 py-2 rounded-lg hover:bg-white/20 transition text-sm font-medium flex items-center gap-2">
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z"></path><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 13a3 3 0 11-6 0 3 3 0 016 0z"></path></svg>
            Analyze
          </a>
          <a href="/dashboard" class="px-4 py-2 rounded-lg bg-white/20 shadow-sm text-sm font-bold flex items-center gap-2 ring-1 ring-white/20">
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z"></path></svg>
            Dashboard
          </a>
          <a href="/map" class="px-4 py-2 rounded-lg hover:bg-white/20 transition text-sm font-medium flex items-center gap-2">
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7"></path></svg>
            Map
          </a>
        </div>

        <div class="flex items-center gap-3">
           <div class="hidden sm:flex items-center gap-2 px-3 py-1.5 bg-emerald-900/30 rounded-full border border-emerald-400/30">
              <span class="relative flex h-2 w-2">
                <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                <span class="relative inline-flex rounded-full h-2 w-2 bg-emerald-400"></span>
              </span>
              <span class="text-xs font-medium text-emerald-100">System Online</span>
           </div>
           <button id="navToggle" class="md:hidden inline-flex items-center rounded-lg p-2 bg-white/10 hover:bg-white/20 transition">
              <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16m-7 6h7"></path></svg>
           </button>
        </div>
      </div>
    </div>
    <!-- Mobile Menu -->
    <div id="navMenu" class="hidden md:hidden border-t border-white/10 bg-[#1b5e20]">
      <div class="px-4 pt-2 pb-4 space-y-1">
        <a href="/" class="block px-3 py-2 rounded-lg hover:bg-white/10 text-base font-medium">Analyze Tree</a>
        <a href="/dashboard" class="block px-3 py-2 rounded-lg bg-white/20 text-base font-bold">Dashboard</a>
        <a href="/map" class="block px-3 py-2 rounded-lg hover:bg-white/10 text-base font-medium">Tree Map</a>
        <a href="/training" class="block px-3 py-2 rounded-lg hover:bg-white/10 text-base font-medium">Training</a>
        <a href="/about" class="block px-3 py-2 rounded-lg hover:bg-white/10 text-base font-medium">About</a>
      </div>
    </div>
  </header>
  <script>const t=document.getElementById('navToggle');const m=document.getElementById('navMenu');if(t) t.addEventListener('click',()=>{m.classList.toggle('hidden')})</script>
  
  <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
    <div class="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-8">
      <div>
        <h1 class="text-3xl font-bold text-gray-900 tracking-tight">Analysis Dashboard</h1>
        <p class="mt-1 text-sm text-gray-500 flex items-center gap-2">
           <svg class="w-4 h-4 text-emerald-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path></svg>
           Manage and monitor your Bagtikan tree analysis data
        </p>
      </div>
      <div class="flex items-center gap-3">
         <div class="relative group">
            <button class="inline-flex items-center rounded-xl bg-white border border-gray-200 px-4 py-2 text-gray-700 font-medium shadow-sm hover:bg-gray-50 transition">
               <span>Quick Actions</span>
               <svg class="w-4 h-4 ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
            </button>
            <div class="absolute right-0 mt-2 w-48 bg-white rounded-xl shadow-xl border border-gray-100 hidden group-hover:block z-50 overflow-hidden animate-fade-in-down">
               <a href="/" class="block px-4 py-2.5 text-sm text-gray-700 hover:bg-gray-50 hover:text-forest-700 transition">Analyze New Tree</a>
               <a href="/map" class="block px-4 py-2.5 text-sm text-gray-700 hover:bg-gray-50 hover:text-forest-700 transition">Open Full Map</a>
               <button onclick="alert('Export feature coming soon!')" class="block w-full text-left px-4 py-2.5 text-sm text-gray-700 hover:bg-gray-50 hover:text-forest-700 transition">Export CSV Data</button>
            </div>
         </div>
         <a href="/" class="inline-flex items-center rounded-xl bg-forest-700 px-5 py-2 text-white font-semibold shadow-md hover:bg-[#1b5e20] hover:shadow-lg transition transform hover:-translate-y-0.5">
            <svg class="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4"></path></svg>
            New Analysis
         </a>
      </div>
    </div>

    <!-- Summary Metrics Cards -->
    <div id="statsRow" class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
      <!-- Total Trees Card -->
      <div class="group relative overflow-hidden rounded-2xl bg-white p-5 shadow-sm border border-gray-100 hover:shadow-md hover:border-emerald-200 transition-all duration-300 cursor-pointer" onclick="document.getElementById('filterSpecies').value=''; render();">
        <div class="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition">
           <svg class="w-16 h-16 text-emerald-600" fill="currentColor" viewBox="0 0 24 24"><path d="M5 3h14v2H5zM5 7h14v2H5zM5 11h14v2H5zM5 15h14v2H5zM5 19h14v2H5z"/></svg>
        </div>
        <div class="flex items-center gap-3 mb-1">
          <div class="p-2 bg-emerald-50 rounded-lg group-hover:bg-emerald-100 transition">
             <svg class="w-5 h-5 text-emerald-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path></svg>
          </div>
          <span class="text-sm font-medium text-gray-500">Total Analyzed</span>
        </div>
        <div class="flex items-baseline gap-2">
           <span id="statTotal" class="text-3xl font-bold text-gray-900">0</span>
           <span class="text-xs font-medium text-emerald-600 bg-emerald-50 px-1.5 py-0.5 rounded-full">Trees</span>
        </div>
      </div>

      <!-- Bagtikan Detected Card -->
      <div class="group relative overflow-hidden rounded-2xl bg-white p-5 shadow-sm border border-gray-100 hover:shadow-md hover:border-emerald-200 transition-all duration-300 cursor-pointer" onclick="document.getElementById('filterSpecies').value='Bagtikan'; render();">
        <div class="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition">
           <svg class="w-16 h-16 text-emerald-600" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2C7 2 4 7 4 11c0 5 4 9 8 11 4-2 8-6 8-11 0-4-3-9-8-9z"/></svg>
        </div>
        <div class="flex items-center gap-3 mb-1">
          <div class="p-2 bg-emerald-50 rounded-lg group-hover:bg-emerald-100 transition">
             <svg class="w-5 h-5 text-emerald-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z"></path></svg>
          </div>
          <span class="text-sm font-medium text-gray-500">Bagtikan Found</span>
        </div>
        <div class="flex items-baseline gap-2">
           <span id="statBagtikan" class="text-3xl font-bold text-gray-900">0</span>
           <span class="text-xs font-medium text-emerald-600 bg-emerald-50 px-1.5 py-0.5 rounded-full">Target Species</span>
        </div>
      </div>

      <!-- Avg Confidence Card -->
      <div class="group relative overflow-hidden rounded-2xl bg-white p-5 shadow-sm border border-gray-100 hover:shadow-md hover:border-emerald-200 transition-all duration-300">
        <div class="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition">
           <svg class="w-16 h-16 text-emerald-600" fill="currentColor" viewBox="0 0 24 24"><path d="M12 3a9 9 0 100 18 9 9 0 000-18z"/></svg>
        </div>
        <div class="flex items-center gap-3 mb-1">
          <div class="p-2 bg-emerald-50 rounded-lg group-hover:bg-emerald-100 transition">
             <svg class="w-5 h-5 text-emerald-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
          </div>
          <span class="text-sm font-medium text-gray-500">Avg Confidence</span>
        </div>
        <div class="flex items-baseline gap-2">
           <span id="statAvgConf" class="text-3xl font-bold text-gray-900">—</span>
           <span class="text-xs font-medium text-emerald-600 bg-emerald-50 px-1.5 py-0.5 rounded-full">Reliability</span>
        </div>
      </div>

      <!-- Trees Mapped Card -->
      <div class="group relative overflow-hidden rounded-2xl bg-white p-5 shadow-sm border border-gray-100 hover:shadow-md hover:border-emerald-200 transition-all duration-300">
        <div class="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition">
           <svg class="w-16 h-16 text-emerald-600" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2l3 7h7l-5.5 4 2 7-6.5-4.5L5.5 20l2-7L2 9h7z"/></svg>
        </div>
        <div class="flex items-center gap-3 mb-1">
          <div class="p-2 bg-emerald-50 rounded-lg group-hover:bg-emerald-100 transition">
             <svg class="w-5 h-5 text-emerald-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z"></path><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z"></path></svg>
          </div>
          <span class="text-sm font-medium text-gray-500">Geotagged</span>
        </div>
        <div class="flex items-baseline gap-2">
           <span id="statMapped" class="text-3xl font-bold text-gray-900">0</span>
           <span class="text-xs font-medium text-emerald-600 bg-emerald-50 px-1.5 py-0.5 rounded-full">Locations</span>
        </div>
      </div>
    </div>

    <!-- Tabs -->
    <div class="mt-8 flex space-x-1 rounded-xl bg-gray-100 p-1 mb-6 w-fit">
      <button id="btnTabHistory" class="rounded-lg bg-white px-4 py-2 text-sm font-semibold text-forest-700 shadow transition">Analysis History</button>
      <button id="btnTabTracking" class="rounded-lg px-4 py-2 text-sm font-semibold text-gray-600 hover:bg-white/50 hover:text-gray-800 transition">Tracked Trees</button>
    </div>

    <!-- Tab: History -->
    <div id="tabHistory">
    <div class="mt-6 rounded-2xl bg-white shadow-sm border border-gray-200 overflow-hidden">
      <button id="toggleFilters" class="w-full flex items-center justify-between p-4 bg-gray-50 hover:bg-gray-100 transition">
         <div class="flex items-center gap-2">
            <svg class="w-5 h-5 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z"></path></svg>
            <span class="font-semibold text-gray-700">Filter & Search</span>
         </div>
         <svg id="filterChevron" class="w-5 h-5 text-gray-400 transform transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
      </button>
      
      <div id="filterPanel" class="p-4 border-t border-gray-200">
        <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
          <!-- Species Search -->
          <div>
            <label class="block text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">Species</label>
            <div class="relative">
               <svg class="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
               <input id="filterSpecies" type="text" class="w-full pl-9 pr-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 text-sm" placeholder="Search species...">
            </div>
          </div>

          <!-- Age Category Pills -->
          <div>
             <label class="block text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">Age Category</label>
             <div class="flex flex-wrap gap-2" id="ageFilterContainer">
                <button class="age-filter-btn px-3 py-1.5 rounded-full text-xs font-medium bg-emerald-100 text-emerald-800 border border-emerald-200 ring-2 ring-emerald-500 transition" data-value="">All</button>
                <button class="age-filter-btn px-3 py-1.5 rounded-full text-xs font-medium bg-white text-gray-600 border border-gray-200 hover:bg-gray-50 transition" data-value="Seedling">Seedling</button>
                <button class="age-filter-btn px-3 py-1.5 rounded-full text-xs font-medium bg-white text-gray-600 border border-gray-200 hover:bg-gray-50 transition" data-value="Young">Young</button>
                <button class="age-filter-btn px-3 py-1.5 rounded-full text-xs font-medium bg-white text-gray-600 border border-gray-200 hover:bg-gray-50 transition" data-value="Middle age">Middle</button>
                <button class="age-filter-btn px-3 py-1.5 rounded-full text-xs font-medium bg-white text-gray-600 border border-gray-200 hover:bg-gray-50 transition" data-value="Mature">Mature</button>
             </div>
             <input type="hidden" id="filterAge" value="">
          </div>

          <!-- Confidence Range -->
          <div>
             <div class="flex items-center justify-between mb-2">
                <label class="block text-xs font-semibold text-gray-500 uppercase tracking-wide">Min Confidence</label>
                <span id="confValueDisplay" class="text-xs font-mono font-bold text-emerald-600">0%</span>
             </div>
             <input type="range" id="filterConf" min="0" max="100" value="0" class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-emerald-600">
             <div class="flex justify-between text-[10px] text-gray-400 mt-1">
                <span>0%</span>
                <span>50%</span>
                <span>100%</span>
             </div>
          </div>
        </div>
      </div>
    </div>

    <div class="mt-3 border-t border-gray-200"></div>
    <div id="grid" class="mt-6 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6"></div>
    </div> <!-- End tabHistory -->

    <!-- Tab: Tracking -->
    <div id="tabTracking" class="hidden">
         <div class="mt-6 mb-4">
            <h2 class="text-xl font-bold text-gray-800">Tracked Trees</h2>
            <p class="text-sm text-gray-600">Select a tree to view its growth timeline and longitudinal analysis.</p>
         </div>
         <div id="trackingGrid" class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6"></div>
    </div>
  </div>

  <div id="confirmModal" class="hidden fixed inset-0 z-50 bg-black/40 backdrop-blur-sm">
    <div class="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-[95%] sm:w-[480px] rounded-2xl bg-white shadow-xl">
      <div class="p-5">
        <div class="flex items-start gap-3">
          <img id="modalThumb" class="h-16 w-16 object-cover rounded-lg border border-gray-200" alt="thumb">
          <div>
            <div class="text-lg font-semibold text-gray-800">Confirm Delete</div>
            <div class="mt-1 text-sm text-gray-600">This action will remove the analysis entry and its uploaded image. This cannot be undone.</div>
          </div>
        </div>
        <div class="mt-4 flex justify-end gap-2">
          <button id="btnCancel" class="inline-flex items-center rounded-lg bg-gray-100 px-3 py-2 text-gray-700 hover:bg-gray-200">Cancel</button>
          <button id="btnConfirm" class="inline-flex items-center rounded-lg bg-red-600 px-3 py-2 text-white hover:bg-red-700">Confirm Delete</button>
        </div>
      </div>
    </div>
  </div>

  <div class="max-w-6xl mx-auto px-4">
    <div class="mt-6 grid grid-cols-1 lg:grid-cols-3 gap-4">
      <div class="lg:col-span-2 rounded-2xl bg-white shadow-md overflow-hidden">
        <div id="dashMap" style="height:60vh"></div>
      </div>
      <div class="rounded-2xl bg-white shadow-md p-4">
        <div class="text-lg font-semibold text-forest-700">Map Analytics</div>
        <div class="mt-3 space-y-3">
          <label class="flex items-center justify-between">
            <span class="text-sm text-gray-700">Density Heatmap</span>
            <input id="dashToggleHeat" type="checkbox" class="h-4 w-4 rounded border-gray-300 text-forest-700 focus:ring-forest-700">
          </label>
          <label class="flex items-center justify-between">
            <span class="text-sm text-gray-700">Age Distribution</span>
            <input id="dashToggleAge" type="checkbox" class="h-4 w-4 rounded border-gray-300 text-forest-700 focus:ring-forest-700">
          </label>
          <label class="flex items-center justify-between">
            <span class="text-sm text-gray-700">Species Clusters</span>
            <input id="dashToggleCluster" type="checkbox" class="h-4 w-4 rounded border-gray-300 text-forest-700 focus:ring-forest-700" checked>
          </label>
        </div>
        <div class="mt-4 rounded-lg bg-[#a5d6a7]/30 p-3 text-sm text-gray-700">
          <div class="font-semibold text-forest-700">Region Stats</div>
          <div id="dashRegionStats" class="mt-2">Click a cluster to view aggregated statistics.</div>
        </div>
      </div>
    </div>
  </div>

  <div class="max-w-6xl mx-auto px-4 mt-8">
    <div class="flex items-center justify-between">
      <div>
        <div class="text-2xl font-bold text-forest-700">Dashboard Analytics</div>
        <div class="text-sm text-gray-600">Interactive charts for age, confidence, species, and trends</div>
      </div>
      <div class="flex items-center gap-2">
        <input id="dateStart" type="date" class="rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm">
        <span class="text-gray-500">to</span>
        <input id="dateEnd" type="date" class="rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm">
      </div>
    </div>
    <div class="mt-4 grid grid-cols-1 lg:grid-cols-2 gap-4">
      <div class="rounded-2xl bg-white p-4 shadow-md">
        <div class="text-lg font-semibold text-gray-800">Age Category Distribution</div>
        <canvas id="ageChart"></canvas>
      </div>
      <div class="rounded-2xl bg-white p-4 shadow-md">
        <div class="text-lg font-semibold text-gray-800">Confidence Score Distribution</div>
        <canvas id="confChart"></canvas>
      </div>
      <div class="rounded-2xl bg-white p-4 shadow-md">
        <div class="text-lg font-semibold text-gray-800">Species Distribution</div>
        <canvas id="speciesChart"></canvas>
      </div>
      <div class="rounded-2xl bg-white p-4 shadow-md">
        <div class="text-lg font-semibold text-gray-800">Analysis Trends Over Time</div>
        <canvas id="trendChart"></canvas>
      </div>
    </div>
  </div>

  <script>
    const items = JSON.parse(atob('%DATA%'))
    const grid = document.getElementById('grid')
    const filterSpecies = document.getElementById('filterSpecies')
    const filterAgeInput = document.getElementById('filterAge') // Renamed for clarity
    const filterConf = document.getElementById('filterConf')
    const confValueDisplay = document.getElementById('confValueDisplay')
    const statTotal = document.getElementById('statTotal')
    const statBagtikan = document.getElementById('statBagtikan')
    const statAvgConf = document.getElementById('statAvgConf')
    const statMapped = document.getElementById('statMapped')
    const confirmModal = document.getElementById('confirmModal')
    const modalThumb = document.getElementById('modalThumb')
    const btnCancel = document.getElementById('btnCancel')
    const btnConfirm = document.getElementById('btnConfirm')
    let pendingDeleteId = null

    // Filter UI Logic
    const toggleFilters = document.getElementById('toggleFilters')
    const filterPanel = document.getElementById('filterPanel')
    const filterChevron = document.getElementById('filterChevron')
    let filtersOpen = true

    if(toggleFilters) {
        toggleFilters.addEventListener('click', () => {
           filtersOpen = !filtersOpen
           if(filtersOpen) {
              filterPanel.classList.remove('hidden')
              filterChevron.classList.remove('rotate-180')
           } else {
              filterPanel.classList.add('hidden')
              filterChevron.classList.add('rotate-180')
           }
        })
    }

    // Age Pills Logic
    const ageBtns = document.querySelectorAll('.age-filter-btn')
    ageBtns.forEach(btn => {
       btn.addEventListener('click', () => {
          ageBtns.forEach(b => {
             b.classList.remove('bg-emerald-100', 'text-emerald-800', 'border-emerald-200', 'ring-2', 'ring-emerald-500')
             b.classList.add('bg-white', 'text-gray-600', 'border-gray-200', 'hover:bg-gray-50')
          })
          btn.classList.remove('bg-white', 'text-gray-600', 'border-gray-200', 'hover:bg-gray-50')
          btn.classList.add('bg-emerald-100', 'text-emerald-800', 'border-emerald-200', 'ring-2', 'ring-emerald-500')
          
          filterAgeInput.value = btn.dataset.value
          render()
          if(typeof updateCharts === 'function') updateCharts()
       })
    })

    // Confidence Slider Logic
    if(filterConf) {
        filterConf.addEventListener('input', () => {
           confValueDisplay.textContent = filterConf.value + '%'
           render()
           if(typeof updateCharts === 'function') updateCharts()
        })
    }

    // Tabs
    const btnTabHistory = document.getElementById('btnTabHistory');
    const btnTabTracking = document.getElementById('btnTabTracking');
    const tabHistory = document.getElementById('tabHistory');
    const tabTracking = document.getElementById('tabTracking');

    if(btnTabHistory && btnTabTracking) {
        btnTabHistory.addEventListener('click', () => {
            tabHistory.classList.remove('hidden');
            tabTracking.classList.add('hidden');
            btnTabHistory.classList.add('bg-white', 'text-forest-700', 'shadow');
            btnTabHistory.classList.remove('text-gray-600', 'hover:bg-white/50');
            btnTabTracking.classList.remove('bg-white', 'text-forest-700', 'shadow');
            btnTabTracking.classList.add('text-gray-600', 'hover:bg-white/50');
        });

        btnTabTracking.addEventListener('click', () => {
            tabHistory.classList.add('hidden');
            tabTracking.classList.remove('hidden');
            btnTabTracking.classList.add('bg-white', 'text-forest-700', 'shadow');
            btnTabTracking.classList.remove('text-gray-600', 'hover:bg-white/50');
            btnTabHistory.classList.remove('bg-white', 'text-forest-700', 'shadow');
            btnTabHistory.classList.add('text-gray-600', 'hover:bg-white/50');
            loadTrackedTrees();
        });
    }

    async function loadTrackedTrees() {
        const container = document.getElementById('trackingGrid');
        container.innerHTML = '<div class="col-span-full text-center py-8 text-gray-500">Loading tracked trees...</div>';
        
        try {
            const res = await fetch('/api/tracked-trees');
            const data = await res.json();
            container.innerHTML = '';
            
            if (!data.trees || data.trees.length === 0) {
                container.innerHTML = '<div class="col-span-full text-center py-8 text-gray-500">No tracked trees found. Start a new analysis and select "New Tree" or an existing tree to track it.</div>';
                return;
            }

            data.trees.forEach(tree => {
                const thumb = tree.latest_image ? `/uploads/${tree.latest_image}` : '';
                const el = document.createElement('div');
                el.className = 'group rounded-2xl border border-gray-200 bg-white overflow-hidden shadow-sm hover:shadow-lg hover:-translate-y-0.5 transition transform';
                el.innerHTML = `
                    <div class="relative bg-gray-100 h-48">
                        ${thumb ? `<img src="${thumb}" class="h-full w-full object-cover">` : '<div class="h-full w-full flex items-center justify-center text-gray-400">No Image</div>'}
                        <div class="absolute inset-0 bg-gradient-to-t from-black/70 via-black/20 to-transparent"></div>
                        <div class="absolute bottom-3 left-4 text-white">
                            <div class="font-bold text-lg leading-tight">Tree ${tree.tree_id.substring(0,8)}...</div>
                            <div class="text-xs opacity-90 font-medium mt-1">${tree.count} Records</div>
                        </div>
                    </div>
                    <div class="p-4">
                        <div class="flex justify-between items-center mb-3">
                            <span class="text-xs font-bold uppercase tracking-wider text-forest-700 bg-forest-50 px-2 py-1 rounded">${tree.species || 'Unknown'}</span>
                            <span class="text-xs text-gray-500">${tree.last_updated ? new Date(tree.last_updated).toLocaleDateString() : '—'}</span>
                        </div>
                        <div class="flex items-center justify-between text-sm text-gray-700 mb-4 border-b border-gray-100 pb-3">
                            <span>Latest Age:</span>
                            <span class="font-semibold text-gray-900">${tree.latest_age ? tree.latest_age + ' yrs' : 'Unknown'}</span>
                        </div>
                        <a href="/tracking/${tree.tree_id}" class="block w-full text-center rounded-lg bg-forest-700 px-3 py-2 text-white font-semibold shadow hover:bg-[#256b2b] transition flex items-center justify-center gap-2">
                            <span>View Growth Timeline</span>
                            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7l5 5m0 0l-5 5m5-5H6"></path></svg>
                        </a>
                    </div>
                `;
                container.appendChild(el);
            });
        } catch (e) {
            console.error(e);
            container.innerHTML = '<div class="col-span-full text-center py-8 text-red-500">Failed to load tracked trees.</div>';
        }
    }

    function animateNumber(el, target, duration=600) {
      const start = 0
      const diff = target - start
      const step = Math.max(16, duration / 40)
      let cur = start
      const t = setInterval(() => {
        cur += diff / (duration / step)
        if ((diff > 0 && cur >= target) || (diff < 0 && cur <= target)) { cur = target ; clearInterval(t) }
        el.textContent = Math.round(cur)
      }, step)
    }

    function updateStats() {
      const total = items.length
      const bagtikan = items.filter(x => x.is_bagtikan).length
      const confs = items.map(x => x.confidence).filter(v => typeof v === 'number')
      const avgConf = confs.length ? (confs.reduce((a,b)=>a+b,0) / confs.length) : null
      const mapped = items.filter(x => !!x.coords).length
      animateNumber(statTotal, total)
      animateNumber(statBagtikan, bagtikan)
      statAvgConf.textContent = (typeof avgConf==='number') ? (avgConf*100).toFixed(1)+'%' : '—'
      animateNumber(statMapped, mapped)
    }

    function ageBadgeClass(cat) {
      const m = {
        'Seedling': 'bg-teal-50 text-teal-700 border border-teal-200',
        'Young': 'bg-emerald-50 text-emerald-700 border border-emerald-200',
        'Middle age': 'bg-amber-50 text-amber-700 border border-amber-200',
        'Mature': 'bg-purple-50 text-purple-700 border border-purple-200'
      }
      return m[cat] || 'bg-gray-50 text-gray-700 border border-gray-200'
    }

    function openDelete(id, fileName) {
      pendingDeleteId = id
      modalThumb.src = fileName ? `/uploads/${fileName}` : ''
      confirmModal.classList.remove('hidden')
    }
    function closeDelete() {
      pendingDeleteId = null
      confirmModal.classList.add('hidden')
    }
    btnCancel.addEventListener('click', closeDelete)
    btnConfirm.addEventListener('click', async () => {
      if (!pendingDeleteId) return
      try {
        const res = await fetch(`/history/${pendingDeleteId}`, { method: 'DELETE' })
        const data = await res.json()
        if (!res.ok || !data.success) { alert(data.error || 'Failed to delete') ; return }
        const idx = items.findIndex(x => x.id === pendingDeleteId)
        if (idx >= 0) { items.splice(idx, 1) }
        closeDelete()
        render()
        updateStats()
      } catch (e) { alert('Delete failed') }
    })

    function render() {
      grid.innerHTML = ''
      const fs = (filterSpecies.value || '').toLowerCase()
      const fa = filterAgeInput.value || ''
      const fc = parseInt(filterConf ? filterConf.value : 0) || 0

      items.filter(x => {
         const confPct = x.confidence != null ? Math.round(x.confidence * 100) : 0
         return (!fs || (x.species||'').toLowerCase().includes(fs)) && 
                (!fa || (x.age?.category||'')===fa) &&
                (confPct >= fc)
      }).forEach(x => {
        const confPct = x.confidence!=null ? Math.round(x.confidence*100) : null
        const confText = (confPct!=null) ? confPct.toFixed(0)+'%' : '—'
        const ageCat = x.age?.category || '—'
        const thumb = x.file_name ? `/uploads/${x.file_name}` : ''
        const hasMap = !!x.coords
        
        let reliability = 'Low'
        let relColor = 'bg-red-500'
        if (confPct >= 85) { reliability = 'High'; relColor = 'bg-emerald-500' }
        else if (confPct >= 70) { reliability = 'Moderate'; relColor = 'bg-amber-500' }

        const el = document.createElement('div')
        el.className = 'group flex flex-col rounded-2xl border border-gray-200 bg-white overflow-hidden shadow-sm hover:shadow-xl hover:-translate-y-1 transition-all duration-300'
        el.innerHTML = `
          <div class="relative bg-gray-100 aspect-[4/3] overflow-hidden">
            <img src="${thumb}" alt="${x.file_name}" class="h-full w-full object-cover transition duration-700 group-hover:scale-110">
            <div class="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent opacity-60"></div>
            
            <div class="absolute top-3 right-3 flex gap-2 opacity-0 group-hover:opacity-100 transition duration-300 transform translate-y-2 group-hover:translate-y-0">
               <a href="/details/${x.id}" class="p-2 bg-white/90 rounded-lg text-gray-700 hover:text-emerald-600 shadow-sm backdrop-blur-sm transition" title="View Details">
                  <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"></path><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"></path></svg>
               </a>
               <a href="/map?id=${x.id}" class="p-2 bg-white/90 rounded-lg text-gray-700 hover:text-emerald-600 shadow-sm backdrop-blur-sm transition" title="View on Map">
                  <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7"></path></svg>
               </a>
            </div>

            <div class="absolute bottom-3 left-3 right-3 text-white">
               <div class="flex items-center justify-between">
                  <span class="text-xs font-medium bg-black/30 backdrop-blur-sm px-2 py-1 rounded-lg border border-white/10">${new Date(x.uploaded_at).toLocaleDateString()}</span>
                  <span class="text-xs font-bold px-2 py-1 rounded-lg ${relColor}/90 text-white shadow-sm backdrop-blur-md">${reliability}</span>
               </div>
            </div>
          </div>
          
          <div class="p-5 flex-1 flex flex-col">
            <div class="flex items-start justify-between mb-2">
               <div>
                  <h3 class="text-lg font-bold text-gray-900 group-hover:text-emerald-700 transition line-clamp-1">${x.species||'Unknown'}</h3>
                  <div class="text-sm text-gray-500 mt-0.5">${ageCat} Stage</div>
               </div>
               ${hasMap ? '<span class="flex h-2 w-2 rounded-full bg-emerald-500 ring-2 ring-emerald-100" title="Geotagged"></span>' : '<span class="flex h-2 w-2 rounded-full bg-gray-300" title="No Location"></span>'}
            </div>
            
            <div class="mt-auto pt-4 border-t border-gray-100">
               <div class="flex justify-between items-center text-sm mb-1.5">
                  <span class="text-gray-600 font-medium">Confidence Score</span>
                  <span class="font-bold text-gray-900">${confText}</span>
               </div>
               <div class="h-2.5 w-full bg-gray-100 rounded-full overflow-hidden">
                  <div class="h-full bg-gradient-to-r from-emerald-500 to-emerald-400 rounded-full transition-all duration-1000 ease-out" style="width:${confPct!=null?confPct:0}%"></div>
               </div>
               
               <div class="mt-4 flex gap-2">
                  <button onclick="openDelete('${x.id}','${x.file_name||''}')" class="flex-1 py-2 rounded-lg border border-red-100 text-red-600 text-sm font-semibold hover:bg-red-50 transition">Delete</button>
                  <a href="/details/${x.id}" class="flex-1 py-2 rounded-lg bg-emerald-50 text-emerald-700 text-sm font-semibold hover:bg-emerald-100 transition text-center">Details</a>
               </div>
            </div>
          </div>
        `
        grid.appendChild(el)
      })
    }
    if(filterSpecies) filterSpecies.addEventListener('input', () => { render(); if(typeof updateCharts === 'function') updateCharts(); })
    // Removed old filterAge listener since we use buttons now
    updateStats()
    render()

    const dashHeatToggle = document.getElementById('dashToggleHeat')
    const dashAgeToggle = document.getElementById('dashToggleAge')
    const dashClusterToggle = document.getElementById('dashToggleCluster')
    const dashRegionStats = document.getElementById('dashRegionStats')
    let dashMap = L.map('dashMap')
    let dashBase = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 19, attribution: '&copy; OpenStreetMap' })
    dashBase.addTo(dashMap)
    dashMap.setView([14.5995, 120.9842], 6)

    const dashMarkers = L.markerClusterGroup({
      showCoverageOnHover: false,
      iconCreateFunction: function(cluster) {
        const children = cluster.getAllChildMarkers()
        let bag=0, oth=0
        children.forEach(m => { if (m.options.is_bagtikan) bag++; else oth++; })
        const total = children.length
        const domIsBagtikan = bag >= oth
        const color = domIsBagtikan ? '#2e7d32' : '#64748b'
        const html = `<div style="background:${color};color:white;border-radius:9999px;display:flex;align-items:center;justify-content:center;width:32px;height:32px;font-weight:700">${total}</div>`
        return L.divIcon({ html, className: 'species-cluster-icon', iconSize: [32,32] })
      }
    })
    let dashHeatLayer = null
    let dashAgeGroup = L.layerGroup()

    function dashMarkerPopupContent(x) {
      const thumb = x.file_name ? `/uploads/${x.file_name}` : ''
      const confPct = (typeof x.confidence==='number') ? (x.confidence*100).toFixed(1)+'%' : '—'
      const ageCat = (x.age||{}).category || '—'
      return `
        <div class="p-1">
          ${thumb ? `<img src="${thumb}" class="w-32 h-20 object-cover rounded-lg mb-2" />` : ''}
          <div class="text-sm font-semibold text-forest-700">${x.species||'Unknown'}</div>
          <div class="text-xs text-gray-700">Date: ${x.uploaded_at ? new Date(x.uploaded_at).toLocaleString() : '—'}</div>
          <div class="text-xs text-gray-700">Age: ${ageCat}</div>
          <div class="text-xs text-gray-700">Confidence: ${confPct}</div>
          <div class="mt-1"><a href="/details/${x.id}" class="text-xs inline-flex items-center rounded bg-forest-700 px-2 py-1 text-white">View Details</a></div>
        </div>`
    }

    function dashUpdateRegionStats(clusterLayer) {
      const children = clusterLayer.getAllChildMarkers()
      let bag=0,oth=0; let s=0,y=0,m=0,mt=0
      children.forEach(mm=>{
        if (mm.options.is_bagtikan) bag++; else oth++;
        const cat = (mm.options.age_category||'').toLowerCase()
        if (cat.includes('seedling')) s++; else if (cat.includes('young')) y++; else if (cat.includes('middle')) m++; else if (cat.includes('mature')) mt++;
      })
      const total = children.length
      dashRegionStats.innerHTML = `
        <div class="grid grid-cols-2 gap-2">
          <div class="rounded bg-white p-2 shadow"><div class="text-xs text-gray-600">Total</div><div class="text-lg font-semibold text-forest-700">${total}</div></div>
          <div class="rounded bg-white p-2 shadow"><div class="text-xs text-gray-600">Bagtikan</div><div class="text-lg font-semibold text-forest-700">${bag}</div></div>
          <div class="rounded bg-white p-2 shadow"><div class="text-xs text-gray-600">Other</div><div class="text-lg font-semibold text-forest-700">${oth}</div></div>
          <div class="rounded bg-white p-2 shadow"><div class="text-xs text-gray-600">Seedling</div><div class="text-lg font-semibold text-forest-700">${s}</div></div>
          <div class="rounded bg-white p-2 shadow"><div class="text-xs text-gray-600">Young</div><div class="text-lg font-semibold text-forest-700">${y}</div></div>
          <div class="rounded bg-white p-2 shadow"><div class="text-xs text-gray-600">Middle</div><div class="text-lg font-semibold text-forest-700">${m}</div></div>
          <div class="rounded bg-white p-2 shadow"><div class="text-xs text-gray-600">Mature</div><div class="text-lg font-semibold text-forest-700">${mt}</div></div>
        </div>`
    }

    function dashAddMarkers(items) {
      dashMarkers.clearLayers()
      items.filter(x=>x.coords).forEach(x=>{
        const iconHtml = `<div style="background:${x.is_bagtikan ? '#2e7d32' : '#64748b'};width:10px;height:10px;border-radius:9999px;border:2px solid white;box-shadow:0 0 0 2px rgba(0,0,0,0.15)"></div>`
        const icon = L.divIcon({ html: iconHtml, className: 'tree-marker', iconSize: [10,10] })
        const m = L.marker([x.coords.lat, x.coords.lng], { icon, is_bagtikan: !!x.is_bagtikan, age_category: (x.age||{}).category || '' })
        m.bindPopup(dashMarkerPopupContent(x))
        dashMarkers.addLayer(m)
      })
      dashMarkers.on('clusterclick', e => { dashUpdateRegionStats(e.layer) })
    }

    function dashBuildHeat(items) {
      const pts = items.filter(x=>x.coords).map(x=>[x.coords.lat, x.coords.lng, 0.6])
      dashHeatLayer = L.heatLayer(pts, { radius: 25, blur: 15, minOpacity: 0.2, gradient: { 0.2: 'green', 0.5: 'yellow', 0.8: 'red' } })
    }
    function dashBuildAge(items) {
      dashAgeGroup.clearLayers()
      const bins = {}
      const res = 0.05
      items.filter(x=>x.coords).forEach(x=>{
        const lat = x.coords.lat, lng = x.coords.lng
        const key = (Math.round(lat/res)*res).toFixed(2)+','+(Math.round(lng/res)*res).toFixed(2)
        const cat = ((x.age||{}).category||'Unknown').toLowerCase()
        if (!bins[key]) bins[key] = { lat: Math.round(lat/res)*res, lng: Math.round(lng/res)*res, seedling:0, young:0, middle:0, mature:0, total:0 }
        if (cat.includes('seedling')) bins[key].seedling++
        else if (cat.includes('young')) bins[key].young++
        else if (cat.includes('middle')) bins[key].middle++
        else if (cat.includes('mature')) bins[key].mature++
        bins[key].total++
      })
      Object.values(bins).forEach(b=>{
        const baseRadius = 180
        const rSeed = baseRadius * (b.seedling / Math.max(1,b.total))
        const rYoung = baseRadius * (b.young / Math.max(1,b.total))
        const rMid = baseRadius * (b.middle / Math.max(1,b.total))
        const rMat = baseRadius * (b.mature / Math.max(1,b.total))
        if (b.seedling>0) L.circle([b.lat,b.lng], { radius: rSeed, color: '#A7F3D0', weight: 1, fillOpacity: 0.25 }).addTo(dashAgeGroup)
        if (b.young>0) L.circle([b.lat,b.lng], { radius: rYoung, color: '#6EE7B7', weight: 1, fillOpacity: 0.25 }).addTo(dashAgeGroup)
        if (b.middle>0) L.circle([b.lat,b.lng], { radius: rMid, color: '#34D399', weight: 1, fillOpacity: 0.25 }).addTo(dashAgeGroup)
        if (b.mature>0) L.circle([b.lat,b.lng], { radius: rMat, color: '#059669', weight: 1, fillOpacity: 0.25 }).addTo(dashAgeGroup)
      })
    }

    function initDashboardMapAnalytics() {
      const mappedItems = items
      dashAddMarkers(mappedItems)
      dashBuildHeat(mappedItems)
      dashBuildAge(mappedItems)
      dashMap.addLayer(dashMarkers)
      const bounds = L.latLngBounds(mappedItems.filter(x=>x.coords).map(x=>[x.coords.lat,x.coords.lng]))
      if (bounds.isValid()) dashMap.fitBounds(bounds, { padding: [20,20] })
    }
    initDashboardMapAnalytics()
    dashHeatToggle.addEventListener('change', () => {
      if (!dashHeatLayer) return
      if (dashHeatToggle.checked) { dashHeatLayer.addTo(dashMap) } else { dashMap.removeLayer(dashHeatLayer) }
    })
    dashAgeToggle.addEventListener('change', () => {
      if (dashAgeToggle.checked) { dashAgeGroup.addTo(dashMap) } else { dashMap.removeLayer(dashAgeGroup) }
    })
    dashClusterToggle.addEventListener('change', () => {
      if (dashClusterToggle.checked) { dashMap.addLayer(dashMarkers) } else { dashMap.removeLayer(dashMarkers) }
    })

    const dateStartEl = document.getElementById('dateStart')
    const dateEndEl = document.getElementById('dateEnd')
    let ageChart, confChart, speciesChart, trendChart

    function normalizeAge(cat) {
      const s = (cat||'').toLowerCase()
      if (s.includes('seedling')) return 'Seedling'
      if (s.includes('young')) return 'Young'
      if (s.includes('middle')) return 'Middle'
      if (s.includes('mature')) return 'Mature'
      return 'Unknown'
    }
    function inDateRange(d) {
      const dt = new Date(d)
      const s = dateStartEl.value ? new Date(dateStartEl.value) : null
      const e = dateEndEl.value ? new Date(dateEndEl.value) : null
      if (s && dt < s) return false
      if (e) {
        const endDay = new Date(e)
        endDay.setHours(23,59,59,999)
        if (dt > endDay) return false
      }
      return true
    }
    function filtered() {
      const fs = (filterSpecies.value || '').toLowerCase()
      const fa = filterAge.value || ''
      return items.filter(x => {
        const okDate = inDateRange(x.uploaded_at || '')
        const okSpecies = (!fs || (x.species||'').toLowerCase().includes(fs))
        const okAge = (!fa || (x.age?.category||'')===fa)
        return okDate && okSpecies && okAge
      })
    }
    function computeAgeData(arr) {
      const counts = { Seedling:0, Young:0, Middle:0, Mature:0 }
      arr.forEach(x => { const a = normalizeAge((x.age||{}).category); if (counts[a]!=null) counts[a]++ })
      const total = arr.length || 1
      const labels = Object.keys(counts)
      const data = labels.map(k => counts[k])
      const pct = labels.map(k => Math.round(100*counts[k]/total))
      return { labels, data, pct }
    }
    function computeConfData(arr) {
      const bins = { '0–50%':0, '51–70%':0, '71–85%':0, '86–100%':0 }
      arr.forEach(x => {
        const c = x.confidence
        if (typeof c !== 'number') return
        const p = c*100
        if (p <= 50) bins['0–50%']++
        else if (p <= 70) bins['51–70%']++
        else if (p <= 85) bins['71–85%']++
        else bins['86–100%']++
      })
      const labels = Object.keys(bins)
      const data = labels.map(k => bins[k])
      return { labels, data }
    }
    function computeSpeciesData(arr) {
      let bag=0, oth=0
      arr.forEach(x => { if (x.is_bagtikan) bag++; else oth++; })
      return { labels: ['Bagtikan','Other'], data: [bag, oth] }
    }
    function computeTrendData(arr) {
      const byDay = {}
      arr.forEach(x => {
        const d = new Date(x.uploaded_at || '')
        if (isNaN(d.getTime())) return
        const key = d.getFullYear()+'-'+String(d.getMonth()+1).padStart(2,'0')+'-'+String(d.getDate()).padStart(2,'0')
        byDay[key] = (byDay[key]||0) + 1
      })
      const labels = Object.keys(byDay).sort()
      const data = labels.map(k => byDay[k])
      return { labels, data }
    }
    function initCharts() {
      const ctxA = document.getElementById('ageChart').getContext('2d')
      const ctxC = document.getElementById('confChart').getContext('2d')
      const ctxS = document.getElementById('speciesChart').getContext('2d')
      const ctxT = document.getElementById('trendChart').getContext('2d')
      const age = computeAgeData(filtered())
      const conf = computeConfData(filtered())
      const spec = computeSpeciesData(filtered())
      const trend = computeTrendData(filtered())
      ageChart = new Chart(ctxA, { type:'bar', data:{ labels: age.labels, datasets:[{ label:'Count', data: age.data, backgroundColor:['#A7F3D0','#6EE7B7','#34D399','#059669'] }] }, options:{ responsive:true, plugins:{ tooltip:{ enabled:true }, legend:{ display:false } } } })
      confChart = new Chart(ctxC, { type:'bar', data:{ labels: conf.labels, datasets:[{ label:'Count', data: conf.data, backgroundColor:'#4caf50' }] }, options:{ responsive:true, plugins:{ tooltip:{ enabled:true }, legend:{ display:false } } } })
      speciesChart = new Chart(ctxS, { type:'bar', data:{ labels: spec.labels, datasets:[{ label:'Count', data: spec.data, backgroundColor:['#2e7d32','#64748b'] }] }, options:{ responsive:true, plugins:{ tooltip:{ enabled:true }, legend:{ display:false } } } })
      trendChart = new Chart(ctxT, { type:'line', data:{ labels: trend.labels, datasets:[{ label:'Analyses per day', data: trend.data, borderColor:'#2e7d32', tension:0.2 }] }, options:{ responsive:true, plugins:{ tooltip:{ enabled:true }, legend:{ position:'bottom' } }, scales:{ x:{ ticks:{ maxTicksLimit:8 } } } } })
    }
    function updateCharts() {
      const age = computeAgeData(filtered())
      const conf = computeConfData(filtered())
      const spec = computeSpeciesData(filtered())
      const trend = computeTrendData(filtered())
      ageChart.data.labels = age.labels
      ageChart.data.datasets[0].data = age.data
      ageChart.update()
      confChart.data.labels = conf.labels
      confChart.data.datasets[0].data = conf.data
      confChart.update()
      speciesChart.data.labels = spec.labels
      speciesChart.data.datasets[0].data = spec.data
      speciesChart.update()
      trendChart.data.labels = trend.labels
      trendChart.data.datasets[0].data = trend.data
      trendChart.update()
    }
    initCharts()
    dateStartEl.addEventListener('change', () => { updateCharts() })
    dateEndEl.addEventListener('change', () => { updateCharts() })
    filterSpecies.addEventListener('input', () => { updateCharts() })
    filterAge.addEventListener('change', () => { updateCharts() })
  </script>
</body>
</html>'''
    encoded = base64.b64encode(json.dumps(items).encode('utf-8')).decode('ascii')
    html = html.replace('%DATA%', encoded)
    return html


@app.route('/details/<analysis_id>', methods=['GET'])
def details(analysis_id):
    ensure_storage()
    items = load_history()
    item = None
    for x in items:
        if x.get('id') == analysis_id:
            item = x
            break
    if not item:
        return jsonify({'error': 'Not found'}), 404
    html = r'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Analysis Details | BagtikanAI</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Poppins:wght@400;600;700&display=swap" rel="stylesheet">
  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    tailwind.config = {
      theme: {
        extend: {
          colors: { forest: { 700: '#1b5e20', 500: '#2e7d32', 300: '#4caf50' } },
          fontFamily: { sans: ['Inter','Poppins','ui-sans-serif','system-ui'] },
          animation: { 'fade-in-down': 'fadeInDown 0.3s ease-out' },
          keyframes: { fadeInDown: { '0%': { opacity: '0', transform: 'translateY(-10px)' }, '100%': { opacity: '1', transform: 'translateY(0)' } } }
        }
      }
    }
  </script>
</head>
<body class="min-h-screen bg-gray-50 font-sans text-gray-900">
  <header class="sticky top-0 z-50 bg-gradient-to-r from-[#1b5e20] to-[#2e7d32] text-white shadow-lg border-b border-white/10 backdrop-blur-sm bg-opacity-95">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="h-16 flex items-center justify-between">
        <a href="/" class="flex items-center gap-3 group">
          <span class="inline-grid place-items-center h-10 w-10 rounded-xl bg-white/10 group-hover:bg-white/20 transition duration-300">
            <svg class="h-6 w-6" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 2C7 2 4 7 4 11C4 16 8 20 12 22C16 20 20 16 20 11C20 7 17 2 12 2Z" fill="white" fill-opacity="0.9"/></svg>
          </span>
          <div class="flex flex-col">
            <span class="text-base font-bold tracking-wide leading-tight">BagtikanAI</span>
            <span class="text-[10px] text-emerald-100 uppercase tracking-wider font-medium">Research Dashboard v2.0</span>
          </div>
        </a>
        <a href="/dashboard" class="px-4 py-2 rounded-lg bg-white/20 hover:bg-white/30 transition text-sm font-bold flex items-center gap-2">
           <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"></path></svg>
           Back to Dashboard
        </a>
      </div>
    </div>
  </header>

  <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
    <div class="mb-8">
       <div class="flex items-center gap-3 text-sm text-gray-500 mb-2">
          <span class="font-mono bg-gray-100 px-2 py-0.5 rounded text-xs">ID: %ID%</span>
          <span>•</span>
          <span>%DATE%</span>
       </div>
       <h1 class="text-3xl font-bold text-gray-900 flex items-center gap-3">
          %SPECIES% 
          <span class="px-3 py-1 rounded-full bg-emerald-100 text-emerald-800 text-sm font-medium border border-emerald-200">%AGECAT%</span>
       </h1>
    </div>

    <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
       <!-- Left Column: Image & Basic Info -->
       <div class="lg:col-span-2 space-y-6">
          <div class="rounded-2xl bg-white shadow-sm border border-gray-100 overflow-hidden">
             <div class="relative aspect-video bg-gray-100">
                <img id="explanationImage" src="/uploads/%FILE%" class="w-full h-full object-contain">
                <div class="absolute bottom-4 right-4 flex gap-2">
                   <button id="showOriginal" class="px-3 py-1.5 bg-white/90 backdrop-blur text-xs font-semibold rounded-lg shadow hover:bg-white transition text-gray-700 flex items-center gap-1">
                      <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path></svg>
                      Original
                   </button>
                   <button id="showHeatmap" class="px-3 py-1.5 bg-emerald-600/90 backdrop-blur text-xs font-semibold rounded-lg shadow hover:bg-emerald-700 transition text-white flex items-center gap-1">
                      <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"></path></svg>
                      AI Heatmap
                   </button>
                </div>
             </div>
          </div>
          
          <div class="rounded-2xl bg-white shadow-sm border border-gray-100 p-6">
             <h3 class="text-lg font-bold text-gray-900 mb-4 flex items-center gap-2">
                <svg class="w-5 h-5 text-emerald-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                Quick Insights & AI Explanation
             </h3>
             <div class="prose prose-sm max-w-none text-gray-600">
                <p id="textExplanation" class="leading-relaxed">Loading AI explanation...</p>
                <div class="mt-4 p-4 bg-emerald-50 rounded-xl border border-emerald-100">
                   <p id="confidenceInterpretation" class="text-emerald-800 font-medium"></p>
                </div>
             </div>
          </div>
       </div>

       <!-- Right Column: Stats & Details -->
       <div class="space-y-6">
          <div class="rounded-2xl bg-white shadow-sm border border-gray-100 p-6">
             <h3 class="text-xs font-bold text-gray-400 uppercase tracking-wider mb-4">Tree Profile</h3>
             
             <div class="space-y-5">
                <div>
                   <div class="flex justify-between items-end mb-1">
                      <span class="text-sm text-gray-500">Confidence Score</span>
                      <span class="font-bold text-gray-900">%CONF%</span>
                   </div>
                   <div class="h-2.5 bg-gray-100 rounded-full overflow-hidden">
                      <div class="h-full bg-emerald-500 rounded-full" style="width: %CONF%"></div>
                   </div>
                </div>

                <div class="grid grid-cols-2 gap-4 pt-4 border-t border-gray-50">
                   <div>
                      <div class="text-sm text-gray-500 mb-1">Est. Age</div>
                      <div class="font-bold text-gray-900 text-lg">%AGEEST%</div>
                   </div>
                   <div>
                      <div class="text-sm text-gray-500 mb-1">Age Range</div>
                      <div class="font-bold text-gray-900 text-lg">%AGERANGE%</div>
                   </div>
                </div>
             </div>
          </div>

          %LOCATION_HTML%

          <div class="rounded-2xl bg-white shadow-sm border border-gray-100 p-6">
              <h3 class="text-xs font-bold text-gray-400 uppercase tracking-wider mb-3">Metadata</h3>
              <div class="space-y-2 text-sm">
                 <div class="flex justify-between">
                    <span class="text-gray-500">Uploaded</span>
                    <span class="text-gray-900">%DATE%</span>
                 </div>
                 <div class="flex justify-between">
                    <span class="text-gray-500">File Name</span>
                    <span class="text-gray-900 truncate max-w-[150px]">%FILE%</span>
                 </div>
              </div>
          </div>
       </div>
    </div>
  </div>
  <script>
    const analysisId = '%ID%'
    const originalUrl = '/uploads/%FILE%'
    const imgEl = document.getElementById('explanationImage')
    const txtEl = document.getElementById('textExplanation')
    const confEl = document.getElementById('confidenceInterpretation')
    const btnOrig = document.getElementById('showOriginal')
    const btnHeat = document.getElementById('showHeatmap')
    let heatmapUrl = null

    async function loadExplanation() {
      try {
        const resImg = await fetch(originalUrl)
        const blob = await resImg.blob()
        const dataUrl = await new Promise((resolve, reject) => { const r = new FileReader(); r.onload = () => resolve(r.result); r.onerror = reject; r.readAsDataURL(blob) })
        const res = await fetch('/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ image: dataUrl, no_save: true }) })
        const data = await res.json()
        if (!res.ok || !data.success) { throw new Error(data.error || 'Failed to load explanation') }
        const xai = data.xai || {}
        heatmapUrl = xai.heatmap || null
        imgEl.src = heatmapUrl || originalUrl
        txtEl.textContent = xai.explanation || 'Explanation unavailable'
        confEl.textContent = xai.confidence_level || ''
        if (!heatmapUrl) {
          btnHeat.disabled = true
          btnHeat.classList.add('opacity-50', 'cursor-not-allowed')
          btnHeat.title = 'Heatmap not available'
        }
      } catch (e) {
        imgEl.src = originalUrl
        txtEl.textContent = 'Explanation unavailable'
        confEl.textContent = ''
      }
    }

    btnOrig.addEventListener('click', () => { imgEl.src = originalUrl })
    btnHeat.addEventListener('click', () => { if (heatmapUrl) imgEl.src = heatmapUrl })

    loadExplanation()
  </script>
</body>
</html>'''
    conf = item.get('confidence')
    conf_str = (str(round(conf*100,1)) + '%') if isinstance(conf, (int,float)) else '0%'
    agecat = (item.get('age') or {}).get('category') or 'Unknown'
    agerange = (item.get('age') or {}).get('age_range') or '—'
    ageest_v = (item.get('age') or {}).get('estimated_age')
    ageest = (str(int(ageest_v)) + ' yrs') if isinstance(ageest_v, (int,float)) else '—'
    
    # Location Logic
    loc_html = ''
    coords = item.get('coords')
    if coords:
        lat, lng = coords.get('lat'), coords.get('lng')
        loc_html = f'''
          <div class="rounded-2xl bg-white shadow-sm border border-gray-100 p-6">
             <h3 class="text-xs font-bold text-gray-400 uppercase tracking-wider mb-4">Location</h3>
             <div class="flex items-center gap-2 mb-3">
                <div class="p-2 bg-red-50 rounded-full text-red-500">
                   <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z"></path><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z"></path></svg>
                </div>
                <div>
                   <div class="text-xs text-gray-500">Coordinates</div>
                   <div class="text-sm font-mono font-bold text-gray-900">{lat:.5f}, {lng:.5f}</div>
                </div>
             </div>
             <a href="https://www.google.com/maps/search/?api=1&query={lat},{lng}" target="_blank" class="block w-full py-2.5 bg-gray-50 text-gray-700 text-sm font-semibold rounded-xl text-center hover:bg-gray-100 transition border border-gray-200">Open in Google Maps</a>
          </div>
        '''
    else:
        loc_html = f'''
          <div class="rounded-2xl bg-white shadow-sm border border-gray-100 p-6">
             <h3 class="text-xs font-bold text-gray-400 uppercase tracking-wider mb-4">Location</h3>
             <div class="text-sm text-gray-500 italic mb-3">No GPS data available for this tree.</div>
             <a href="/map?id={item.get('id')}" class="block w-full py-2.5 bg-emerald-50 text-emerald-700 text-sm font-semibold rounded-xl text-center hover:bg-emerald-100 transition border border-emerald-100">Add Location Manually</a>
          </div>
        '''

    html = html.replace('%FILE%', item.get('file_name') or '')
    html = html.replace('%DATE%', item.get('uploaded_at') or '')
    html = html.replace('%SPECIES%', item.get('species') or 'Unknown')
    html = html.replace('%CONF%', conf_str)
    html = html.replace('%AGEEST%', ageest)
    html = html.replace('%AGECAT%', agecat)
    html = html.replace('%AGERANGE%', agerange)
    html = html.replace('%ID%', item.get('id') or '')
    html = html.replace('%LOCATION_HTML%', loc_html)
    return html


@app.route('/explain/<analysis_id>', methods=['GET'])
def explain_analysis(analysis_id):
    try:
        ensure_storage()
        items = load_history()
        item = None
        for x in items:
            if x.get('id') == analysis_id:
                item = x
                break
        if not item:
            return jsonify({'error': 'Not found'}), 404
        filename = item.get('file_name')
        if not filename:
            return jsonify({'error': 'No image found'}), 404
        path = os.path.join(UPLOAD_DIR, filename)
        if not os.path.exists(path):
            return jsonify({'error': 'Image missing'}), 404
        with open(path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode('ascii')
        ext = os.path.splitext(filename)[1].lower().strip('.')
        if ext == 'jpeg':
            ext = 'jpg'
        mime = 'image/jpeg' if ext in ['jpg', 'jpeg'] else f'image/{ext}'
        data_url = 'data:' + mime + ';base64,' + b64
        species_result = {
            'is_bagtikan': item.get('is_bagtikan'),
            'species': item.get('species'),
            'confidence': item.get('confidence')
        }
        xai = generate_xai(data_url, species_result, saved_filename=filename)
        return jsonify({'success': True, 'xai': xai})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/history/<analysis_id>', methods=['DELETE'])
def delete_history(analysis_id):
    ensure_storage()
    ok = delete_history_entry(analysis_id)
    if not ok:
        return jsonify({'error': 'Not found or delete failed'}), 404
    return jsonify({'success': True, 'deleted_id': analysis_id})


@app.route('/api/trees', methods=['GET'])
def api_trees():
    ensure_storage()
    items = load_history()
    mapped = [x for x in items if isinstance(x.get('coords'), dict)]
    return jsonify({'success': True, 'items': mapped})


@app.route('/api/trees/<analysis_id>/location', methods=['POST'])
def api_save_location(analysis_id):
    ensure_storage()
    data = request.get_json() or {}
    lat = data.get('lat')
    lng = data.get('lng')
    accuracy = data.get('accuracy')
    source = data.get('source')
    if lat is None or lng is None:
        return jsonify({'error': 'Missing lat/lng'}), 400
    ok = update_history_location(analysis_id, lat, lng, source=source, accuracy=accuracy)
    if not ok:
        return jsonify({'error': 'Analysis not found'}), 404
    resp = {'success': True, 'id': analysis_id, 'coords': {'lat': float(lat), 'lng': float(lng)}}
    if accuracy is not None: resp['accuracy'] = accuracy
    if source is not None: resp['source'] = source
    return jsonify(resp)


@app.route('/map', methods=['GET'])
def tree_map():
    ensure_storage()
    html = r'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Tree Map</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Poppins:wght@400;600;700&display=swap" rel="stylesheet">
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style> html, body { height: 100%; } #map { height: 70vh; } </style>
  <script>
    tailwind.config = { theme: { extend: { colors: { forest: { 700: '#2e7d32', 500: '#4caf50', 300: '#a5d6a7' } }, fontFamily: { sans: ['Inter','Poppins','ui-sans-serif','system-ui'] } } } }
  </script>
</head>
<body class="min-h-screen bg-gray-50 font-sans text-gray-900">
  <header class="sticky top-0 z-50 bg-gradient-to-r from-[#1b5e20] to-[#2e7d32] text-white shadow-lg border-b border-white/10 backdrop-blur-sm bg-opacity-95">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="h-16 flex items-center justify-between">
        <a href="/" class="flex items-center gap-3 group">
          <span class="inline-grid place-items-center h-10 w-10 rounded-xl bg-white/10 group-hover:bg-white/20 transition duration-300">
            <svg class="h-6 w-6" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 2C7 2 4 7 4 11C4 16 8 20 12 22C16 20 20 16 20 11C20 7 17 2 12 2Z" fill="white" fill-opacity="0.9"/></svg>
          </span>
          <div class="flex flex-col">
            <span class="text-base font-bold tracking-wide leading-tight">BagtikanAI</span>
            <span class="text-[10px] text-emerald-100 uppercase tracking-wider font-medium">Research Dashboard v2.0</span>
          </div>
        </a>
        <a href="/dashboard" class="px-4 py-2 rounded-lg bg-white/20 hover:bg-white/30 transition text-sm font-bold flex items-center gap-2">
           <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"></path></svg>
           Back to Dashboard
        </a>
      </div>
    </div>
  </header>
  <script>const t=document.getElementById('navToggle');const m=document.getElementById('navMenu');if(t) t.addEventListener('click',()=>{m.classList.toggle('hidden')})</script>
  <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
    <div class="flex items-center justify-between mb-8">
      <div>
        <h1 class="text-3xl font-bold text-gray-900 tracking-tight">Tree Map</h1>
        <p class="mt-1 text-sm text-gray-500">Geospatial visualization of analyzed trees</p>
      </div>
      <a href="/" class="inline-flex items-center rounded-xl bg-forest-700 px-5 py-2 text-white font-semibold shadow-md hover:bg-[#1b5e20] transition">New Analysis</a>
    </div>
    <div class="grid grid-cols-1 lg:grid-cols-3 gap-4">
      <div class="lg:col-span-2 rounded-2xl bg-white shadow-md overflow-hidden">
        <div id="map"></div>
      </div>
      <div class="rounded-2xl bg-white shadow-md p-4">
        <div class="flex items-center justify-between mb-3">
          <div class="text-lg font-semibold text-forest-700">Mapped Trees</div>
          <button id="useLoc" class="inline-flex items-center rounded-lg bg-forest-700 px-3 py-2 text-white font-semibold shadow hover:bg-[#256b2b]">Use My Location</button>
        </div>
        <div id="list" class="space-y-3"></div>
      </div>
    </div>
  </div>
  <script>
    const urlParams = new URLSearchParams(location.search)
    const setId = urlParams.get('id')
    const list = document.getElementById('list')
    let map = L.map('map')
    let base = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 19, attribution: '&copy; OpenStreetMap' })
    base.addTo(map)
    map.setView([14.5995, 120.9842], 6) // Center to PH as a reasonable default

    function renderList(items) {
      list.innerHTML = ''
      items.forEach(x => {
        const conf = x.confidence!=null ? (x.confidence*100).toFixed(1)+'%' : '—'
        const el = document.createElement('div')
        el.className = 'rounded-xl border border-gray-200 p-3'
        el.innerHTML = `
          <div class="flex items-center gap-3">
            <img src="/uploads/${x.file_name}" class="h-12 w-12 object-cover rounded-md" alt="thumb">
            <div>
              <div class="text-sm text-gray-500">${new Date(x.uploaded_at).toLocaleString()}</div>
              <div class="text-sm text-gray-800"><span class="font-semibold">${x.species||'Unknown'}</span> • ${x.age?.category||'—'} • ${conf}</div>
            </div>
          </div>
        `
        list.appendChild(el)
      })
    }

    function addMarkers(items) {
      items.forEach(x => {
        if (!x.coords) return
        const m = L.marker([x.coords.lat, x.coords.lng]).addTo(map)
        const conf = x.confidence!=null ? (x.confidence*100).toFixed(1)+'%' : '—'
        const ageCat = x.age?.category || '—'
        const html = `
          <div class="w-52">
            <img src="/uploads/${x.file_name}" class="h-24 w-full object-cover rounded-md" />
            <div class="mt-1 text-sm text-gray-800"><span class="font-semibold">${x.species||'Unknown'}</span></div>
            <div class="text-xs text-gray-700">Age: ${ageCat}</div>
            <div class="text-xs text-gray-700">Confidence: ${conf}</div>
            <div class="text-xs text-gray-500">${new Date(x.uploaded_at).toLocaleString()}</div>
          </div>`
        m.bindPopup(html)
      })
    }

    async function loadData() {
      const res = await fetch('/api/trees')
      const data = await res.json()
      if (!res.ok || !data.success) return
      const items = data.items || []
      renderList(items)
      addMarkers(items)
      const bounds = L.latLngBounds(items.filter(x=>x.coords).map(x=>[x.coords.lat,x.coords.lng]))
      if (bounds.isValid()) map.fitBounds(bounds, { padding: [20,20] })
    }

    async function useMyLocation() {
      if (!navigator.geolocation) { alert('Geolocation not supported') ; return }
      navigator.geolocation.getCurrentPosition(pos => {
        const lat = pos.coords.latitude, lng = pos.coords.longitude
        marker.setLatLng([lat, lng])
        map.setView([lat, lng], 16)
        if (accuracyCircle) accuracyCircle.remove()
        accuracyCircle = L.circle([lat, lng], { radius: pos.coords.accuracy||15, color: '#4caf50', opacity: 0.6, fillOpacity: 0.15 }).addTo(map)
        latInput.value = lat.toFixed(6)
        lngInput.value = lng.toFixed(6)
        accText.textContent = `Accuracy ±${Math.round(pos.coords.accuracy||0)} m`
      }, err => { alert('Failed to get location') })
    }

    let marker
    let accuracyCircle
    let latInput
    let lngInput
    let accText
    if (setId) {
      marker = L.marker(map.getCenter(), { draggable: true }).addTo(map)
      marker.bindPopup('Click map or drag pin to set location, then Save').openPopup()
      const panel = document.createElement('div')
      panel.className = 'fixed bottom-6 left-6 rounded-xl bg-white shadow p-3 border border-gray-200'
      panel.innerHTML = `
        <div class="text-sm text-gray-700">Pin Location</div>
        <div class="mt-2 flex gap-2">
          <input id="latInput" class="rounded-lg border border-gray-300 px-2 py-1 text-sm w-36 focus:ring-2 focus:ring-forest-700 focus:outline-none" placeholder="Latitude">
          <input id="lngInput" class="rounded-lg border border-gray-300 px-2 py-1 text-sm w-36 focus:ring-2 focus:ring-forest-700 focus:outline-none" placeholder="Longitude">
        </div>
        <div id="accText" class="mt-1 text-xs text-gray-500"></div>
      `
      document.body.appendChild(panel)
      latInput = document.getElementById('latInput')
      lngInput = document.getElementById('lngInput')
      accText = document.getElementById('accText')
      const init = marker.getLatLng()
      latInput.value = init.lat.toFixed(6)
      lngInput.value = init.lng.toFixed(6)
      latInput.addEventListener('change', () => { const v = parseFloat(latInput.value); const p = marker.getLatLng(); if (!isNaN(v)) { marker.setLatLng([v, p.lng]); map.setView([v, p.lng], map.getZoom()) } })
      lngInput.addEventListener('change', () => { const v = parseFloat(lngInput.value); const p = marker.getLatLng(); if (!isNaN(v)) { marker.setLatLng([p.lat, v]); map.setView([p.lat, v], map.getZoom()) } })
      marker.on('drag', () => { const p = marker.getLatLng(); latInput.value = p.lat.toFixed(6); lngInput.value = p.lng.toFixed(6) })
      map.on('click', (e) => { marker.setLatLng(e.latlng); latInput.value = e.latlng.lat.toFixed(6); lngInput.value = e.latlng.lng.toFixed(6); })
      const saveBtn = document.createElement('button')
      saveBtn.textContent = 'Save Pin Location'
      saveBtn.className = 'fixed bottom-6 right-6 inline-flex items-center rounded-xl bg-forest-700 px-4 py-2 text-white font-semibold shadow-md hover:bg-[#256b2b]'
      saveBtn.onclick = async () => {
        const { lat, lng } = marker.getLatLng()
        const res = await fetch(`/api/trees/${setId}/location`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ lat, lng, source: 'manual', accuracy: null }) })
        const data = await res.json()
        if (!res.ok || !data.success) { alert(data.error || 'Failed to save location') ; return }
        location.href = '/map'
      }
      document.body.appendChild(saveBtn)
      document.getElementById('useLoc').onclick = useMyLocation
    } else {
      document.getElementById('useLoc').onclick = () => alert('To add a location, use the "Add Location" button on an analysis result.')
    }

    loadData()
  </script>
</body>
</html>'''
    return html
@app.route('/about', methods=['GET'])
def about():
    html = r'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>About</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Poppins:wght@400;600;700&display=swap" rel="stylesheet">
  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    tailwind.config = { theme: { extend: { colors: { forest: { 700: '#2e7d32', 500: '#4caf50', 300: '#a5d6a7' } }, fontFamily: { sans: ['Inter','Poppins','ui-sans-serif','system-ui'] } } } }
  </script>
</head>
<body class="min-h-screen bg-gradient-to-br from-green-50 to-emerald-100 font-sans">
  <header class="sticky top-0 z-40 bg-gradient-to-r from-[#2e7d32] to-[#4caf50] text-white shadow-lg">
    <div class="max-w-6xl mx-auto px-4">
      <div class="h-14 flex items-center justify-between">
        <a href="/" class="flex items-center gap-2">
          <span class="inline-grid place-items-center h-9 w-9 rounded-lg bg-white/10">
            <svg class="h-5 w-5" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 2C7 2 4 7 4 11C4 16 8 20 12 22C16 20 20 16 20 11C20 7 17 2 12 2Z" fill="white" fill-opacity="0.9"/></svg>
          </span>
          <span class="text-sm sm:text-base font-semibold tracking-wide">Enhanced Bagtikan Tree Detector</span>
        </a>
        <button id="navToggle" class="sm:hidden inline-flex items-center rounded-md px-3 py-2 bg-white/10 hover:bg-white/20 transition">Menu</button>
        <nav id="navMenu" class="hidden sm:flex items-center gap-2">
          <a href="/" class="px-3 py-2 rounded-lg hover:bg-white/20 transition text-sm font-medium ring-1 ring-white/10">Analyze Tree</a>
          <a href="/dashboard" class="px-3 py-2 rounded-lg hover:bg-white/20 transition text-sm font-medium ring-1 ring-white/10">Dashboard</a>
          <a href="/map" class="px-3 py-2 rounded-lg hover:bg-white/20 transition text-sm font-medium ring-1 ring-white/10">Tree Map</a>
          <a href="/training" class="px-3 py-2 rounded-lg hover:bg-white/20 transition text-sm font-medium ring-1 ring-white/10">Training</a>
          <a href="/about" class="px-3 py-2 rounded-lg bg-white/15 hover:bg-white/25 transition text-sm font-medium ring-1 ring-white/10">About</a>
        </nav>
      </div>
    </div>
  </header>
  <script>const t=document.getElementById('navToggle');const m=document.getElementById('navMenu');if(t) t.addEventListener('click',()=>{m.classList.toggle('hidden')})</script>
  <div class="max-w-3xl mx-auto px-4 py-10">
    <div class="rounded-2xl bg-white p-6 shadow-md">
      <h1 class="text-xl font-semibold text-forest-700">About / Help</h1>
      <p class="mt-3 text-gray-700">This application performs image-only species detection for Bagtikan (Parashorea malaanonan) and estimates age categories using visual features. It provides a Dashboard of analyses and a Tree Map to visualize locations.</p>
      <ul class="mt-4 space-y-2 text-gray-700">
        <li><span class="font-semibold">Analyze Tree:</span> Upload an image and run species detection + age estimation.</li>
        <li><span class="font-semibold">Dashboard:</span> View history, details, and delete entries.</li>
        <li><span class="font-semibold">Tree Map:</span> Pin locations for analyses and view all markers.</li>
      </ul>
    </div>
  </div>
</body>
</html>'''
    return html
@app.route('/health', methods=['GET'])
def health_check():
    """Enhanced health check endpoint with dataset status"""
    global age_dataset
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'dataset_loaded': age_dataset is not None,
        'dataset_records': len(age_dataset) if age_dataset is not None else 0
    })


@app.route('/dataset/stats', methods=['GET'])
def dataset_statistics():
    """Get dataset statistics endpoint"""
    stats = get_dataset_statistics()
    if stats is None:
        return jsonify({'error': 'Dataset not loaded'}), 404

    return jsonify({
        'success': True,
        'statistics': stats,
        'total_records': len(age_dataset),
        'source': 'Parashorea malaanona age-DBH dataset'
    })


@app.route('/debug/exists', methods=['GET'])
def debug_exists():
    try:
        ok = 'predict_species_heuristic' in globals()
        return jsonify({'predict_species_heuristic_in_globals': bool(ok)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    """Enhanced prediction endpoint with dataset-based age estimation"""
    try:
        # Get data from request
        data = request.get_json()
        no_save = bool((data or {}).get('no_save'))
        tree_id = (data or {}).get('tree_id')

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        image_data = data.get('image')

        if not image_data:
            return jsonify({'error': 'No image provided'}), 400

        # Extract GPS coordinates from EXIF
        coords = None
        try:
            temp_data = image_data
            if 'base64,' in temp_data:
                _, temp_data = temp_data.split(',', 1)
            
            image_bytes = base64.b64decode(temp_data)
            pil_image = Image.open(io.BytesIO(image_bytes))
            coords = get_exif_location(pil_image)
            if coords:
                logger.info(f"Extracted GPS coordinates: {coords}")
        except Exception as e:
            logger.warning(f"GPS extraction failed: {e}")

        # Fallback when model is not loaded: use heuristic species detection
        if model is None:
            species_result = predict_species_heuristic(image_data)

            response_data = {
                'success': True,
                'species': species_result
            }

            age_result = estimate_age_from_image(image_data)
            if age_result is not None:
                response_data['age'] = age_result

            saved_filename = None
            if not no_save:
                ensure_storage()
                analysis_id = str(uuid.uuid4())
                saved_filename = save_base64_image(image_data, analysis_id)
                timestamp = datetime.utcnow().isoformat() + 'Z'

                entry = {
                'id': analysis_id,
                'tree_id': tree_id if tree_id else analysis_id,
                'file_name': saved_filename,
                'uploaded_at': timestamp,
                'species': response_data['species']['species'],
                'is_bagtikan': response_data['species']['is_bagtikan'],
                'confidence': response_data['species']['confidence'],
                'age': response_data.get('age')
            }
            if coords:
                entry['coords'] = coords
                entry['loc_source'] = 'exif'
            save_history_entry(entry)

            if not no_save:
                response_data['analysis_id'] = analysis_id
                response_data['detail_url'] = f"/details/{analysis_id}"
                response_data['map_url'] = f"/map?id={analysis_id}"
                response_data['file_url'] = f"/uploads/{saved_filename}" if saved_filename else None
            response_data['xai'] = generate_xai(image_data, response_data['species'], saved_filename=saved_filename)

            return jsonify(response_data)

        # Preprocess image
        processed_image = preprocess_image(image_data)
        if processed_image is None:
            return jsonify({'error': 'Failed to process image'}), 400

        prediction = model.predict(processed_image, verbose=0)[0][0]

        # Interpret prediction (assuming bagtikan=0, others=1 based on training)
        is_bagtikan = prediction < 0.5
        confidence = (1 - prediction) if is_bagtikan else prediction

        species_result = {
            'is_bagtikan': bool(is_bagtikan),
            'species': 'Bagtikan (Parashorea malaanona)' if is_bagtikan else 'Other Tree Species',
            'confidence': float(confidence),
            'raw_prediction': float(prediction)
        }

        response_data = {
            'success': True,
            'species': species_result
        }

        age_result = estimate_age_from_image(image_data)
        if age_result is not None:
            response_data['age'] = age_result

        saved_filename = None
        if not no_save:
            ensure_storage()
            analysis_id = str(uuid.uuid4())
            saved_filename = save_base64_image(image_data, analysis_id)
            timestamp = datetime.utcnow().isoformat() + 'Z'

            entry = {
                'id': analysis_id,
                'tree_id': tree_id if tree_id else analysis_id,
                'file_name': saved_filename,
                'uploaded_at': timestamp,
                'species': response_data['species']['species'],
                'is_bagtikan': response_data['species']['is_bagtikan'],
                'confidence': response_data['species']['confidence'],
                'age': response_data.get('age')
            }
            if coords:
                entry['coords'] = coords
                entry['loc_source'] = 'exif'
            save_history_entry(entry)

        if not no_save:
            response_data['analysis_id'] = analysis_id
            response_data['detail_url'] = f"/details/{analysis_id}"
            response_data['map_url'] = f"/map?id={analysis_id}"
            response_data['file_url'] = f"/uploads/{saved_filename}" if saved_filename else None
        response_data['xai'] = generate_xai(image_data, response_data['species'], saved_filename=saved_filename)

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/model_info', methods=['GET'])
def model_info():
    """Get enhanced model and dataset information"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        model_info = {
            'model': {
                'input_shape': model.input_shape,
                'output_shape': model.output_shape,
                'total_params': model.count_params()
            }
        }

        # Add dataset information if available
        if age_dataset is not None:
            model_info['dataset'] = {
                'total_records': len(age_dataset),
                'categories': age_dataset['Age Category'].unique().tolist(),
                'age_range': [
                    int(age_dataset['Estimated Age (years)'].min()),
                    int(age_dataset['Estimated Age (years)'].max())
                ],
                'statistics': get_dataset_statistics()
            }

        return jsonify(model_info)
    except Exception as e:
        return jsonify({'error': f'Failed to get model info: {str(e)}'}), 500


def _latest_training_timestamp():
    try:
        reports_dir = 'training_reports'
        if not os.path.exists(reports_dir):
            return None
        files = [f for f in os.listdir(reports_dir) if f.startswith('training_report_') and f.endswith('.json')]
        if not files:
            return None
        files.sort()
        latest = files[-1]
        ts = latest.replace('training_report_', '').replace('.json', '')
        return ts
    except Exception:
        return None

@app.route('/training/metrics', methods=['GET'])
def training_metrics():
    try:
        ts = _latest_training_timestamp()
        if ts is None:
            return jsonify({'error': 'No training reports found'}), 404
        reports_dir = 'training_reports'
        report_path = os.path.join(reports_dir, f'training_report_{ts}.json')
        history_path = os.path.join(reports_dir, f'training_history_{ts}.json')
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
        history = None
        if os.path.exists(history_path):
            with open(history_path, 'r', encoding='utf-8') as hf:
                history = json.load(hf)
        resp = {
            'success': True,
            'timestamp': ts,
            'final_metrics': report.get('final_metrics'),
            'classification_report_text': report.get('classification_report_text'),
            'confusion_matrix': report.get('confusion_matrix'),
            'class_names': report.get('class_names'),
            'history': history
        }
        return jsonify(resp)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/training', methods=['GET'])
def training_page():
    html = r'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Training Metrics</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Poppins:wght@400;600;700&display=swap" rel="stylesheet">
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <script>
    tailwind.config = { theme: { extend: { colors: { forest: { 700: '#2e7d32', 500: '#4caf50', 300: '#a5d6a7' } }, fontFamily: { sans: ['Inter','Poppins','ui-sans-serif','system-ui'] } } } }
  </script>
  <style>
    canvas { max-height: 300px; }
  </style>
</head>
<body class="min-h-screen bg-gradient-to-br from-green-50 to-emerald-100 font-sans">
  <header class="sticky top-0 z-40 bg-gradient-to-r from-[#2e7d32] to-[#4caf50] text-white shadow-lg">
    <div class="max-w-6xl mx-auto px-4">
      <div class="h-14 flex items-center justify-between">
        <a href="/" class="flex items-center gap-2">
          <span class="inline-grid place-items-center h-9 w-9 rounded-lg bg-white/10">
            <svg class="h-5 w-5" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 2C7 2 4 7 4 11C4 16 8 20 12 22C16 20 20 16 20 11C20 7 17 2 12 2Z" fill="white" fill-opacity="0.9"/></svg>
          </span>
          <span class="text-sm sm:text-base font-semibold tracking-wide">Enhanced Bagtikan Tree Detector</span>
        </a>
        <button id="navToggle" class="sm:hidden inline-flex items-center rounded-md px-3 py-2 bg-white/10 hover:bg-white/20 transition">Menu</button>
        <nav id="navMenu" class="hidden sm:flex items-center gap-2">
          <a href="/" class="px-3 py-2 rounded-lg hover:bg-white/20 transition text-sm font-medium ring-1 ring-white/10">Analyze Tree</a>
          <a href="/dashboard" class="px-3 py-2 rounded-lg hover:bg-white/20 transition text-sm font-medium ring-1 ring-white/10">Dashboard</a>
          <a href="/map" class="px-3 py-2 rounded-lg hover:bg-white/20 transition text-sm font-medium ring-1 ring-white/10">Tree Map</a>
          <a href="/training" class="px-3 py-2 rounded-lg bg-white/15 hover:bg-white/25 transition text-sm font-medium ring-1 ring-white/10">Training</a>
          <a href="/about" class="px-3 py-2 rounded-lg hover:bg-white/20 transition text-sm font-medium ring-1 ring-white/10">About</a>
        </nav>
      </div>
    </div>
  </header>
  <script>const t=document.getElementById('navToggle');const m=document.getElementById('navMenu');if(t) t.addEventListener('click',()=>{m.classList.toggle('hidden')})</script>
  <div class="max-w-6xl mx-auto px-4 py-10">
    <div class="flex items-center justify-between">
      <h1 class="text-2xl sm:text-3xl font-bold text-forest-700">Training Metrics</h1>
      <a href="/" class="inline-flex items-center rounded-xl bg-forest-700 px-4 py-2 text-white font-semibold shadow-md hover:bg-[#256b2b]">New Analysis</a>
    </div>
    <div id="status" class="mt-4 text-gray-700"></div>
    <div class="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
      <div class="rounded-2xl bg-white p-5 shadow-md">
        <div class="text-sm text-gray-600">Final Validation Accuracy</div>
        <div id="valAcc" class="mt-1 text-2xl font-bold text-forest-700">—</div>
      </div>
      <div class="rounded-2xl bg-white p-5 shadow-md">
        <div class="text-sm text-gray-600">Final Validation Loss</div>
        <div id="valLoss" class="mt-1 text-2xl font-bold text-forest-700">—</div>
      </div>
      <div class="rounded-2xl bg-white p-5 shadow-md">
        <div class="text-sm text-gray-600">Best Val Accuracy</div>
        <div id="bestValAcc" class="mt-1 text-2xl font-bold text-forest-700">—</div>
      </div>
    </div>
    <div class="mt-6 grid grid-cols-1 lg:grid-cols-2 gap-4">
      <div class="rounded-2xl bg-white p-6 shadow-md">
        <div class="text-lg font-semibold text-gray-800">Accuracy per Epoch</div>
        <canvas id="accChart"></canvas>
      </div>
      <div class="rounded-2xl bg-white p-6 shadow-md">
        <div class="text-lg font-semibold text-gray-800">Loss per Epoch</div>
        <canvas id="lossChart"></canvas>
      </div>
    </div>
    <div class="mt-6 rounded-2xl bg-white p-6 shadow-md">
      <div class="text-lg font-semibold text-gray-800">Confusion Matrix</div>
      <div id="cmGrid" class="mt-3 grid grid-cols-2 md:grid-cols-3 gap-2"></div>
    </div>
    <div class="mt-6 rounded-2xl bg-white p-6 shadow-md">
      <div class="text-lg font-semibold text-gray-800">Classification Report</div>
      <pre id="clfText" class="mt-3 text-sm text-gray-700 whitespace-pre-wrap"></pre>
    </div>
  </div>
  <script>
    async function loadMetrics() {
      const res = await fetch('/training/metrics')
      if (!res.ok) { document.getElementById('status').textContent = 'No training reports found.' ; return }
      const data = await res.json()
      document.getElementById('status').textContent = 'Latest training: ' + data.timestamp
      const fm = data.final_metrics || {}
      const fmtPct = x => (typeof x==='number') ? (x*100).toFixed(2)+'%' : '—'
      document.getElementById('valAcc').textContent = fmtPct(fm.validation_accuracy)
      document.getElementById('valLoss').textContent = (typeof fm.validation_loss==='number') ? fm.validation_loss.toFixed(4) : '—'
      document.getElementById('bestValAcc').textContent = fmtPct(fm.best_val_accuracy)
      const h = data.history || {}
      const acc = h.accuracy || []
      const vacc = h.val_accuracy || []
      const loss = h.loss || []
      const vloss = h.val_loss || []
      const labels = acc.map((_,i)=>i+1)
      new Chart(document.getElementById('accChart').getContext('2d'), { type:'line', data:{ labels, datasets:[ { label:'Train', data:acc, borderColor:'#2e7d32' }, { label:'Val', data:vacc, borderColor:'#4caf50' } ] }, options:{ responsive:true, plugins:{ legend:{ position:'bottom' } } } })
      new Chart(document.getElementById('lossChart').getContext('2d'), { type:'line', data:{ labels, datasets:[ { label:'Train', data:loss, borderColor:'#f59e0b' }, { label:'Val', data:vloss, borderColor:'#ef4444' } ] }, options:{ responsive:true, plugins:{ legend:{ position:'bottom' } } } })
      const cm = data.confusion_matrix || []
      const names = data.class_names || []
      const grid = document.getElementById('cmGrid')
      grid.innerHTML = ''
      for (let r=0; r<cm.length; r++) {
        for (let c=0; c<cm[r].length; c++) {
          const el = document.createElement('div')
          el.className = 'rounded-lg border border-gray-200 p-3 text-center'
          el.innerHTML = `<div class="text-sm text-gray-600">${names[c]||('Class '+c)}</div><div class="text-xl font-semibold text-gray-800">${cm[r][c]}</div>`
          grid.appendChild(el)
        }
      }
      document.getElementById('clfText').textContent = data.classification_report_text || ''
    }
    loadMetrics()
  </script>
</body>
</html>'''
    return html

@app.route('/analytics', methods=['GET'])
def analytics_view():
    html = r'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Map Analytics</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Poppins:wght@400;600;700&display=swap" rel="stylesheet">
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.css"/>
  <link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.Default.css"/>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://unpkg.com/leaflet.heat@0.2.0/dist/leaflet-heat.js"></script>
  <script src="https://unpkg.com/leaflet.markercluster@1.5.3/dist/leaflet.markercluster.js"></script>
  <script>
    tailwind.config = { theme: { extend: { colors: { forest: { 700: '#2e7d32', 500: '#4caf50', 300: '#a5d6a7' }, age: { seedling: '#A7F3D0', young: '#6EE7B7', middle: '#34D399', mature: '#059669' } }, fontFamily: { sans: ['Inter','Poppins','ui-sans-serif','system-ui'] } } } }
  </script>
  <style>
    #map { height: 70vh; }
  </style>
</head>
<body class="min-h-screen bg-gradient-to-br from-green-50 to-emerald-100 font-sans">
  <header class="sticky top-0 z-40 bg-gradient-to-r from-[#2e7d32] to-[#4caf50] text-white shadow-lg">
    <div class="max-w-6xl mx-auto px-4">
      <div class="h-14 flex items-center justify-between">
        <a href="/" class="flex items-center gap-2">
          <span class="inline-grid place-items-center h-9 w-9 rounded-lg bg-white/10">
            <svg class="h-5 w-5" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 2C7 2 4 7 4 11C4 16 8 20 12 22C16 20 20 16 20 11C20 7 17 2 12 2Z" fill="white" fill-opacity="0.9"/></svg>
          </span>
          <span class="text-sm sm:text-base font-semibold tracking-wide">Enhanced Bagtikan Tree Detector</span>
        </a>
        <nav class="hidden sm:flex items-center gap-2">
          <a href="/" class="px-3 py-2 rounded-lg hover:bg-white/20 transition text-sm font-medium ring-1 ring-white/10">Analyze Tree</a>
          <a href="/dashboard" class="px-3 py-2 rounded-lg hover:bg-white/20 transition text-sm font-medium ring-1 ring-white/10">Dashboard</a>
          <a href="/map" class="px-3 py-2 rounded-lg hover:bg-white/20 transition text-sm font-medium ring-1 ring-white/10">Tree Map</a>
          <a href="/analytics" class="px-3 py-2 rounded-lg bg-white/15 hover:bg-white/25 transition text-sm font-medium ring-1 ring-white/10">Map Analytics</a>
          <a href="/training" class="px-3 py-2 rounded-lg hover:bg-white/20 transition text-sm font-medium ring-1 ring-white/10">Training</a>
        </nav>
      </div>
    </div>
  </header>
  <div class="max-w-6xl mx-auto px-4 py-8">
    <div class="flex items-center justify-between">
      <h1 class="text-2xl sm:text-3xl font-bold text-forest-700">Map Analytics</h1>
      <div class="text-sm text-gray-700">Spatial analysis for forestry planning and LGU reporting</div>
    </div>
    <div class="mt-4 grid grid-cols-1 lg:grid-cols-3 gap-4">
      <div class="lg:col-span-2 rounded-2xl bg-white shadow-md overflow-hidden">
        <div id="map"></div>
      </div>
      <div class="rounded-2xl bg-white shadow-md p-4">
        <div class="text-lg font-semibold text-forest-700">Analytics Panel</div>
        <div class="mt-3 space-y-3">
          <label class="flex items-center justify-between">
            <span class="text-sm text-gray-700">Density Heatmap</span>
            <input id="toggleHeat" type="checkbox" class="h-4 w-4 rounded border-gray-300 text-forest-700 focus:ring-forest-700">
          </label>
          <label class="flex items-center justify-between">
            <span class="text-sm text-gray-700">Age Distribution</span>
            <input id="toggleAge" type="checkbox" class="h-4 w-4 rounded border-gray-300 text-forest-700 focus:ring-forest-700">
          </label>
          <label class="flex items-center justify-between">
            <span class="text-sm text-gray-700">Species Clusters</span>
            <input id="toggleCluster" type="checkbox" class="h-4 w-4 rounded border-gray-300 text-forest-700 focus:ring-forest-700" checked>
          </label>
        </div>
      </div>
    </div>
    <div class="mt-4 rounded-2xl bg-white shadow-md p-4">
      <div class="text-lg font-semibold text-gray-800">Selected Region Stats</div>
      <div id="regionStats" class="mt-2 text-sm text-gray-700">Click a cluster to view aggregated statistics.</div>
    </div>
  </div>
  <script>
    let map = L.map('map')
    let base = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 19, attribution: '&copy; OpenStreetMap' })
    base.addTo(map)
    map.setView([14.5995, 120.9842], 6)

    const heatToggle = document.getElementById('toggleHeat')
    const ageToggle = document.getElementById('toggleAge')
    const clusterToggle = document.getElementById('toggleCluster')
    const regionStats = document.getElementById('regionStats')

    const markers = L.markerClusterGroup({
      showCoverageOnHover: false,
      iconCreateFunction: function(cluster) {
        const children = cluster.getAllChildMarkers()
        let bag=0, oth=0
        children.forEach(m => { if (m.options.is_bagtikan) bag++; else oth++; })
        const total = children.length
        const domIsBagtikan = bag >= oth
        const color = domIsBagtikan ? '#2e7d32' : '#64748b'
        const html = `<div style="background:${color};color:white;border-radius:9999px;display:flex;align-items:center;justify-content:center;width:36px;height:36px;font-weight:700">${total}</div>`
        return L.divIcon({ html, className: 'species-cluster-icon', iconSize: [36,36] })
      }
    })

    function markerPopupContent(x) {
      const thumb = x.file_name ? `/uploads/${x.file_name}` : ''
      const confPct = (typeof x.confidence==='number') ? (x.confidence*100).toFixed(1)+'%' : '—'
      const ageCat = (x.age||{}).category || '—'
      return `
        <div class="p-1">
          ${thumb ? `<img src="${thumb}" class="w-40 h-24 object-cover rounded-lg mb-2" />` : ''}
          <div class="text-sm font-semibold text-forest-700">${x.species||'Unknown'}</div>
          <div class="text-xs text-gray-700">Date: ${x.uploaded_at ? new Date(x.uploaded_at).toLocaleString() : '—'}</div>
          <div class="text-xs text-gray-700">Age: ${ageCat}</div>
          <div class="text-xs text-gray-700">Confidence: ${confPct}</div>
          <div class="mt-1"><a href="/details/${x.id}" class="text-xs inline-flex items-center rounded bg-forest-700 px-2 py-1 text-white">View Details</a></div>
        </div>`
    }

    let heatLayer = null
    let ageLayerGroup = L.layerGroup()

    function buildHeatLayer(items) {
      const pts = items.filter(x=>x.coords).map(x=>[x.coords.lat, x.coords.lng, 0.6])
      heatLayer = L.heatLayer(pts, { radius: 25, blur: 15, minOpacity: 0.2, gradient: { 0.2: 'green', 0.5: 'yellow', 0.8: 'red' } })
    }

    function buildAgeDistributionLayer(items) {
      ageLayerGroup.clearLayers()
      const bins = {}
      const res = 0.05
      items.filter(x=>x.coords).forEach(x=>{
        const lat = x.coords.lat, lng = x.coords.lng
        const key = (Math.round(lat/res)*res).toFixed(2)+','+(Math.round(lng/res)*res).toFixed(2)
        const cat = ((x.age||{}).category||'Unknown').toLowerCase()
        if (!bins[key]) bins[key] = { lat: Math.round(lat/res)*res, lng: Math.round(lng/res)*res, seedling:0, young:0, middle:0, mature:0, total:0 }
        if (cat.includes('seedling')) bins[key].seedling++
        else if (cat.includes('young')) bins[key].young++
        else if (cat.includes('middle')) bins[key].middle++
        else if (cat.includes('mature')) bins[key].mature++
        bins[key].total++
      })
      Object.values(bins).forEach(b=>{
        const baseRadius = 200
        const rSeed = baseRadius * (b.seedling / Math.max(1,b.total))
        const rYoung = baseRadius * (b.young / Math.max(1,b.total))
        const rMid = baseRadius * (b.middle / Math.max(1,b.total))
        const rMat = baseRadius * (b.mature / Math.max(1,b.total))
        if (b.seedling>0) L.circle([b.lat,b.lng], { radius: rSeed, color: '#A7F3D0', weight: 1, fillOpacity: 0.25 }).addTo(ageLayerGroup)
        if (b.young>0) L.circle([b.lat,b.lng], { radius: rYoung, color: '#6EE7B7', weight: 1, fillOpacity: 0.25 }).addTo(ageLayerGroup)
        if (b.middle>0) L.circle([b.lat,b.lng], { radius: rMid, color: '#34D399', weight: 1, fillOpacity: 0.25 }).addTo(ageLayerGroup)
        if (b.mature>0) L.circle([b.lat,b.lng], { radius: rMat, color: '#059669', weight: 1, fillOpacity: 0.25 }).addTo(ageLayerGroup)
        const popup = `
          <div class="text-sm">
            <div class="font-semibold text-forest-700 mb-1">Age Distribution</div>
            <div class="grid grid-cols-2 gap-2">
              <div>Seedling: ${b.seedling}</div>
              <div>Young: ${b.young}</div>
              <div>Middle: ${b.middle}</div>
              <div>Mature: ${b.mature}</div>
            </div>
            <div class="mt-1 text-gray-600">Total: ${b.total}</div>
          </div>`
        L.circleMarker([b.lat,b.lng], { radius: 0.1, opacity: 0 }).addTo(ageLayerGroup).bindPopup(popup)
      })
    }

    function updateRegionStatsFromCluster(clusterLayer) {
      const children = clusterLayer.getAllChildMarkers()
      let bag=0,oth=0; let s=0,y=0,m=0,mt=0
      children.forEach(m=>{
        if (m.options.is_bagtikan) bag++; else oth++;
        const cat = (m.options.age_category||'').toLowerCase()
        if (cat.includes('seedling')) s++; else if (cat.includes('young')) y++; else if (cat.includes('middle')) m++; else if (cat.includes('mature')) mt++;
      })
      const total = children.length
      regionStats.innerHTML = `
        <div class="grid grid-cols-2 md:grid-cols-4 gap-2">
          <div class="rounded bg-white p-2 shadow"><div class="text-xs text-gray-600">Total</div><div class="text-lg font-semibold text-forest-700">${total}</div></div>
          <div class="rounded bg-white p-2 shadow"><div class="text-xs text-gray-600">Bagtikan</div><div class="text-lg font-semibold text-forest-700">${bag}</div></div>
          <div class="rounded bg-white p-2 shadow"><div class="text-xs text-gray-600">Other</div><div class="text-lg font-semibold text-forest-700">${oth}</div></div>
          <div class="rounded bg-white p-2 shadow col-span-2 md:col-span-1"><div class="text-xs text-gray-600">Seedling</div><div class="text-lg font-semibold text-forest-700">${s}</div></div>
          <div class="rounded bg-white p-2 shadow"><div class="text-xs text-gray-600">Young</div><div class="text-lg font-semibold text-forest-700">${y}</div></div>
          <div class="rounded bg-white p-2 shadow"><div class="text-xs text-gray-600">Middle</div><div class="text-lg font-semibold text-forest-700">${m}</div></div>
          <div class="rounded bg-white p-2 shadow"><div class="text-xs text-gray-600">Mature</div><div class="text-lg font-semibold text-forest-700">${mt}</div></div>
        </div>`
    }

    function addMarkers(items) {
      markers.clearLayers()
      items.filter(x=>x.coords).forEach(x=>{
        const iconHtml = `<div style="background:${x.is_bagtikan ? '#2e7d32' : '#64748b'};width:12px;height:12px;border-radius:9999px;border:2px solid white;box-shadow:0 0 0 2px rgba(0,0,0,0.15)"></div>`
        const icon = L.divIcon({ html: iconHtml, className: 'tree-marker', iconSize: [12,12] })
        const m = L.marker([x.coords.lat, x.coords.lng], { icon, is_bagtikan: !!x.is_bagtikan, age_category: (x.age||{}).category || '' })
        m.bindPopup(markerPopupContent(x))
        markers.addLayer(m)
      })
      markers.on('clusterclick', e => { updateRegionStatsFromCluster(e.layer) })
    }

    async function loadData() {
      const res = await fetch('/api/trees')
      const data = await res.json()
      if (!res.ok || !data.success) return
      const items = data.items || []
      addMarkers(items)
      buildHeatLayer(items)
      buildAgeDistributionLayer(items)
      map.addLayer(markers)
      const bounds = L.latLngBounds(items.filter(x=>x.coords).map(x=>[x.coords.lat,x.coords.lng]))
      if (bounds.isValid()) map.fitBounds(bounds, { padding: [20,20] })
    }

    heatToggle.addEventListener('change', () => {
      if (!heatLayer) return
      if (heatToggle.checked) { heatLayer.addTo(map) } else { map.removeLayer(heatLayer) }
    })
    ageToggle.addEventListener('change', () => {
      if (ageToggle.checked) { ageLayerGroup.addTo(map) } else { map.removeLayer(ageLayerGroup) }
    })
    clusterToggle.addEventListener('change', () => {
      if (clusterToggle.checked) { map.addLayer(markers) } else { map.removeLayer(markers) }
    })

    loadData()
  </script>
</body>
</html>'''
    return html
@app.route('/tracking/<tree_id>', methods=['GET'])
def tracking_view(tree_id):
    ensure_storage()
    history = load_history()
    # Filter for this tree
    tree_history = [h for h in history if h.get('tree_id') == tree_id or (not h.get('tree_id') and h.get('id') == tree_id)]
    
    if not tree_history:
        # Fallback: check if tree_id matches an analysis ID (backward compatibility for old single-record trees)
        tree_history = [h for h in history if h.get('id') == tree_id]
        if not tree_history:
            return "Tree not found", 404
    
    # Sort by date
    tree_history.sort(key=lambda x: x.get('uploaded_at', ''), reverse=True)
    
    latest = tree_history[0]
    first = tree_history[-1]
    
    html = r'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Tree Growth Tracking</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Poppins:wght@400;600;700&display=swap" rel="stylesheet">
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/moment.js/2.29.4/moment.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-moment"></script>
  <script>
    tailwind.config = { theme: { extend: { colors: { forest: { 700: '#2e7d32', 500: '#4caf50', 300: '#a5d6a7' } }, fontFamily: { sans: ['Inter','Poppins','ui-sans-serif','system-ui'] } } } }
  </script>
</head>
<body class="min-h-screen bg-gradient-to-br from-green-50 to-emerald-100 font-sans">
  <header class="sticky top-0 z-40 bg-gradient-to-r from-[#2e7d32] to-[#4caf50] text-white shadow-lg">
    <div class="max-w-6xl mx-auto px-4">
      <div class="h-14 flex items-center justify-between">
        <a href="/" class="flex items-center gap-2">
          <span class="inline-grid place-items-center h-9 w-9 rounded-lg bg-white/10">
            <svg class="h-5 w-5" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 2C7 2 4 7 4 11C4 16 8 20 12 22C16 20 20 16 20 11C20 7 17 2 12 2Z" fill="white" fill-opacity="0.9"/></svg>
          </span>
          <span class="text-sm sm:text-base font-semibold tracking-wide">Enhanced Bagtikan Tree Detector</span>
        </a>
        <nav class="hidden sm:flex items-center gap-2">
          <a href="/" class="px-3 py-2 rounded-lg hover:bg-white/20 transition text-sm font-medium ring-1 ring-white/10">Analyze Tree</a>
          <a href="/dashboard" class="px-3 py-2 rounded-lg hover:bg-white/20 transition text-sm font-medium ring-1 ring-white/10">Dashboard</a>
          <a href="/map" class="px-3 py-2 rounded-lg hover:bg-white/20 transition text-sm font-medium ring-1 ring-white/10">Tree Map</a>
        </nav>
      </div>
    </div>
  </header>

  <div class="max-w-6xl mx-auto px-4 py-8">
    <!-- Header Info -->
    <div class="rounded-2xl bg-white p-6 shadow-md mb-6 border border-gray-200">
      <div class="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div class="text-sm text-forest-700 font-medium uppercase tracking-wider">Longitudinal Analysis</div>
          <h1 class="text-2xl sm:text-3xl font-bold text-gray-800 mt-1">Tree Tracking Record</h1>
          <p class="text-gray-600 mt-2 flex items-center gap-2">
            <span class="font-mono bg-gray-100 px-2 py-1 rounded text-xs">ID: %TREE_ID%</span>
            <span class="text-gray-400">|</span>
            <span>%SPECIES%</span>
          </p>
        </div>
        <div class="flex gap-3">
            <div class="text-right">
                <div class="text-sm text-gray-500">First Observed</div>
                <div class="font-medium text-gray-800">%FIRST_DATE%</div>
            </div>
            <div class="w-px bg-gray-200"></div>
            <div class="text-right">
                <div class="text-sm text-gray-500">Total Observations</div>
                <div class="font-medium text-forest-700 text-lg">%COUNT%</div>
            </div>
        </div>
      </div>
      %LOCATION_HTML%
    </div>

    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <!-- Left Column: Timeline -->
      <div class="lg:col-span-1">
        <div class="bg-white rounded-2xl shadow-md p-6 border border-gray-200 h-full">
          <h2 class="text-lg font-bold text-gray-800 mb-4 flex items-center gap-2">
            <svg class="h-5 w-5 text-forest-700" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
            Growth Timeline
          </h2>
          <div class="space-y-6 relative before:absolute before:inset-0 before:ml-5 before:-translate-x-px md:before:mx-auto md:before:translate-x-0 before:h-full before:w-0.5 before:bg-gradient-to-b before:from-transparent before:via-gray-300 before:to-transparent">
            <!-- Timeline items injected here -->
            %TIMELINE_ITEMS%
          </div>
        </div>
      </div>

      <!-- Right Column: Analytics & Comparison -->
      <div class="lg:col-span-2 space-y-6">
        
        <!-- Growth Summary -->
        <div class="bg-white rounded-2xl shadow-md p-6 border border-gray-200">
            <h2 class="text-lg font-bold text-gray-800 mb-2">Growth Summary</h2>
            <p class="text-gray-600 leading-relaxed">%SUMMARY_TEXT%</p>
        </div>

        <!-- Charts -->
        <div class="bg-white rounded-2xl shadow-md p-6 border border-gray-200">
          <h2 class="text-lg font-bold text-gray-800 mb-4">Estimated Age Progression</h2>
          <div class="h-64">
            <canvas id="ageChart"></canvas>
          </div>
        </div>
        
        <div class="bg-white rounded-2xl shadow-md p-6 border border-gray-200">
          <h2 class="text-lg font-bold text-gray-800 mb-4">Confidence Trend</h2>
          <div class="h-48">
            <canvas id="confChart"></canvas>
          </div>
        </div>

        <!-- Visual Comparison -->
        <div class="bg-white rounded-2xl shadow-md p-6 border border-gray-200">
            <h2 class="text-lg font-bold text-gray-800 mb-4">Visual Progression (First vs. Latest)</h2>
            <div class="grid grid-cols-2 gap-4">
                <div>
                    <div class="text-sm text-gray-500 mb-1">%FIRST_DATE%</div>
                    <img src="%FIRST_IMG%" class="w-full h-48 object-cover rounded-lg border border-gray-200" alt="First">
                    <div class="mt-2 text-center text-sm font-medium">%FIRST_AGE%</div>
                </div>
                <div>
                    <div class="text-sm text-gray-500 mb-1">%LATEST_DATE%</div>
                    <img src="%LATEST_IMG%" class="w-full h-48 object-cover rounded-lg border border-gray-200" alt="Latest">
                    <div class="mt-2 text-center text-sm font-medium">%LATEST_AGE%</div>
                </div>
            </div>
        </div>

      </div>
    </div>
  </div>

  <script>
    const historyData = %HISTORY_JSON%;
    
    // Prepare Chart Data
    const labels = historyData.map(d => d.uploaded_at);
    const ageData = historyData.map(d => d.age ? d.age.estimated_age : null);
    const confData = historyData.map(d => d.confidence);

    // Age Chart
    new Chart(document.getElementById('ageChart'), {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Estimated Age (Years)',
                data: ageData,
                borderColor: '#2e7d32',
                backgroundColor: 'rgba(46, 125, 50, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: 'time',
                    time: { unit: 'month' },
                    grid: { display: false }
                },
                y: {
                    beginAtZero: true,
                    title: { display: true, text: 'Age (Years)' }
                }
            }
        }
    });

    // Confidence Chart
    new Chart(document.getElementById('confChart'), {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Model Confidence',
                data: confData,
                borderColor: '#4caf50',
                borderWidth: 2,
                pointBackgroundColor: '#fff',
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: 'time',
                    time: { unit: 'month' },
                    display: false
                },
                y: {
                    min: 0, max: 1,
                    title: { display: true, text: 'Confidence' }
                }
            }
        }
    });
  </script>
</body>
</html>'''
    
    # Process Placeholders
    count = len(tree_history)
    first_date = datetime.fromisoformat(first['uploaded_at'].replace('Z', '')).strftime('%Y-%m-%d')
    latest_date = datetime.fromisoformat(latest['uploaded_at'].replace('Z', '')).strftime('%Y-%m-%d')
    
    timeline_html = ""
    for entry in tree_history:
        date_str = datetime.fromisoformat(entry['uploaded_at'].replace('Z', '')).strftime('%b %d, %Y')
        img_url = f"/uploads/{entry['file_name']}" if entry.get('file_name') else ""
        age_display = f"{entry.get('age', {}).get('estimated_age', '?')} yrs"
        conf_display = f"{int(entry.get('confidence', 0)*100)}%"
        
        timeline_html += f'''
        <div class="relative flex items-center justify-between md:justify-normal md:odd:flex-row-reverse group is-active">
            <div class="flex items-center justify-center w-10 h-10 rounded-full border border-white bg-forest-700 text-white shadow shrink-0 md:order-1 md:group-odd:-translate-x-1/2 md:group-even:translate-x-1/2 z-10">
                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6v6m0 0v6m0-6h6m-6 0H6"></path></svg>
            </div>
            <div class="w-[calc(100%-4rem)] md:w-[calc(50%-2.5rem)] p-4 rounded-xl bg-white border border-gray-200 shadow-sm hover:shadow-md transition cursor-pointer" onclick="window.location.href='/details/{entry['id']}'">
                <div class="flex items-start gap-3">
                    <img src="{img_url}" class="w-16 h-16 rounded-lg object-cover bg-gray-100">
                    <div>
                        <time class="mb-1 text-xs font-semibold uppercase text-gray-500">{date_str}</time>
                        <div class="text-sm font-bold text-gray-800">{age_display} <span class="text-gray-400 font-normal">| {conf_display}</span></div>
                        <div class="text-xs text-forest-700 mt-1">View Analysis &rarr;</div>
                    </div>
                </div>
            </div>
        </div>
        '''

    # Generate Summary
    age_diff = (latest.get('age', {}).get('estimated_age', 0) or 0) - (first.get('age', {}).get('estimated_age', 0) or 0)
    summary = f"This tree has been tracked for {count} observations from {first_date} to {latest_date}. "
    if age_diff > 0:
        summary += f"The estimated age has progressed by approximately {age_diff} years. "
    elif age_diff < 0:
        summary += "The age estimation shows some variance (decrease observed), which may be due to lighting or angle differences. "
    else:
        summary += "The estimated age has remained consistent. "
        
    summary += f"Current confidence is {int(latest.get('confidence',0)*100)}%."

    # Replacements
    html = html.replace('%TREE_ID%', tree_id)
    html = html.replace('%SPECIES%', latest.get('species', 'Unknown'))
    html = html.replace('%FIRST_DATE%', first_date)
    html = html.replace('%COUNT%', str(count))
    html = html.replace('%TIMELINE_ITEMS%', timeline_html)
    html = html.replace('%SUMMARY_TEXT%', summary)
    html = html.replace('%HISTORY_JSON%', json.dumps(tree_history))
    
    html = html.replace('%FIRST_IMG%', f"/uploads/{first['file_name']}" if first.get('file_name') else "")
    html = html.replace('%LATEST_IMG%', f"/uploads/{latest['file_name']}" if latest.get('file_name') else "")
    html = html.replace('%FIRST_AGE%', f"{first.get('age', {}).get('estimated_age', '?')} years")
    html = html.replace('%LATEST_AGE%', f"{latest.get('age', {}).get('estimated_age', '?')} years")
    html = html.replace('%LATEST_DATE%', latest_date)

    # Location
    loc_html = ""
    if latest.get('coords'):
        lat = latest['coords']['lat']
        lng = latest['coords']['lng']
        loc_html = f'''
        <div class="mt-4 pt-4 border-t border-gray-100">
            <div class="text-xs text-gray-500 uppercase tracking-wide font-semibold">Last Known Location</div>
            <div class="flex items-center gap-2 mt-1">
                <svg class="w-4 h-4 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z"></path><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z"></path></svg>
                <span class="text-gray-800 font-mono text-sm">{lat:.6f}, {lng:.6f}</span>
                <a href="https://www.google.com/maps/search/?api=1&query={lat},{lng}" target="_blank" class="text-xs text-forest-700 hover:underline ml-2">Open in Maps &rarr;</a>
            </div>
        </div>
        '''
    html = html.replace('%LOCATION_HTML%', loc_html)

    return html

@app.route('/api/tracked-trees', methods=['GET'])
def api_get_tracked_trees():
    ensure_storage()
    history = load_history()
    
    # Group by tree_id
    trees = {}
    for h in history:
        tid = h.get('tree_id')
        if not tid:
            tid = h.get('id') # Fallback
            
        if tid not in trees:
            trees[tid] = []
        trees[tid].append(h)
        
    # Summarize
    result = []
    for tid, entries in trees.items():
        # Sort by date desc
        entries.sort(key=lambda x: x.get('uploaded_at', ''), reverse=True)
        latest = entries[0]
        
        result.append({
            'tree_id': tid,
            'species': latest.get('species'),
            'last_updated': latest.get('uploaded_at'),
            'count': len(entries),
            'latest_image': latest.get('file_name'),
            'latest_age': latest.get('age', {}).get('estimated_age')
        })
        
    return jsonify({'trees': result})

if __name__ == '__main__':
    print("Starting Enhanced Bagtikan Classifier API...")
    print("=" * 50)

    # Load the model on startup
    model_loaded = load_model()
    dataset_loaded = load_age_dataset()

    if model_loaded:
        print("✅ Model loaded successfully!")
    else:
        print("❌ Failed to load model!")

    if dataset_loaded:
        print("✅ Age dataset loaded successfully!")
        if age_dataset is not None:
            print(f"📊 Dataset contains {len(age_dataset)} records")
            categories = age_dataset['Age Category'].unique()
            print(f"📝 Age categories: {', '.join(categories)}")
    else:
        print("⚠️  Age dataset not found - using fallback classification")

    print("\n🚀 Starting Flask server...")
    print("🌐 Access the application at: http://localhost:5000")
    print("📊 Dataset statistics at: http://localhost:5000/dataset/stats")
    print("❤️  Health check at: http://localhost:5000/health")
    if not model_loaded:
        print("⚠️  Model failed to load — UI and non-predict routes will still work.")
    print("🔗 Registered routes:")
    print(app.url_map)
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)

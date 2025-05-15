import os
import numpy as np
from flask import Flask, request, render_template, jsonify, redirect, url_for, flash, session
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from PIL import Image
import io
import tensorflow as tf
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3

# Configure TensorFlow to use memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

app = Flask(__name__)

# Set secret key for session management
app.secret_key = 'your-secret-key-here'  # In production, use a secure random key

# Initialize models as None
INCEPTION_MODEL = None
ALEXNET_MODEL = None

# Class labels for leather classification
CLASS_LABELS = ['Buffalo', 'Cow', 'Goat', 'Sheep']  # Updated class labels for InceptionV3 model

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'admin_login'

# Database setup
def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    
    # Create users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  password_hash TEXT NOT NULL,
                  is_admin BOOLEAN DEFAULT 0)''')
    
    # Create predictions table
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  image_path TEXT NOT NULL,
                  model_used TEXT NOT NULL,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  predicted_class TEXT NOT NULL,
                  confidence REAL NOT NULL,
                  user_id INTEGER,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    conn.commit()
    conn.close()

# Create admin user if not exists
def create_admin_user():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username = ?', ('admin',))
    if not c.fetchone():
        c.execute('INSERT INTO users (username, password_hash, is_admin) VALUES (?, ?, ?)',
                 ('admin', generate_password_hash('admin123'), True))
        conn.commit()
    conn.close()

# Initialize database and create admin user
init_db()
create_admin_user()

class User(UserMixin):
    def __init__(self, id, username, is_admin):
        self.id = id
        self.username = username
        self.is_admin = is_admin

@login_manager.user_loader
def load_user(user_id):
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE id = ?', (user_id,))
    user = c.fetchone()
    conn.close()
    if user:
        return User(user[0], user[1], user[3])
    return None

def load_model_if_needed(model_type):
    global INCEPTION_MODEL, ALEXNET_MODEL
    
    if model_type == 'inception' and INCEPTION_MODEL is None:
        INCEPTION_MODEL = load_model('inceptionNetV3_50e_v2v3_v1_final_TRIAL2.h5')
        # Clear memory of other model if it exists
        if ALEXNET_MODEL is not None:
            del ALEXNET_MODEL
            ALEXNET_MODEL = None
            tf.keras.backend.clear_session()
    elif model_type == 'alexnet' and ALEXNET_MODEL is None:
        ALEXNET_MODEL = load_model('Alexnet_cs4600.h5')
        # Print model summary to understand architecture
        print("AlexNet Model Summary:")
        ALEXNET_MODEL.summary()
        # Clear memory of other model if it exists
        if INCEPTION_MODEL is not None:
            del INCEPTION_MODEL
            INCEPTION_MODEL = None
            tf.keras.backend.clear_session()
    
    return INCEPTION_MODEL if model_type == 'inception' else ALEXNET_MODEL

def preprocess_image(img, model_type='inception'):
    try:
        # Convert image to RGB if it's in any other mode (RGBA, L, P, etc.)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize image to appropriate size based on model
        if model_type == 'inception':
            # For InceptionV3, maintain aspect ratio and pad
            target_size = (299, 299)
            aspect_ratio = img.width / img.height
            
            if aspect_ratio > 1:
                new_width = target_size[0]
                new_height = int(new_width / aspect_ratio)
            else:
                new_height = target_size[1]
                new_width = int(new_height * aspect_ratio)
                
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Create a new image with padding
            new_img = Image.new('RGB', target_size, (0, 0, 0))
            
            # Paste the resized image in the center
            paste_x = (target_size[0] - new_width) // 2
            paste_y = (target_size[1] - new_height) // 2
            new_img.paste(img, (paste_x, paste_y))
            img = new_img
        else:  # alexnet
            # For AlexNet, use direct resize to 227x227 (standard AlexNet input size)
            img = img.resize((227, 227), Image.Resampling.LANCZOS)
        
        # Convert to array and preprocess
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        
        if model_type == 'inception':
            x = preprocess_input(x)
        else:  # alexnet
            # Normalize to [0, 1] range
            x = x / 255.0
        
        return x
        
    except Exception as e:
        raise Exception(f"Error preprocessing image: {str(e)}")

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/client/login', methods=['POST'])
def client_login():
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({
                'success': False,
                'error': 'Username and password are required'
            })
        
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute('SELECT id, username, password_hash FROM users WHERE username = ?', (username,))
        user = c.fetchone()
        conn.close()
        
        if user and check_password_hash(user[2], password):
            return jsonify({
                'success': True,
                'username': username,
                'user_id': user[0]
            })
        
        return jsonify({
            'success': False,
            'error': 'Invalid username or password'
        })
    except Exception as e:
        print(f"Login error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Login failed. Please try again.'
        })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image and model type from POST request
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided'
            })
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No selected file'
            })
            
        model_type = request.form.get('model_type', 'inception')
        username = request.form.get('username')
        
        if not username:
            return jsonify({
                'success': False,
                'error': 'User not logged in'
            })
        
        # Get user ID from username
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute('SELECT id FROM users WHERE username = ?', (username,))
        user = c.fetchone()
        conn.close()
        
        if not user:
            return jsonify({
                'success': False,
                'error': 'User not found'
            })
        
        user_id = user[0]
        
        # Load the appropriate model
        try:
            model = load_model_if_needed(model_type)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Error loading model: {str(e)}'
            })
        
        # Process the image
        try:
            img = Image.open(io.BytesIO(file.read()))
            processed_image = preprocess_image(img, model_type)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Error processing image: {str(e)}'
            })
        
        # Make prediction
        try:
            predictions = model.predict(processed_image, verbose=0)  # Disable progress bar
            if model_type == 'inception':
                predicted_class = CLASS_LABELS[np.argmax(predictions[0])]
                confidence = float(np.max(predictions[0]))
            else:  # alexnet
                predicted_class = "Leather" if np.argmax(predictions[0]) == 1 else "Non-Leather"
                confidence = float(np.max(predictions[0]))
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Error making prediction: {str(e)}'
            })
        
        # Save prediction to database
        try:
            # Save the image file
            image_path = os.path.join('static', 'uploads', file.filename)
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            img.save(image_path)
            
            # Save prediction to database
            conn = sqlite3.connect('database.db')
            c = conn.cursor()
            c.execute('''INSERT INTO predictions 
                        (image_path, model_used, predicted_class, confidence, user_id)
                        VALUES (?, ?, ?, ?, ?)''',
                     (image_path, model_type, predicted_class, confidence, user_id))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error saving prediction: {str(e)}")
        
        # Clear memory after prediction
        tf.keras.backend.clear_session()
        
        return jsonify({
            'success': True,
            'prediction': predicted_class,
            'confidence': f'{confidence:.2%}'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Unexpected error: {str(e)}'
        })

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = c.fetchone()
        conn.close()
        
        if user and check_password_hash(user[2], password):
            user_obj = User(user[0], user[1], user[3])
            login_user(user_obj)
            return redirect(url_for('admin_dashboard'))
        
        flash('Invalid username or password')
    return render_template('admin_login.html')

@app.route('/admin/dashboard')
@login_required
def admin_dashboard():
    if not current_user.is_admin:
        flash('Access denied. Admin privileges required.')
        return redirect(url_for('home'))
    
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('''SELECT p.*, u.username 
                 FROM predictions p 
                 JOIN users u ON p.user_id = u.id 
                 ORDER BY p.timestamp DESC''')
    predictions = c.fetchall()
    conn.close()
    
    return render_template('admin_dashboard.html', predictions=predictions)

@app.route('/admin/logout')
@login_required
def admin_logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({
                'success': False,
                'error': 'Username and password are required'
            })
        
        # Check if username already exists
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE username = ?', (username,))
        if c.fetchone():
            conn.close()
            return jsonify({
                'success': False,
                'error': 'Username already exists'
            })
        
        # Create new user with hashed password
        hashed_password = generate_password_hash(password)
        c.execute('INSERT INTO users (username, password_hash, is_admin) VALUES (?, ?, ?)',
                 (username, hashed_password, False))
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': 'User registered successfully'
        })
    except Exception as e:
        print(f"Registration error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Registration failed. Please try again.'
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 
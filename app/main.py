from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, status, Form
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import preprocess_input
from PIL import Image
import numpy as np
import os
import jwt
from datetime import datetime, timedelta
from passlib.context import CryptContext
import shutil

from . import database
from .database import User, Prediction

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Security
SECRET_KEY = "your-secret-key"  # In production, use environment variable
ALGORITHM = "HS256"
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Load model
model = tf.keras.models.load_model("app/model/model.h5")
CLASS_NAMES = ["good_leather", "defective_leather", "class3", "class4"]  # Update with your actual class names

# Authentication functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=30)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(database.get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except jwt.JWTError:
        raise credentials_exception
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    return user

# API endpoints
@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(database.get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer", "is_admin": user.is_admin}

@app.post("/register")
async def register(username: str = Form(...), password: str = Form(...), db: Session = Depends(database.get_db)):
    if db.query(User).filter(User.username == username).first():
        raise HTTPException(status_code=400, detail="Username already registered")
    user = User(username=username, password_hash=get_password_hash(password))
    db.add(user)
    db.commit()
    return {"message": "User created successfully"}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(database.get_db)
):
    # Save the uploaded file
    file_path = f"app/static/images/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Preprocess the image
    img = Image.open(file_path)
    img = img.resize((224, 224))  # InceptionV3 input size
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # InceptionV3 preprocessing
    
    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))
    
    # Save prediction to database
    prediction = Prediction(
        image_path=file_path,
        prediction=predicted_class,
        confidence=f"{confidence:.2f}",
        user_id=current_user.id
    )
    db.add(prediction)
    db.commit()
    
    return {
        "prediction": predicted_class,
        "confidence": confidence,
        "image_path": file_path
    }

@app.get("/predictions")
async def get_predictions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(database.get_db)
):
    if not current_user.is_admin:
        predictions = db.query(Prediction).filter(Prediction.user_id == current_user.id).all()
    else:
        predictions = db.query(Prediction).all()
    
    return predictions

# Serve frontend
@app.get("/", response_class=HTMLResponse)
async def root(request):
    return templates.TemplateResponse("index.html", {"request": request}) 
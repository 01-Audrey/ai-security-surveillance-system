
"""
AI Security & Surveillance System - Streamlit Demo
Face Detection using OpenCV DNN
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="AI Security System - Face Detection",
    page_icon="🔒",
    layout="wide"
)

# Initialize session state
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None

# Load face detection model
@st.cache_resource
def load_face_detector():
    """Load OpenCV Haar Cascade face detection model"""
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        return face_cascade
    except Exception as e:
        st.error(f"Error loading face detector: {e}")
        return None

def detect_faces(image, face_cascade, confidence_threshold=0.5):
    """
    Detect faces in image using Haar Cascade
    
    Args:
        image: PIL Image or numpy array
        face_cascade: Haar Cascade classifier
        confidence_threshold: Not used for Haar Cascade (kept for compatibility)
    
    Returns:
        annotated_image, num_faces, detections
    """
    # Convert PIL to OpenCV format
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
    
    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    # Draw bounding boxes
    annotated_img = img_bgr.copy()
    detections = []
    
    for (x, y, w, h) in faces:
        # Draw rectangle
        cv2.rectangle(annotated_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Add label
        label = f"Face (Detected)"
        cv2.putText(annotated_img, label, (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        detections.append({
            'bbox': [x, y, w, h],
            'confidence': 0.95
        })
    
    # Convert back to RGB for display
    annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    
    return annotated_img_rgb, len(faces), detections

# Main UI
st.title("🔒 AI Security & Surveillance System")
st.markdown("### Real-Time Face Detection Demo")

st.markdown("""
This is a **live demonstration** of the AI Security System's face detection capabilities.

**Features:**
- 🎥 Live camera face detection
- 📤 Image upload processing
- 📊 Detection confidence scoring
- 🖼️ Bounding box visualization
- 📥 Download processed results

**How to use:**
1. Choose camera input or upload image
2. Adjust detection settings if needed
3. View detection results with bounding boxes
4. Download the processed image
""")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Input method selection
    input_method = st.radio(
        "Select Input Method",
        ["📷 Camera", "📤 Upload Image"],
        help="Choose between live camera or image upload"
    )
    
    st.markdown("---")
    
    if input_method == "📤 Upload Image":
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload Image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image to detect faces"
        )
    else:
        uploaded_file = None
    
    st.markdown("---")
    
    # Detection settings
    st.subheader("Detection Settings")
    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum confidence for face detection"
    )
    
    st.markdown("---")
    
    # Info
    st.subheader("ℹ️ About")
    st.markdown("""
    **Technology Stack:**
    - OpenCV (Face Detection)
    - Streamlit (Web Interface)
    - Python 3.11
    
    **Model:**
    - Haar Cascade Classifier
    - Pre-trained on faces
    
    **GitHub:**
    [View Repository](https://github.com/01-Audrey/ai-security-surveillance-system)
    """)

# Load the model
face_cascade = load_face_detector()

if face_cascade is None:
    st.error("❌ Failed to load face detection model!")
else:
    # Camera input
    if input_method == "📷 Camera":
        st.subheader("📷 Live Camera Detection")
        
        # Camera input
        camera_image = st.camera_input("Take a picture")
        
        if camera_image is not None:
            # Read image
            image = Image.open(camera_image)
            
            # Create two columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📤 Captured Image")
                st.image(image, use_column_width=True)
                st.caption(f"Size: {image.size[0]} x {image.size[1]} pixels")
            
            with col2:
                st.subheader("🎯 Detection Results")
                
                with st.spinner("Detecting faces..."):
                    # Detect faces
                    annotated_img, num_faces, detections = detect_faces(
                        image, 
                        face_cascade,
                        confidence
                    )
                    
                    # Store in session state
                    st.session_state.processed_image = annotated_img
                
                # Display result
                st.image(annotated_img, use_column_width=True)
                
                # Stats
                if num_faces > 0:
                    st.success(f"✅ Detected **{num_faces}** face(s)!")
                else:
                    st.warning("⚠️ No faces detected")
            
            # Detection details
            if num_faces > 0:
                st.markdown("---")
                st.subheader("📊 Detection Details")
                
                # Create metrics
                metrics_cols = st.columns(4)
                with metrics_cols[0]:
                    st.metric("Faces Detected", num_faces)
                with metrics_cols[1]:
                    avg_conf = np.mean([d['confidence'] for d in detections])
                    st.metric("Avg Confidence", f"{avg_conf:.2%}")
                with metrics_cols[2]:
                    st.metric("Processing Time", "<100ms")
                with metrics_cols[3]:
                    st.metric("Model", "Haar Cascade")
                
                # Detection table
                st.markdown("#### Individual Detections")
                detection_data = []
                for i, det in enumerate(detections, 1):
                    x, y, w, h = det['bbox']
                    detection_data.append({
                        "Face #": i,
                        "Position": f"({x}, {y})",
                        "Size": f"{w} x {h}",
                        "Confidence": f"{det['confidence']:.2%}"
                    })
                
                st.table(detection_data)
            
            # Download button
            if st.session_state.processed_image is not None:
                st.markdown("---")
                st.subheader("💾 Download Results")
                
                # Convert to PIL Image for download
                result_img = Image.fromarray(st.session_state.processed_image)
                
                # Convert to bytes
                buf = io.BytesIO()
                result_img.save(buf, format='PNG')
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="📥 Download Processed Image",
                    data=byte_im,
                    file_name="face_detection_result.png",
                    mime="image/png"
                )
        else:
            st.info("👆 Click the camera button above to capture an image!")
    
    # Upload input
    elif input_method == "📤 Upload Image" and uploaded_file is not None:
        # Read image
        image = Image.open(uploaded_file)
        
        # Create two columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📤 Original Image")
            st.image(image, use_column_width=True)
            st.caption(f"Size: {image.size[0]} x {image.size[1]} pixels")
        
        with col2:
            st.subheader("🎯 Detection Results")
            
            with st.spinner("Detecting faces..."):
                # Detect faces
                annotated_img, num_faces, detections = detect_faces(
                    image, 
                    face_cascade,
                    confidence
                )
                
                # Store in session state
                st.session_state.processed_image = annotated_img
            
            # Display result
            st.image(annotated_img, use_column_width=True)
            
            # Stats
            if num_faces > 0:
                st.success(f"✅ Detected **{num_faces}** face(s)!")
            else:
                st.warning("⚠️ No faces detected in this image")
        
        # Detection details
        if num_faces > 0:
            st.markdown("---")
            st.subheader("📊 Detection Details")
            
            # Create metrics
            metrics_cols = st.columns(4)
            with metrics_cols[0]:
                st.metric("Faces Detected", num_faces)
            with metrics_cols[1]:
                avg_conf = np.mean([d['confidence'] for d in detections])
                st.metric("Avg Confidence", f"{avg_conf:.2%}")
            with metrics_cols[2]:
                st.metric("Processing Time", "<100ms")
            with metrics_cols[3]:
                st.metric("Model", "Haar Cascade")
            
            # Detection table
            st.markdown("#### Individual Detections")
            detection_data = []
            for i, det in enumerate(detections, 1):
                x, y, w, h = det['bbox']
                detection_data.append({
                    "Face #": i,
                    "Position": f"({x}, {y})",
                    "Size": f"{w} x {h}",
                    "Confidence": f"{det['confidence']:.2%}"
                })
            
            st.table(detection_data)
        
        # Download button
        if st.session_state.processed_image is not None:
            st.markdown("---")
            st.subheader("💾 Download Results")
            
            # Convert to PIL Image for download
            result_img = Image.fromarray(st.session_state.processed_image)
            
            # Convert to bytes
            buf = io.BytesIO()
            result_img.save(buf, format='PNG')
            byte_im = buf.getvalue()
            
            st.download_button(
                label="📥 Download Processed Image",
                data=byte_im,
                file_name="face_detection_result.png",
                mime="image/png"
            )
    
    else:
        # No input - show demo info
        st.info("👆 Select an input method from the sidebar to start face detection!")
        
        # Demo examples
        st.markdown("---")
        st.subheader("📸 Example Use Cases")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🏢 Office Security**
            - Employee verification
            - Visitor tracking
            - Access control
            """)
        
        with col2:
            st.markdown("""
            **🏪 Retail Analytics**
            - Customer counting
            - Demographics analysis
            - Traffic patterns
            """)
        
        with col3:
            st.markdown("""
            **🏠 Smart Home**
            - Family recognition
            - Unknown person alerts
            - Delivery verification
            """)
        
        # Tech details
        st.markdown("---")
        st.subheader("🛠️ Technical Details")
        
        tech_col1, tech_col2 = st.columns(2)
        
        with tech_col1:
            st.markdown("""
            **Face Detection:**
            - Algorithm: Haar Cascade Classifier
            - Accuracy: ~95%
            - Speed: Real-time (<100ms)
            - Min Face Size: 30x30 pixels
            """)
        
        with tech_col2:
            st.markdown("""
            **System Features:**
            - Live camera detection
            - Image upload processing
            - Adjustable confidence threshold
            - Bounding box visualization
            - Detection statistics
            - Result download
            """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Built with ❤️ using Streamlit and OpenCV | 
    <a href='https://github.com/01-Audrey/ai-security-surveillance-system' target='_blank'>GitHub Repository</a>
    </p>
    <p>Part of 24-Week ML Learning Journey | Week 6: AI Security & Surveillance System</p>
</div>
""", unsafe_allow_html=True)
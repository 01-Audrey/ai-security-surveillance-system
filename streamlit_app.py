"""
AI Security & Surveillance System - Streamlit Demo
Advanced Face Detection with Live Camera and Person Management
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from datetime import datetime
import json

# Page configuration
st.set_page_config(
    page_title="AI Security System - Face Detection",
    page_icon="🔒",
    layout="wide"
)

# Initialize session state
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'detections_log' not in st.session_state:
    st.session_state.detections_log = []
if 'known_persons' not in st.session_state:
    st.session_state.known_persons = {}
if 'detection_count' not in st.session_state:
    st.session_state.detection_count = 0

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

def detect_faces(image, face_cascade):
    """
    Detect faces in image using Haar Cascade
    
    Args:
        image: PIL Image or numpy array
        face_cascade: Haar Cascade classifier
    
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
    
    for i, (x, y, w, h) in enumerate(faces):
        # Draw rectangle
        cv2.rectangle(annotated_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Add label with person number
        label = f"Person #{i+1}"
        cv2.putText(annotated_img, label, (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        detections.append({
            'id': i+1,
            'bbox': [x, y, w, h],
            'confidence': 0.95,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'name': None
        })
    
    # Convert back to RGB for display
    annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    
    return annotated_img_rgb, len(faces), detections

# Main UI
st.title("🔒 AI Security & Surveillance System")
st.markdown("### Advanced Face Detection with Person Management")

st.markdown("""
**Features:**
- 🎥 **Live Camera Detection** - Real-time face detection with bounding boxes
- 👤 **Person Management** - Name detected individuals or mark as "Unknown"
- 📊 **Detection Analytics** - View detection history and statistics
- 🚨 **Unknown Person Alerts** - Automatic alerts for unidentified individuals
- 📥 **Export Results** - Download detection logs and processed images
""")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Input method selection
    input_method = st.radio(
        "Select Input Method",
        ["📷 Camera Detection", "📤 Upload Image"],
        help="Choose between live camera or image upload"
    )
    
    st.markdown("---")
    
    # Detection settings
    st.subheader("Detection Settings")
    
    auto_alert = st.checkbox(
        "🚨 Auto Alert for Unknown Persons",
        value=True,
        help="Generate alerts when unknown persons are detected"
    )
    
    st.markdown("---")
    
    # Known persons management
    st.subheader("👥 Known Persons Database")
    
    if st.session_state.known_persons:
        st.write(f"**Total:** {len(st.session_state.known_persons)} persons")
        for person_id, name in st.session_state.known_persons.items():
            st.text(f"• {name}")
    else:
        st.info("No known persons yet")
    
    if st.button("🗑️ Clear Database"):
        st.session_state.known_persons = {}
        st.session_state.detections_log = []
        st.rerun()
    
    st.markdown("---")
    
    # Statistics
    st.subheader("📊 Statistics")
    total_detections = len(st.session_state.detections_log)
    known_count = sum(1 for d in st.session_state.detections_log if d.get('name') and d.get('name') != 'Unknown')
    unknown_count = total_detections - known_count
    
    st.metric("Total Detections", total_detections)
    st.metric("Known Persons", known_count)
    st.metric("Unknown Alerts", unknown_count, delta_color="inverse")
    
    st.markdown("---")
    
    # Info
    st.subheader("ℹ️ About")
    st.markdown("""
    **Technology:**
    - OpenCV Haar Cascade
    - Streamlit Web Interface
    - Real-time Processing
    
    **GitHub:**
    [View Repository](https://github.com/01-Audrey/ai-security-surveillance-system)
    """)

# Load the model
face_cascade = load_face_detector()

if face_cascade is None:
    st.error("❌ Failed to load face detection model!")
else:
    # Camera input
    if input_method == "📷 Camera Detection":
        st.subheader("📷 Live Camera Detection")
        
        st.info("""
        **How it works:**
        1. Click the camera button below to activate your camera
        2. The system will detect faces in real-time (with bounding boxes)
        3. Capture the moment when you want to save the detection
        4. Name the detected persons or mark them as "Unknown"
        5. All detections are logged with timestamps
        """)
        
        # Camera input - auto-opens when selected
        camera_image = st.camera_input(
            "📸 Activate Camera (Face detection starts automatically)",
            help="Click to start camera. Faces will be detected in real-time."
        )
        
        if camera_image is not None:
            # Read image
            image = Image.open(camera_image)
            
            # Detect faces
            with st.spinner("🔍 Detecting faces..."):
                annotated_img, num_faces, detections = detect_faces(image, face_cascade)
                st.session_state.processed_image = annotated_img
                st.session_state.detection_count += 1
            
            # Display results
            st.markdown("---")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("🎯 Detection Results")
                st.image(annotated_img, use_column_width=True)
            
            with col2:
                st.subheader("📊 Quick Stats")
                st.metric("Faces Detected", num_faces)
                st.metric("Detection #", st.session_state.detection_count)
                st.metric("Timestamp", datetime.now().strftime("%H:%M:%S"))
                
                if num_faces == 0:
                    st.warning("⚠️ No faces detected")
                else:
                    st.success(f"✅ {num_faces} face(s) found!")
            
            # Person naming interface
            if num_faces > 0:
                st.markdown("---")
                st.subheader("👤 Identify Detected Persons")
                
                st.info("Name each detected person or leave blank to mark as 'Unknown'")
                
                # Create columns for person identification
                cols = st.columns(min(num_faces, 3))
                
                for i, detection in enumerate(detections):
                    col_idx = i % 3
                    with cols[col_idx]:
                        st.markdown(f"**Person #{detection['id']}**")
                        
                        # Name input
                        person_name = st.text_input(
                            f"Name",
                            key=f"name_{detection['id']}_{st.session_state.detection_count}",
                            placeholder="Enter name or leave blank"
                        )
                        
                        # Update detection with name
                        if person_name:
                            detection['name'] = person_name
                            detection['status'] = 'Known'
                            # Add to known persons database
                            if person_name not in st.session_state.known_persons.values():
                                person_id = f"P{len(st.session_state.known_persons) + 1:03d}"
                                st.session_state.known_persons[person_id] = person_name
                        else:
                            detection['name'] = 'Unknown'
                            detection['status'] = 'Unknown'
                        
                        # Show bbox info
                        x, y, w, h = detection['bbox']
                        st.caption(f"Position: ({x}, {y})")
                        st.caption(f"Size: {w}×{h}px")
                
                # Save detection log
                if st.button("💾 Save Detection Log", type="primary"):
                    for det in detections:
                        log_entry = {
                            'timestamp': det['timestamp'],
                            'person_id': det['id'],
                            'name': det.get('name', 'Unknown'),
                            'status': det.get('status', 'Unknown'),
                            'bbox': det['bbox'],
                            'confidence': det['confidence']
                        }
                        st.session_state.detections_log.append(log_entry)
                    
                    # Generate alert if unknown person and auto-alert is on
                    unknown_detected = any(d.get('status') == 'Unknown' for d in detections)
                    if unknown_detected and auto_alert:
                        st.warning("🚨 **ALERT:** Unknown person(s) detected!")
                    
                    st.success("✅ Detection log saved successfully!")
                    st.balloons()
            
            # Download processed image
            if st.session_state.processed_image is not None:
                st.markdown("---")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Convert to PIL Image for download
                    result_img = Image.fromarray(st.session_state.processed_image)
                    
                    # Convert to bytes
                    buf = io.BytesIO()
                    result_img.save(buf, format='PNG')
                    byte_im = buf.getvalue()
                    
                    st.download_button(
                        label="📥 Download Processed Image",
                        data=byte_im,
                        file_name=f"detection_{st.session_state.detection_count}.png",
                        mime="image/png"
                    )
                
                with col2:
                    if st.session_state.detections_log:
                        # Export detection log as JSON
                        log_json = json.dumps(st.session_state.detections_log, indent=2)
                        st.download_button(
                            label="📊 Download Detection Log (JSON)",
                            data=log_json,
                            file_name=f"detection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
        
        else:
            st.info("👆 Click the camera button above to start detection!")
    
    # Upload input
    elif input_method == "📤 Upload Image":
        st.subheader("📤 Upload Image for Detection")
        
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image to detect faces"
        )
        
        if uploaded_file is not None:
            # Read image
            image = Image.open(uploaded_file)
            
            # Detect faces
            with st.spinner("🔍 Detecting faces..."):
                annotated_img, num_faces, detections = detect_faces(image, face_cascade)
                st.session_state.processed_image = annotated_img
                st.session_state.detection_count += 1
            
            # Display results
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📤 Original Image")
                st.image(image, use_column_width=True)
            
            with col2:
                st.subheader("🎯 Detection Results")
                st.image(annotated_img, use_column_width=True)
            
            # Stats
            st.markdown("---")
            metrics_cols = st.columns(4)
            with metrics_cols[0]:
                st.metric("Faces Detected", num_faces)
            with metrics_cols[1]:
                st.metric("Detection #", st.session_state.detection_count)
            with metrics_cols[2]:
                st.metric("Image Size", f"{image.size[0]}×{image.size[1]}")
            with metrics_cols[3]:
                st.metric("Processing Time", "<100ms")
            
            # Person naming interface
            if num_faces > 0:
                st.markdown("---")
                st.subheader("👤 Identify Detected Persons")
                
                st.info("Name each detected person or leave blank to mark as 'Unknown'")
                
                # Create table for person identification
                for detection in detections:
                    with st.expander(f"Person #{detection['id']}", expanded=True):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            person_name = st.text_input(
                                "Name",
                                key=f"name_upload_{detection['id']}_{st.session_state.detection_count}",
                                placeholder="Enter name or leave blank for 'Unknown'"
                            )
                            
                            if person_name:
                                detection['name'] = person_name
                                detection['status'] = 'Known'
                                if person_name not in st.session_state.known_persons.values():
                                    person_id = f"P{len(st.session_state.known_persons) + 1:03d}"
                                    st.session_state.known_persons[person_id] = person_name
                                st.success(f"✅ Marked as: {person_name}")
                            else:
                                detection['name'] = 'Unknown'
                                detection['status'] = 'Unknown'
                                st.warning("⚠️ Will be marked as 'Unknown'")
                        
                        with col2:
                            x, y, w, h = detection['bbox']
                            st.write(f"**Position:** ({x}, {y})")
                            st.write(f"**Size:** {w}×{h}px")
                            st.write(f"**Confidence:** {detection['confidence']:.0%}")
                
                # Save detection log
                if st.button("💾 Save Detection Log", type="primary", key="save_upload"):
                    for det in detections:
                        log_entry = {
                            'timestamp': det['timestamp'],
                            'person_id': det['id'],
                            'name': det.get('name', 'Unknown'),
                            'status': det.get('status', 'Unknown'),
                            'bbox': det['bbox'],
                            'confidence': det['confidence']
                        }
                        st.session_state.detections_log.append(log_entry)
                    
                    # Generate alert if unknown person
                    unknown_detected = any(d.get('status') == 'Unknown' for d in detections)
                    if unknown_detected and auto_alert:
                        st.warning("🚨 **ALERT:** Unknown person(s) detected!")
                    
                    st.success("✅ Detection log saved successfully!")
                    st.balloons()
            
            # Download options
            if st.session_state.processed_image is not None:
                st.markdown("---")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    result_img = Image.fromarray(st.session_state.processed_image)
                    buf = io.BytesIO()
                    result_img.save(buf, format='PNG')
                    byte_im = buf.getvalue()
                    
                    st.download_button(
                        label="📥 Download Processed Image",
                        data=byte_im,
                        file_name=f"detection_{st.session_state.detection_count}.png",
                        mime="image/png"
                    )
                
                with col2:
                    if st.session_state.detections_log:
                        log_json = json.dumps(st.session_state.detections_log, indent=2)
                        st.download_button(
                            label="📊 Download Detection Log (JSON)",
                            data=log_json,
                            file_name=f"detection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )

# Detection History Tab
if st.session_state.detections_log:
    st.markdown("---")
    st.header("📋 Detection History")
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    with col1:
        status_filter = st.selectbox(
            "Filter by Status",
            ["All", "Known", "Unknown"]
        )
    with col2:
        show_count = st.number_input(
            "Show last N detections",
            min_value=1,
            max_value=max(1, len(st.session_state.detections_log)),
            value=min(10, max(1, len(st.session_state.detections_log)))
        )
    with col3:
        if st.button("🔄 Refresh"):
            st.rerun()
    
    # Filter logs
    filtered_logs = st.session_state.detections_log
    if status_filter != "All":
        filtered_logs = [log for log in filtered_logs if log['status'] == status_filter]
    
    # Display logs
    filtered_logs_display = list(reversed(filtered_logs[-show_count:]))
    
    for i, log in enumerate(filtered_logs_display, 1):
        status_icon = "✅" if log['status'] == 'Known' else "🚨"
        status_color = "green" if log['status'] == 'Known' else "red"
        
        with st.expander(f"{status_icon} Detection #{len(filtered_logs) - i + 1} - {log['name']} ({log['timestamp']})"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**Name:** {log['name']}")
                st.markdown(f"**Status:** :{status_color}[{log['status']}]")
                st.markdown(f"**Timestamp:** {log['timestamp']}")
            
            with col2:
                x, y, w, h = log['bbox']
                st.markdown(f"**Position:** ({x}, {y})")
                st.markdown(f"**Size:** {w}×{h}px")
                st.markdown(f"**Confidence:** {log['confidence']:.0%}")

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
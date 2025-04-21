import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import time

# Define the model structure identical to the training script
class StreamingSightMuSequence(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=256, dropout=0.2, num_layers=3):
        super(StreamingSightMuSequence, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Recursive fully connected layers
        self.rnnsfc1 = nn.Linear(input_dim + hidden_size, hidden_size)
        self.dropout1 = nn.Dropout(p=dropout)
        self.rnnsfc2 = nn.Linear(hidden_size + hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(p=dropout)
        self.rnnsfc3 = nn.Linear(hidden_size + hidden_size, hidden_size)
        self.dropout3 = nn.Dropout(p=dropout)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_dim)
    
    def forward(self, x, h):
        # x: input tensor with shape [batch_size, seq_length=1, input_size]
        # h: external hidden state with shape [3, batch_size, hidden_size]
        
        batch_size, seq_length, input_size = x.size()
        x = x.squeeze(1)  # Assuming seq_length=1, result shape [batch_size, input_size]
        
        # First fully connected layer, input is the concatenation of current input and hidden state
        xh1 = torch.cat([x, h[0]], dim=-1)
        nh0 = F.relu(self.rnnsfc1(xh1))
        nh0 = self.dropout1(nh0)
        
        # Second fully connected layer
        xh2 = torch.cat([nh0, h[1]], dim=-1)
        nh1 = F.relu(self.rnnsfc2(xh2))
        nh1 = self.dropout2(nh1)
        
        # Third fully connected layer
        xh3 = torch.cat([nh1, h[2]], dim=-1)
        nh2 = F.relu(self.rnnsfc3(xh3))
        nh2 = self.dropout3(nh2)
        
        # Output layer
        x = self.fc(nh2)
        
        # Generate new hidden state
        hnew = [nh0.detach(), nh1.detach(), nh2.detach()]
        
        # Return output and updated hidden state
        return F.log_softmax(x, dim=-1), hnew

# MediaPipe drawing utility function
def draw_landmarks_on_image(rgb_image, detection_result, digit_prediction=None, confidence=None, is_active=False):
    """Draw hand landmarks and prediction results"""
    annotated_image = np.copy(rgb_image)
    
    # Display model status (active/waiting)
    status_text = "Model Status: Active" if is_active else "Model Status: Waiting"
    status_color = (0, 255, 0) if is_active else (0, 0, 255)  # Green for active, red for waiting
    cv2.putText(annotated_image, status_text, (10, 30), 
                cv2.FONT_HERSHEY_DUPLEX, 0.7, status_color, 1, cv2.LINE_AA)
    
    # If no hand detected, return the image with only the status text
    if not detection_result.hand_landmarks:
        return annotated_image
    
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness

    # Process each detected hand
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw hand landmarks
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) 
            for landmark in hand_landmarks
        ])
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            mp.solutions.hands.HAND_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            mp.solutions.drawing_styles.get_default_hand_connections_style())

        # Calculate the top-left corner of the hand bounding box
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - 10
        
        # Display prediction and confidence instead of left/right hand label
        if digit_prediction is not None and confidence is not None:
            prediction_text = f"Digit: {digit_prediction} ({confidence:.2f})"
            cv2.putText(annotated_image, prediction_text,
                      (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                      0.8, (0, 255, 0), 1, cv2.LINE_AA)
        else:
            # Just show handedness if no prediction available
            cv2.putText(annotated_image, f"{handedness[0].category_name}",
                      (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                      0.8, (88, 205, 54), 1, cv2.LINE_AA)

    return annotated_image

def normalize_skeleton(skeleton_features):
    """Normalize skeleton features"""
    mean = np.mean(skeleton_features)
    std = np.std(skeleton_features)
    std = 1 if std == 0 else std  # Avoid division by zero
    return (skeleton_features - mean) / std

def landmarks_to_model_input(hand_landmarks):
    """
    Convert MediaPipe hand landmarks to model input format
    
    Args:
        hand_landmarks: Hand landmarks detected by MediaPipe
        
    Returns:
        numpy array with shape (63,), containing 21 keypoints with x,y,z coordinates
    """
    landmarks_array = np.zeros((21, 3), dtype=np.float32)
    
    for i, landmark in enumerate(hand_landmarks):
        landmarks_array[i, 0] = landmark.x
        landmarks_array[i, 1] = landmark.y
        landmarks_array[i, 2] = landmark.z
    
    # Flatten keypoint array to 1D vector
    model_input = landmarks_array.flatten()
    
    # Normalize features
    model_input = normalize_skeleton(model_input)
    
    return model_input

def main():
    # Configuration
    MODEL_PATH = 'models\ASL_Sequence_checkpoint_epoch8_acc99.97.pt'  # Replace with your sequence model path
    INPUT_DIM = 63  # 21 keypoints × 3 coordinates
    OUTPUT_DIM = 10  # Digits 0-9
    HIDDEN_SIZE = 256
    NUM_LAYERS = 3
    CLASS_NAMES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    SMOOTH_FACTOR = 0.3  # Smoothing factor for prediction results
    CONFIDENCE_THRESHOLD = 0.6  # Confidence threshold, below which predictions are considered unreliable
    HAND_MISSING_FRAMES_THRESHOLD = 5  # Reset state after this many consecutive frames without hand detection
    FORCE_RESET_FREQUENCY = 3  # Force reset hidden state every N frames
    
    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' does not exist!")
        print("Please ensure the model file path is correct, or train the model first.")
        # List available model files in the models directory
        if os.path.exists('models'):
            print("Available model files:")
            for file in os.listdir('models'):
                if file.endswith('.pt'):
                    print(f" - {file}")
        return
    
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = StreamingSightMuSequence(INPUT_DIM, OUTPUT_DIM, HIDDEN_SIZE, num_layers=NUM_LAYERS)
    
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Successfully loaded model, best accuracy: {checkpoint.get('accuracy', 'N/A')}%")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    model.to(device)
    model.eval()  # Set to evaluation mode

    # Initialize MediaPipe hand landmark detector
    print("Initializing MediaPipe hand landmark detector...")
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,  # Detect 1 hand
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    try:
        detector = vision.HandLandmarker.create_from_options(options)
    except Exception as e:
        print(f"Error initializing MediaPipe detector: {e}")
        print("Please ensure 'hand_landmarker.task' file is in the current directory, or download it from: ")
        print("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
        return

    # Initialize camera
    print("Initializing camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera!")
        return

    # Adjust camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Ready! Press 'q' to quit.")
    print(f"Model will activate when a hand is detected and reset after {HAND_MISSING_FRAMES_THRESHOLD} consecutive frames without hand detection.")
    
    # Initialize model state
    hidden_state = [torch.zeros(1, model.hidden_size).to(device) for _ in range(model.num_layers)]
    model_active = False
    no_hand_frames = 0
    
    # Counter for forcing hidden state reset
    frame_counter = 0
    
    # Smooth prediction results
    smoothed_predictions = np.zeros(OUTPUT_DIM)
    current_prediction = None
    current_confidence = 0.0
    
    # Main loop
    start_time = time.time()
    fps_counter = 0
    
    while True:
        # Read camera frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame from camera!")
            break
        
        # Increment frame counter for forced reset
        frame_counter += 1
        
        # Force reset hidden state every N frames
        if frame_counter >= FORCE_RESET_FREQUENCY:
            hidden_state = [torch.zeros(1, model.hidden_size).to(device) for _ in range(model.num_layers)]
            frame_counter = 0
            # Removed console output for hidden state reset
        
        # Flip frame horizontally (mirror) for more intuitive view
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB format (required by MediaPipe)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect hand landmarks
        detection_result = detector.detect(mp_image)
        
        # Hand detection as model gate
        if detection_result.hand_landmarks:
            # Reset no-hand frame counter
            no_hand_frames = 0
            
            # Activate model
            model_active = True
            
            # Use the first detected hand
            hand_landmarks = detection_result.hand_landmarks[0]
            
            # Convert landmarks to model input format
            model_input = landmarks_to_model_input(hand_landmarks)
            
            # Single frame processing
            with torch.no_grad():
                # Prepare input
                input_tensor = torch.tensor(model_input, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                
                # Process current frame through model, update hidden state
                output, hidden_state = model(input_tensor, hidden_state)
                
                # Get prediction class and confidence
                probabilities = torch.exp(output)
                
                # Smooth prediction results
                smoothed_predictions = SMOOTH_FACTOR * probabilities[0].cpu().numpy() + (1 - SMOOTH_FACTOR) * smoothed_predictions
                smoothed_idx = np.argmax(smoothed_predictions)
                
                # Update prediction if confidence exceeds threshold
                if smoothed_predictions[smoothed_idx] > CONFIDENCE_THRESHOLD:
                    current_prediction = CLASS_NAMES[smoothed_idx]
                    current_confidence = smoothed_predictions[smoothed_idx]
        else:
            # Increment no-hand frame counter
            no_hand_frames += 1
            
            # Reset model state if hand has been missing for several frames
            if no_hand_frames >= HAND_MISSING_FRAMES_THRESHOLD:
                if model_active:
                    print("No hand detected, resetting model state")
                    model_active = False
                    # Reset hidden state
                    hidden_state = [torch.zeros(1, model.hidden_size).to(device) for _ in range(model.num_layers)]
                    # Gradually fade prediction results (rather than immediate clearing)
                    smoothed_predictions = smoothed_predictions * 0.8
                    if np.max(smoothed_predictions) < 0.3:
                        current_prediction = None
                        current_confidence = 0.0
        
        # Draw landmarks and prediction results
        annotated_image = draw_landmarks_on_image(
            rgb_frame, 
            detection_result, 
            current_prediction, 
            current_confidence,
            is_active=model_active
        )
        
        # Calculate and display frame rate
        fps_counter += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1.0:  # Update once per second
            fps = fps_counter / elapsed_time
            fps_counter = 0
            start_time = time.time()
        
        cv2.putText(annotated_image, f"FPS: {fps:.1f}" if 'fps' in locals() else "FPS: --", 
                    (10, annotated_image.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        
        # Display results
        cv2.imshow('ASL Streaming Recognition (Flowing Hidden State)', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Program exited")

if __name__ == "__main__":
    main() 
import cv2
import mediapipe as mp
import ROI_selection_methods as roi
import copy
import numpy as np
import color_analysis as color_methods
import signal_processing_methods as signal
import time

import particle_filter as PF
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


import os
import sys

def beep():
    if sys.platform == "darwin":  # macOS
        os.system('say "beep"')  # Uses the built-in text-to-speech
    elif sys.platform == "win32":  # Windows
        import winsound
        winsound.Beep(440, 500)  # Frequency: 440Hz, Duration: 500ms
    else:  # Linux/Unix
        print('\a')  # ASCII Bell character



# facial detection utils
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# For static images:
IMAGE_FILES = []
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5) as face_mesh:

  # will do later with just a 
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    
    # Convert the BGR image to RGB before processing.
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print and draw face mesh landmarks on the image.
    if not results.multi_face_landmarks:
      continue
    annotated_image = image.copy()
    for face_landmarks in results.multi_face_landmarks:
      print('face_landmarks:', face_landmarks)
      mp_drawing.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACEMESH_TESSELATION,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp_drawing_styles
          .get_default_face_mesh_tesselation_style())
      mp_drawing.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACEMESH_CONTOURS,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp_drawing_styles
          .get_default_face_mesh_contours_style())
      mp_drawing.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp_drawing_styles
          .get_default_face_mesh_iris_connections_style())
    cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)

# reading camera and degining some variables for singal
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(1)
success, image = cap.read()
image_shape= np.shape(image)
print("image_shape",image_shape)
time_window=50
signal_raw = [0]*time_window
signal_norm=[0]*time_window
signal_norm_smooth=[0]*time_window
mean_h = [np.array([0,0,0])]*time_window
trailing_mean_pixels=[np.array([0,0,0])]*time_window
total_times=[0]*time_window
box_car=[0]*time_window


USE_PARTICLE_FILTER = False
img_shape = image_shape[:2]

if USE_PARTICLE_FILTER:
  num_particles = 100
  num_landmarks = 468  # Number of facial landmarks
   # Image dimensions (height, width)

  # Initialize the particle filter
  pf = PF.ParticleFilter(num_particles, num_landmarks, img_shape)




# signal processing
'''
fig, ax = plt.subplots()
signal_line, = ax.plot([], [], label="Signal Raw", color="blue")
boxcar_line, = ax.plot([], [], label="Box Car", color="red")
signal_smooth_line, = ax.plot([], [], label="Smooth", color="green")
ax.set_xlim(0, len(signal_norm))  # Set x-axis limits
#ax.set_xlim(0, 100)  # Set x-axis limits
ax.set_ylim(-2,2)  # Adjust y-axis limits based on your signal range
ax.legend()
ax.set_title("Real-Time Signal Plot")
ax.set_xlabel("Time")
ax.set_ylabel("Signal Value")
'''
fig, axes = plt.subplots(nrows=4, ncols=1)

# First subplot: Signal Raw
signal_line, = axes[0].plot([], [], label="Signal Raw", color="blue")
axes[0].set_xlim(0, len(signal_norm))  # Set x-axis limits
axes[0].set_ylim(-2, 2)  # Adjust y-axis limits based on your signal range
axes[0].legend()
axes[0].set_title("Signal Raw")
axes[0].set_ylabel("Signal Value")

# Second subplot: Box Car
boxcar_line, = axes[1].plot([], [], label="Box Car", color="red")
axes[1].set_xlim(0, len(signal_norm))  # Set x-axis limits
axes[1].set_ylim(-2, 2)  # Adjust y-axis limits based on your signal range
axes[1].legend()
axes[1].set_title("Box Car")
axes[1].set_ylabel("Signal Value")

# Third subplot: Smooth
signal_smooth_line, = axes[2].plot([], [], label="Smooth", color="green")
axes[2].set_xlim(0, len(signal_norm))  # Set x-axis limits
axes[2].set_ylim(-2, 2)  # Adjust y-axis limits based on your signal range
axes[2].legend()
axes[2].set_title("Smooth")
axes[2].set_xlabel("Time")
axes[2].set_ylabel("Signal Value")

#h, = axes[3].hist([], [], label="Smooth", color="green")
hist_data=[]
axes[3].hist(hist_data, bins=20, color="purple", label="Histogram")
axes[3].set_xlim(0,len(signal_norm))  # Set x-axis limits
axes[3].set_ylim(0, 50)  # Adjust y-axis limits based on your signal range
axes[3].legend()
axes[3].set_title("Distance")
axes[3].set_xlabel("Distance")
axes[3].set_ylabel("Count")


# Adjust layout for better spacing
fig.tight_layout()


def update(frame):
    signal_line.set_data(range(len(signal_raw)), signal_norm)
    signal_smooth_line.set_data(range(len(signal_norm_smooth)), signal_norm_smooth)
    boxcar_line.set_data(range(len(box_car)), box_car)

    axes[3].cla()  # Clear the histogram axis
    axes[3].hist(signal_norm, bins=10, color="purple", label="Histogram",density=True)  # Update with new data
    axes[3].set_xlim(-2,2)
    axes[3].set_ylim(0, 1)
    axes[3].legend()
    axes[3].set_title("Distance Histogram")
    axes[3].set_xlabel("Distance")
    axes[3].set_ylabel("Count")


    return signal_line, boxcar_line

# Create the animation
ani = FuncAnimation(fig, update, interval=1000)  # Update every 50ms

# Show the plot in a separate window
plt.ion()  # Enable interactive mode
plt.show()






with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  tic=time.time()
  toc=time.time()
  while cap.isOpened():
    success, image = cap.read()
    plt.pause(0.001)
    
    
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #hsv_image = cv2.cvtColor(copy.deepcopy(image), cv2.COLOR_RGB2HSV)
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
      
      # get the lanmarks
      landmarks=results.multi_face_landmarks[0]

      

    # Predict, update, resample, and estimate
      if USE_PARTICLE_FILTER:
        landmark_array = np.array([[lm.x * img_shape[1], lm.y * img_shape[0]] for lm in results.multi_face_landmarks[0].landmark])
        pf.predict()
        pf.update(landmark_array)
        pf.resample()
        smoothed_landmarks = pf.estimate()
        landmark_array = [(int(x), int(y)) for (x,y) in smoothed_landmarks]
      else:
        landmark_array = np.array([[int(lm.x * img_shape[1]), int(lm.y * img_shape[0])] for lm in results.multi_face_landmarks[0].landmark])

      hsv_image = copy.deepcopy(image) #cv2.cvtColor(copy.deepcopy(image), cv2.COLOR_RGB2HSV)
      
      # get the pixesl in the face region
      masked_hsv_image=roi.mask_out_face(hsv_image,landmark_array)

      i,j=np.where(masked_hsv_image[:,:,2]>0)

    

      # computing the median skin pixel value over 10 seconds in that region
      avg_spatial_pixel = np.mean(hsv_image[i,j,:], axis=0)
      med_temporal_pixel = np.median(trailing_mean_pixels, axis=0)
      trailing_mean_pixels.append(avg_spatial_pixel)
      trailing_mean_pixels.pop(0)


      # get the current 
      pixels=masked_hsv_image[i,j,:].astype(float)
      avg_pixel = np.mean(pixels, axis=0)
      #signal_h = masked_hsv_image[i,j,:].astype(float)  - np.mean(mean_h)
      
      distances = np.mean(pixels[:,1]-med_temporal_pixel[1])

      '''
      normalized_pixels = pixels / np.linalg.norm(pixels, axis=1, keepdims=True)
      normalized_temporal_med_pixel = med_temporal_pixel  / np.linalg.norm(med_temporal_pixel)

      # Step 3: Compute cosine distances
      distances = 1-  np.dot(normalized_pixels, normalized_temporal_med_pixel)
      '''
      
      
      #print(np.sum(distances))
      #print("mean sig raw",np.mean(signal_raw))
      #print("std sig raw",np.std(signal_raw))
      #hue_sig = (np.mean(distances) - np.mean(signal_raw))/np.std(signal_raw)
      signal_raw.append(np.sum(distances))
      fs=1/np.mean(total_times)

      

      signal_raw.pop(0)
      #bandpass_filter(signal, lowcut, highcut, fs, order=4)


      hue_sig = np.sum(distances)
      
      signal_norm.append((hue_sig)/(np.max(signal_raw))-np.min(signal_raw))
      
      
      
      signal_norm.pop(0)
      try:
        signal_new = signal.bandpass_filter(signal_norm, .5,4, fs)
      except:
        pass

      signal_norm_smooth.append(signal_norm[-1])
      signal_norm_smooth.pop(0)
      #print("sig_norm",signal_norm[-1])

      print(hue_sig)

      #hue_sig = np.mean(signal_h)
      

      if signal_raw[-1]>0:
        cv2.putText(masked_hsv_image, f'Beep!', (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        box_car.append(1)
        box_car.pop(0)
      else:
        box_car.append(0)
        box_car.pop(0)
      
      #hue_sigmoid = 1/(1 + np.exp(hue_sig))
      
      signal_raw.append(hue_sig)
      signal_raw.pop(0)
      
      toc=time.time()
      dt=toc-tic
      total_times.append(dt)
      total_times.pop(0)
      tic=toc
      freq = 1/dt
      #fft_result = np.fft.fft(signal_raw)
      #frequencies = np.abs(np.fft.fftfreq(len(signal_raw), d=dt))
      #freq_idx=np.argmax(fft_result)
      #max_freq=frequencies[freq_idx]*60
      # acceptable range
      box_car_count=signal.count_boxcars(box_car)
      bpm=60*box_car_count/np.sum(total_times)
      # find proper ROIs
      # find proper color space transformations
      # make sure fft analysis is correct
      # makr sure 


      
      for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            image=masked_hsv_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
        '''
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_contours_style())
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_iris_connections_style())
    # Flip the image horizontally for a selfie-view display.
        '''


      masked_hsv_image=cv2.flip(masked_hsv_image, 1)

      #if mean_h[0]==0:

      #  cv2.putText(masked_hsv_image, f'Calibrating:', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
      #  cv2.imshow('MediaPipe Face Mesh masked',masked_hsv_image)
      #  continue  
      cv2.putText(masked_hsv_image, f'Hue Sum: {hue_sig}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
      cv2.putText(masked_hsv_image, f'frec: {freq}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
      cv2.putText(masked_hsv_image, f'hear_beat: {bpm}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
      if hue_sig<0:
        cv2.putText(masked_hsv_image, f'Beep!!!!!!!!', (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
      #masked_hsv_image=color_methods.plot_signal_on_image(masked_hsv_image, signal_raw, color=(255,0, 0), thickness=2)
      #masked_hsv_image=color_methods.plot_signal_on_image(masked_hsv_image, box_car, color=(255,0, 0), thickness=2)
      cv2.imshow('MediaPipe Face Mesh masked',masked_hsv_image)
    else:
      
      cv2.imshow('MediaPipe Face Mesh masked', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()

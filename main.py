import cv2
import mediapipe as mp
import ROI_selection_methods as roi
import copy
import numpy as np
import color_analysis as color_methods
import signal_processing_methods as signal
import time

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

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
time_window=60
signal_raw = [0]*time_window
mean_h = [0]*time_window
total_times=[0]*time_window
box_car=[0]*time_window
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  tic=time.time()
  toc=time.time()
  while cap.isOpened():
    success, image = cap.read()
    
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
      landmarks=results.multi_face_landmarks[0]
      hsv_image = cv2.cvtColor(copy.deepcopy(image), cv2.COLOR_RGB2HSV)
      
      masked_hsv_image=roi.mask_out_face(hsv_image,landmarks)
      hue_channel = masked_hsv_image[:, :, 0]
      i,j=np.where(masked_hsv_image[:,:,2]>0)
      mean_h.append(np.mean(hue_channel[i,j]))
      mean_h.pop(0)

      mean_mean_h = float(np.mean(mean_h))
      signal_h = hue_channel[i,j].astype(float)  - np.mean(mean_h)
      
      hue_sig = np.mean(signal_h)
      if hue_sig<0:
        cv2.putText(masked_hsv_image, f'Beep!', (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        box_car.append(1)
        box_car.pop(0)
      else:
        box_car.append(0)
        box_car.pop(0)
      
      hue_sigmoid = 1/(1 + np.exp(hue_sig))
      
      signal_raw.append(hue_sigmoid)
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
      cv2.putText(masked_hsv_image, f'Hue Sum: {hue_sig}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
      cv2.putText(masked_hsv_image, f'frec: {freq}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
      cv2.putText(masked_hsv_image, f'hear_beat: {bpm}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
      if hue_sig<0:
        cv2.putText(masked_hsv_image, f'Beep!', (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
      masked_hsv_image=color_methods.plot_signal_on_image(masked_hsv_image, signal_raw, color=(255,0, 0), thickness=2)
      #masked_hsv_image=color_methods.plot_signal_on_image(masked_hsv_image, box_car, color=(255,0, 0), thickness=2)
      cv2.imshow('MediaPipe Face Mesh masked',masked_hsv_image)
    else:
      
      cv2.imshow('MediaPipe Face Mesh masked', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()

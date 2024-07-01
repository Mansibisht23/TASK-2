'''Create a model to predict the car color in the traffic as well as the count of car in the traffic signal. 
  This model should mark red color car as blue and blue color car as red . if the traffic signal has people 
  we should predict the number of males and females in the traffic signal . 
  if the traffic signal has other vehicles apart from car we should predict how many other vehicles are there.'''

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk

def detect_cars(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range of red color in HSV
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # Define range of blue color in HSV
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # Threshold the HSV image to get only red and blue colors
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Combine the masks
    mask_cars = cv2.bitwise_or(mask_red, mask_blue)

    # Find contours
    contours, _ = cv2.findContours(mask_cars, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area to remove noise
    min_area = 100
    max_area = 10000
    cars = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(contour)
            car_color = hsv[y:y+h, x:x+w]

            # Swap red and blue colors
            red_mask = cv2.inRange(car_color, lower_red1, upper_red1) | cv2.inRange(car_color, lower_red2, upper_red2)
            blue_mask = cv2.inRange(car_color, lower_blue, upper_blue)

            image[y:y+h, x:x+w][red_mask > 0] = [255, 0, 0]  # Change red to blue
            image[y:y+h, x:x+w][blue_mask > 0] = [0, 0, 255]  # Change blue to red

            cars.append((x, y, w, h))

    return cars, image

def detect_people(image):
    # Use pre-trained Haar Cascade classifier for people detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    people = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return people

def detect_other_vehicles(image):
    # Dummy function to simulate other vehicle detection
    # This should be replaced with actual vehicle detection logic
    return 5  # Example: 5 other vehicles

def process_image(image_path):
    image = cv2.imread(image_path)
    
    # Detect and swap car colors
    cars, image = detect_cars(image)
    car_count = len(cars)
    
    # Detect people
    people = detect_people(image)
    people_count = len(people)
    
    # Detect other vehicles
    other_vehicles_count = detect_other_vehicles(image)
    
    return car_count, people_count, other_vehicles_count

def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        car_count, people_count, other_vehicles_count = process_image(file_path)
        result_label.config(text=f"Car count: {car_count}\nPeople count: {people_count}\nOther vehicles count: {other_vehicles_count}")

# Create a Tkinter window
window = tk.Tk()
window.title("Traffic Analysis")

# Set the size and center the window
window.geometry('600x400')
window.update_idletasks()
width = window.winfo_width()
height = window.winfo_height()
x = (window.winfo_screenwidth() // 2) - (width // 2)
y = (window.winfo_screenheight() // 2) - (height // 2)
window.geometry('{}x{}+{}+{}'.format(width, height, x, y))

# Create a style object
style = ttk.Style()

# Customize the style of the button
style.configure('TButton',
                font=('Helvetica', 16, 'bold'),  # Bold font
                foreground='white',  # Text color
                background='#4CAF50',  # Button color
                borderwidth=4,  # Border width
                relief='raised',  # Button style
                padding=10)  # Padding around text

style.map('TButton',
          background=[('active', '#45a049'), ('!active', '#4CAF50')],
          relief=[('pressed', 'sunken'), ('!pressed', 'raised')])  # Change color on hover

# Create an upload image button with the customized style
upload_button = ttk.Button(window, text="Upload Image", style='TButton', command=upload_image)
upload_button.pack(pady=20)

# Create a result label
result_label = tk.Label(window, text="", font=('Helvetica', 12))
result_label.pack(pady=20)

# Run the Tkinter event loop
window.mainloop()

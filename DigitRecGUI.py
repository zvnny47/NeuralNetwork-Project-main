import tkinter as tk
import tkinter.ttk as ttk
from PIL import Image, ImageDraw
import numpy as np
import cv2

class DigitRecGUI:
    def __init__(self, window, predict_callback,train_model_callback):
        #Create Window
        self.window = window
        self.window.title('Digit Recognition')
        self.window.attributes("-fullscreen", True)


        #Create Theme
        style = ttk.Style()
        style.theme_use('clam')

        #Var
        self.color = 'black'
        self.points = []
        self.pen_width = 10
        self.image = None

        #Create UI
        self.create_mainUI()
        self.predict_callback = predict_callback
        self.prediction_label = tk.Label(self.window, text="", font=('Helvetica', 15))
        self.prediction_label.pack()
        self.current_label = None

        self.train_button = tk.Button(window, text="Train Model", command=train_model_callback)
        self.train_button.pack()

        #Create custom data
        try:
            loaded_data = np.load('data.npz')
            self.trainImages = loaded_data['images'].tolist()
            self.trainLabels = loaded_data['labels'].tolist()
            print(self.trainLabels)
        except: 
            self.trainImages = []
            self.trainLabels = []

    def paint(self, event):
        x1, y1 = (event.x - self.pen_width), (event.y - self.pen_width)
        x2, y2 = (event.x + self.pen_width), (event.y + self.pen_width)
        self.canvas.create_oval(x1, y1, x2, y2, fill=self.color)
        self.points.append((event.x, event.y))

    def create_mainUI(self):
        canvas_box = tk.Frame(self.window, background='#353535')
        canvas_box.pack(fill='both')
        self.canvas = tk.Canvas(canvas_box, width=375, height=375, background='#505050', highlightthickness=0)
        self.canvas.pack(anchor='nw', padx=25, pady=25)

        # self.canvas.create_line(372, 0, 372, 375, width=5, fill='#202020')  # Right line
        # self.canvas.create_line(0, 372, 375, 372, width=5, fill='#202020')  # Bottom line
        # self.canvas.create_line(2, 0, 2, 375, width=5, fill='#707070')  # Left line
        # self.canvas.create_line(0, 2, 375, 2, width=5, fill='#707070')  # Top line
        self.canvas.bind('<B1-Motion>', self.paint)
        self.drawing = False
        self.lines = []

        digit_buttons = []
        bottom_frame = tk.Frame(self.window, background='#707070')
        bottom_frame.pack(expand=True, fill='both')
        button_frame = tk.Frame(bottom_frame, background='#707070')
        button_frame.pack(anchor='nw', padx=22, pady=22)

        for digit in range(10):
            button = tk.Button(button_frame, text=str(digit), width=2, height=1, command=lambda d=digit: self.button_click(d),
                               bg='#DC7561', fg='black', font=('Helvetica', 15))
            button.pack(side='left', padx=3, pady=3)
            digit_buttons.append(button)

        self.digit_buttons = digit_buttons

        predict_button = tk.Button(button_frame, text="Predict", command=self.predict_wrapper, bg='#DC7561', fg='black', font=('Helvetica', 15))
        predict_button.pack(side='left', padx=10)

    def predict_wrapper(self):
        self.predict()
        self.predict_callback(self.image, self.current_label)  
    def predict(self):
        image = Image.new("L", (375, 375), 'white')
        draw = ImageDraw.Draw(image)
        for point in self.points:
            draw.ellipse([point[0] - self.pen_width, point[1] - self.pen_width, point[0] + self.pen_width, point[1] + self.pen_width], fill='black')


        # Resize to 28x28
        image = image.resize((28, 28))
        
        # Convert to NumPy array
        image = np.invert(np.array(image))
        
        # Thresholding and reshaping
        img = image
        _, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
        img = img.reshape(28, 28)

        # Save the image
        cv2.imwrite('TestImage.png', img)

        # Normalize and reshape for prediction
        _,image = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)
        image = image / 255.0
        image = image.reshape(-1)

        try:
            loaded_data = np.load('data.npz')
            self.trainImages = loaded_data['images'].tolist()
            self.trainLabels = loaded_data['labels'].tolist()
            print(self.trainLabels)
        except:
            print("Failed")
        #Debug
        for img in self.trainImages:
            print(np.array(img).shape)
        if self.current_label is not None:
            print(image.shape)
            self.trainImages.append(image)
            self.trainLabels.append(int(self.current_label))
            np.savez('data.npz',images = self.trainImages,labels = self.trainLabels)

        self.canvas.delete('all')
        self.points = []
        self.image = image

    
    def button_click(self, digit):
        for btn in self.digit_buttons:
            if btn != self.digit_buttons[digit]:
                btn['state'] = tk.NORMAL
                btn['fg'] = 'black'

        if self.digit_buttons[digit]['fg'] == 'black':
            self.digit_buttons[digit]['fg'] = '#39FF14'
            self.current_label = digit
        elif self.digit_buttons[digit]['fg'] == '#39FF14':
            self.digit_buttons[digit]['fg'] = 'black'
            self.current_label = None
            
    def display_prediction(self, prediction):
        self.prediction_label.config(text=f"Predicted digit: {prediction}")
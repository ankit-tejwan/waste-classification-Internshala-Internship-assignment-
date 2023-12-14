import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

class ImageClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Classifier")

        self.model_path = "Adam.h5"
        self.model = tf.keras.models.load_model(self.model_path)
        self.classes = ['glass','metal','paper', 'plastic']

        self.create_widgets()

    def create_widgets(self):
        # Image display
        self.image_label = tk.Label(self.root)
        self.image_label.pack(pady=10)

        # Open File button
        self.open_button = tk.Button(self.root, text="Open Image", command=self.open_image)
        self.open_button.pack(pady=10)

        # Predict button
        self.predict_button = tk.Button(self.root, text="Predict", command=self.predict_image)
        self.predict_button.pack(pady=10)

        # Prediction result
        self.result_label = tk.Label(self.root, text="")
        self.result_label.pack(pady=10)

    def open_image(self):
        file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif")])

        if file_path:
            # Display the selected image
            img = Image.open(file_path)
            img = img.resize((450, 300))  # Resize for display
            img = ImageTk.PhotoImage(img)
            self.image_label.config(image=img)
            self.image_label.image = img

            # Save the file path for prediction
            self.image_path = file_path

    def preprocess_image(self):
        img = Image.open(self.image_path)
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict_image(self):
        if hasattr(self, 'image_path'):
            input_data = self.preprocess_image()
            predictions = self.model.predict(input_data)
            predicted_class = self.classes[np.argmax(predictions)]
            confidence = float(predictions[0, np.argmax(predictions)])
            result_text = f"Predicted class: {predicted_class}\nConfidence: {confidence:.2%}"
            self.result_label.config(text=result_text)
        else:
            self.result_label.config(text="Please open an image first")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageClassifierApp(root)
    root.mainloop()

import tkinter as tk
from tkinter import messagebox
import torch
import torch.nn as nn

input_labels = ['Edge Roughness:',
                'Middle Roughness:',
                'Edge Depth:',
                'Middle Depth:',
                'Inner Depth:',
                'Material Friction:',
                'Estimated Duration:']

output_labels = ['Pridicted Duration:',
                'Percentage Upstairs:',
                'Usage Frequency:']

scalers = [100, 0.1, 0.001]

# Example model class (replace this with your actual model class)
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(7, 16)  # Input: 7 -> Hidden: 16
        self.fc2 = nn.Linear(16, 32)  # Hidden: 16 -> Hidden: 32
        self.fc3 = nn.Linear(32, 3)  # Hidden: 32 -> Output: 3
        self.activation = nn.ReLU()  # ReLU activation function

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)  # No activation for the output layer
        return x

# Load the model architecture and the state dictionary
model = MyModel()  # Create an instance of your model
model.load_state_dict(torch.load('best_model.pth'))  # Load the trained weights
model.eval()  # Set the model to evaluation mode

def predict():
    try:
        # Get input values from the entry fields
        inputs = [float(entry.get()) for entry in entries]
        
        # Convert to tensor and make prediction
        input_tensor = torch.tensor(inputs).float()
        with torch.no_grad():
            output = model(input_tensor)
        
        # Display the output in the result labels
        for i, result_label in enumerate(result_labels):
            result_label.config(text=f"{output_labels[i]} {output[i].item() * scalers[i]:.5f}")
    
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numbers for all inputs.")

# Create the main window
root = tk.Tk()
root.title("Model Prediction Interface")

# Create input labels and fields for 7 inputs
inputs_frame = tk.Frame(root)
inputs_frame.pack(pady=10)

entries = []
for i in range(7):
    label = tk.Label(inputs_frame, text=input_labels[i])
    label.grid(row=i, column=0, padx=10, pady=5)
    
    entry = tk.Entry(inputs_frame)
    entry.grid(row=i, column=1, padx=10, pady=5)
    entries.append(entry)

# Create a button to trigger the prediction
predict_button = tk.Button(root, text="Predict", command=predict)
predict_button.pack(pady=20)

# Create labels to display the 3 output values
result_labels_frame = tk.Frame(root)
result_labels_frame.pack(pady=10)

result_labels = []
for i in range(3):
    result_label = tk.Label(result_labels_frame, text=output_labels[i])
    result_label.grid(row=i, column=0, padx=10, pady=5)
    result_labels.append(result_label)

# Run the application
root.mainloop()

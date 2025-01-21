import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
from train import SimpleCNN, get_class_names
from constants import LIST_FILE

class SimpleCNNWithSoftmax(SimpleCNN):
    def __init__(self, num_classes=37):
        super(SimpleCNNWithSoftmax, self).__init__(num_classes)
   
    def forward(self, x):
        logits = super(SimpleCNNWithSoftmax, self).forward(x)
        probs = F.softmax(logits, dim=1)
        return probs

@st.cache_resource
def load_model(model_path, num_classes):
    """Load the model and set it to evaluation mode."""
    model = SimpleCNNWithSoftmax(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def preprocess_image(image):
    """Preprocess the image for model input."""
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
                             std=[0.229, 0.224, 0.225])   # ImageNet std
    ])
    return preprocess(image).unsqueeze(0)  # Add batch dimension

def main():
    st.title("Pet Classification Demo")
    st.write("Upload an image to predict the pet.")

    # Path to the model file
    model_path = "best_pet_classifier.pth"  # Fixed path

    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found. Please provide the correct path.")
        return

    # Get class names
    class_names = get_class_names(LIST_FILE)  # Adjust path to list.txt if needed
    num_classes = len(class_names)

    # Load the model
    model = load_model(model_path, num_classes)

    # Upload image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            with st.spinner('Predicting...'):
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption='Uploaded Image', use_container_width=True)
                
                # Preprocess the image
                input_tensor = preprocess_image(image)

                # Make prediction
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = outputs.squeeze().cpu().numpy()
                    predicted_idx = probabilities.argmax()
                    predicted_class = class_names[predicted_idx]
                    predicted_prob = probabilities[predicted_idx]
                
                # Display prediction results
                st.write(f"**Predicted Pet**: {predicted_class}")
                st.write(f"**Confidence**: {predicted_prob*100:.2f}%")

                # Display top-5 probabilities as a bar chart
                top5_idx = probabilities.argsort()[-5:][::-1]
                top5_probs = probabilities[top5_idx]
                top5_classes = [class_names[idx] for idx in top5_idx]

                st.bar_chart({
                    "Probabilities": top5_probs
                }, width=700, height=400)

        except Exception as e:
            st.error(f"An error occurred while processing the image: {e}")

if __name__ == "__main__":
    main()

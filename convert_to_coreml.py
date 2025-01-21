import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import coremltools as ct

from train import SimpleCNN, get_class_names
from constants import LIST_FILE, MODEL_PATH

# Extend SimpleCNN to include softmax output
class SimpleCNNWithSoftmax(SimpleCNN):
    def __init__(self, num_classes=37):
        super(SimpleCNNWithSoftmax, self).__init__(num_classes)
    
    def forward(self, x):
        logits = super(SimpleCNNWithSoftmax, self).forward(x)
        probs = F.softmax(logits, dim=1)
        return probs

def validate_conversion(pytorch_model, coreml_model, class_names, test_image_path):
    """Compare inference results between PyTorch and Core ML models using the same image."""
    import torchvision.transforms as transforms
    
    # Preprocessing for PyTorch and Core ML
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform_for_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    # Load and preprocess image
    image = Image.open(test_image_path).convert('RGB')
    input_tensor = transform_for_test(image)   # shape: (3, 224, 224)
    input_tensor = input_tensor.unsqueeze(0)  # shape: (1, 3, 224, 224)
    
    # PyTorch inference
    with torch.no_grad():
        outputs_pt = pytorch_model(input_tensor)  # shape: (1, num_classes)
    probs_pt = outputs_pt.squeeze(0).numpy()  # shape: (num_classes,)
    pt_pred_idx = np.argmax(probs_pt)
    pt_pred_class = class_names[pt_pred_idx]
    pt_pred_conf = probs_pt[pt_pred_idx]
    
    # Display part of PyTorch input tensor
    pt_input_flat = input_tensor.cpu().numpy().ravel()
    print("PyTorch input_tensor[0..4] =", pt_input_flat[:5], "...")
    print("PyTorch input_tensor shape =", input_tensor.shape, "dtype =", input_tensor.dtype)
    
    # Core ML inference
    coreml_input = input_tensor.cpu().numpy().astype(np.float32)  # shape: (1, 3, 224, 224)
    coreml_out = coreml_model.predict({"input_tensor": coreml_input})
    
    # Parse Core ML outputs
    if "classLabel" in coreml_out and "classLabel_probs" in coreml_out:
        cm_pred_class = coreml_out["classLabel"]
        cm_probs_dict = coreml_out["classLabel_probs"]
        cm_pred_conf = cm_probs_dict[cm_pred_class]
    else:
        found_key = [k for k in coreml_out.keys() if k.startswith("var_")]
        if len(found_key) == 1:
            logits_or_probs = coreml_out[found_key[0]]
            probs_cm = logits_or_probs.squeeze(0)
            cm_pred_idx = np.argmax(probs_cm)
            cm_pred_class = class_names[cm_pred_idx]
            cm_pred_conf = probs_cm[cm_pred_idx]
        else:
            raise ValueError(f"Unknown output keys in coreml_out: {list(coreml_out.keys())}")
    
    # Display results
    print("\n===== Validation =====")
    print(f"Test image: {test_image_path}")
    print(f"[PyTorch]  Predicted class: {pt_pred_class}  (Prob = {pt_pred_conf*100:.2f}%)")
    print(f"[Core ML]  Predicted class: {cm_pred_class}  (Prob = {cm_pred_conf*100:.2f}%)")

def main():
    # Load class names
    class_names = get_class_names(LIST_FILE)
    num_classes = len(class_names)
    print("Number of classes:", num_classes)
    
    # Load PyTorch model
    model_pt = SimpleCNNWithSoftmax(num_classes)
    model_path = MODEL_PATH
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
    state_dict = torch.load(model_path, map_location='cpu')
    model_pt.load_state_dict(state_dict)
    model_pt.eval()
    
    # Convert to TorchScript
    dummy_input = torch.randn(1, 3, 224, 224)
    traced_model = torch.jit.trace(model_pt, dummy_input)
    
    # Convert to Core ML
    classifier_config = ct.ClassifierConfig(
        class_labels=class_names,
        predicted_feature_name="classLabel"
    )
    
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(
                name="input_tensor",
                shape=(1, 3, 224, 224),
                dtype=np.float32
            )
        ],
        classifier_config=classifier_config,
        convert_to="mlprogram"
    )
    
    mlmodel.save("pet_classifier.mlpackage")
    print("Core ML model saved to 'pet_classifier.mlpackage'")
    
    # Validate if a test image path is provided
    if len(sys.argv) > 1:
        test_image_path = sys.argv[1]
        if os.path.isfile(test_image_path):
            validate_conversion(model_pt, mlmodel, class_names, test_image_path)
        else:
            print(f"File not found: {test_image_path}")
    else:
        print("No test image provided for validation.")

if __name__ == "__main__":
    main()

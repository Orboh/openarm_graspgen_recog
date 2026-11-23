from transformers import AutoTokenizer, AutoModel, AutoProcessor, pipeline
import torch 
import numpy as np

candidates=["plastic", "wood", "metal", "glass", "ceramic", "paper", "fabric", "rubber", "stone", "cardboard", "foam", "carbon fiber", "composite material"]
model_id="google/siglip2-so400m-patch14-384"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id, device_map="auto", torch_dtype="auto")

def classify_material(crop_bgr: np.ndarray) -> tuple[str, float]:
    """
    Classify the material of an object given its cropped image in BGR format.

    Args:
        crop_bgr (np.ndarray): The cropped image of the object in BGR format.

    Returns:
        str: The predicted material label.
        float: The confidence score of the prediction.
    """
    # Convert BGR to RGB
    crop_rgb = crop_bgr[..., ::-1].copy()

    # Preprocess the image
    inputs = processor(images=crop_rgb, text=candidates, return_tensors="pt", padding="max_length").to(model.device)

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the predicted class index
    probs=torch.sigmoid(outputs.logits_per_image)
    pred_idx=probs.argmax(-1).item()

    # Map the index to the corresponding material label
    predicted_material = candidates[pred_idx]

    # Calculate the confidence score
    confidence = probs.max().item()

    return predicted_material, confidence

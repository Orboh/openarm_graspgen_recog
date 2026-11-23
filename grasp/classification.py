from transformers import AutoTokenizer, AutoModel, AutoProcessor, pipeline
import torch 
import numpy as np

candidates = ["metal", "wood", "composite"]

# grasp/tools/plan_place.py --by-material: where to carry each material, as a --to dx,dy offset [m] from the
# object's own current spot (base frame; right is negative y, per plan_place.py's own convention). Anything
# classify_material returns that isn't a key here (shouldn't happen with the 3-class list above, but a model
# swap or a mis-classification could) falls back to "composite"'s offset.
ROUTE_OFFSET = {
    "metal": (0.0, -0.25),       # right, out of the chest camera's frame
    "wood": (0.0, 0.25),         # left, out of the chest camera's frame
    "composite": (0.20, 0.0),    # further from the robot base -- assumed to exit the top of the frame; unverified
                                  # against the real camera mount, check before trusting it on hardware
}
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

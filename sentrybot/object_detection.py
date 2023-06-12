import torch
import numpy as np
from distinctipy.distinctipy import get_colors, get_text_color
from matplotlib.image import AxesImage
from matplotlib.colors import rgb2hex
import pandas as pd
from PIL import ImageDraw, ImageFont
from IPython.display import display

# Import the transformers model
from transformers import YolosImageProcessor, YolosForObjectDetection


# Function nicked from scivision, with modifications
def display_objects(image: np.ndarray,
                    predictions: list,
                    label_nums: bool = False) -> AxesImage:
    """Display an image with colored bounding boxes for objects identified by a
    huggingface object detection model.

    Parameters
    ----------
    image : np.ndarray
        A numpy ndarray representation of an image.
    predictions : list
        A list of dictionaries, on for each detected object containing 'score',
        'label' and 'box' keys.
        'score' should be a float between 0 and 1.
        'label' should be a string that labels the object.
        'box' should be a dict with 'xmin', 'ymin', 'xmax' and 'ymax' keys,
        which denote the boundaries of each box (detected object).
    label_nums : bool = False
        When True, bounding boxes are numbered in addition to being colored.

    Returns
    -------
    The input image with colored bounding boxes and an accompanying pandas
    dataframe legend showing corresponding labels and scores for each object.
    """
    # Convert image to RGB
    pillow_image = image.convert('RGB')

    # Load font needed for numbering of bounding boxes and set size
    # font_path = os.path.abspath(os.path.dirname(__file__)) + '/fonts/arial.ttf'
    font_path = 'fonts/arial.ttf'
    font = ImageFont.truetype(font_path, 15)

    # Generate visually distinct colors for each object bounding box
    # We want a list of hexadecimal colors to use, and distinctipy can also
    # give us a matching list of the most appropriate text colors
    # for these colors (black or white)
    rgb_colors = get_colors(len(predictions), colorblind_type="Deuteranomaly")
    hex_colors = []
    text_hex_colors = []
    for color in rgb_colors:
        hex_colors.append(rgb2hex(color))
        text_hex_colors.append(rgb2hex(get_text_color(color)))

    # Place the object bounding boxes onto the image and use IPython to display
    index = 0
    for bounding_box in predictions:
        box = bounding_box["box"]
        im_with_rectangle = ImageDraw.Draw(pillow_image)
        im_with_rectangle.rounded_rectangle((box["xmin"], box["ymin"],
                                             box["xmax"], box["ymax"]),
                                            outline=hex_colors[index],
                                            width=2, radius=10)
        if label_nums:
            im_with_rectangle.text((box["xmin"] + 3, box["ymin"] + 1),
                                   str(index), fill=hex_colors[index],
                                   font=font)
        index += 1
    display(pillow_image)

    # Return a colored pandas df table relating to the objects shown in image
    object_predictions = pd.DataFrame(predictions).drop('box', 1)
    object_predictions['bbox'] = object_predictions.index

    def get_col(s):
        """Func to color pandas df"""
        col = 'background-color: ' + hex_colors[s.bbox]
        col += '; color: ' + text_hex_colors[s.bbox]
        return [col] * 3

    return object_predictions.style.apply(get_col, axis=1)



def object_predict(image):
    """Run the huggingface model prediction on an image, then use display_objects().
    """

    model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
    image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")

    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # model predicts bounding boxes and corresponding COCO classes
    logits = outputs.logits
    bboxes = outputs.pred_boxes

    # Generate predictions
    predictions = []
    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        prediction = {}
        prediction['score'] = score.item()
        prediction['label'] = model.config.id2label[label.item()]
        prediction['box'] = {}
        prediction['box']["xmin"] = box[0]
        prediction['box']["ymin"] = box[1]
        prediction['box']["xmax"] = box[2]
        prediction['box']["ymax"] = box[3]
        predictions.append(prediction)
    return display_objects(image, predictions)
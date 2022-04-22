import numpy as np
import cv2
from uuid import uuid4

from face_blur_bot import MODEL_PATH, PROTOTXT_PATH


def face_blur(image_name: str,
              prototxt: str = PROTOTXT_PATH,
              model: str = MODEL_PATH,
              confidence2: float = 0.5,
              blur_type: str = "pixelate") -> str:
    """A method to blur faces in an image using given model.

- Args:
    - `image_name`: `str` -> the name of the image to be processed.
    - `prototxt`: `str` -> the path to the prototxt file.
    - `model`: `str` -> the path to the model file.
    - `confidence2`: `float` -> the confidence threshold for the model.
    - `blur_type`: `str` -> the type of blur to be applied.

- Return:
    - `str` -> path of the processed image.
    """

    # load our serialized model from disk
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    # load the input image and construct an input blob for the image
    image = cv2.imread(image_name)
    (height, width) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > confidence2:
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array(
                [width, height, width, height])
            start_x, start_y, end_x, end_y = box.astype("int")

            face = image[start_y:end_y, start_x:end_x]
            if blur_type == "simple":
                face = anonymize_face_simple(face)
            else:
                face = anonymize_face_pixelate(face)
            image[start_y:end_y, start_x:end_x] = face

    # Return the final image
    img_name = f"{str(uuid4())}.jpg"
    cv2.imwrite(img_name, image)
    return img_name


def anonymize_face_simple(image: np.ndarray,
                          factor: float = 1.0) -> np.ndarray:
    """Blur the given image by the given factor.

- Args:
    - `image`: `np.ndarray` -> the image to be blurred.
    - `factor`: `float` -> the factor by which to blur the image.

- Return:
    - `np.ndarray` -> the blurred image.
"""
    # automatically determine the size of the blurring kernel based
    # on the spatial dimensions of the input image
    (height, width) = image.shape[:2]
    kernel_width = int(width / factor)
    kernel_height = int(height / factor)
    # ensure the width of the kernel is odd
    if kernel_width % 2 == 0:
        kernel_width -= 1
    # ensure the height of the kernel is odd
    if kernel_height % 2 == 0:
        kernel_height -= 1
    # apply a Gaussian blur to the input image using our computed
    # kernel size
    return cv2.GaussianBlur(image, (kernel_width, kernel_height), 0)


def anonymize_face_pixelate(image: np.ndarray, blocks: int = 9) -> np.ndarray:
    """Pixelate the given image by the given factor.
- Args:
    - `image`: `np.ndarray` -> the image to be blurred.
    - `blocks`: `int` -> the number of blocks to be used.

- Return:
    - `np.ndarray` -> the blurred image.
"""
    # divide the input image into NxN blocks
    height, width = image.shape[:2]
    x_steps = np.linspace(0, width, blocks + 1, dtype="int")
    y_steps = np.linspace(0, height, blocks + 1, dtype="int")
    # loop over the blocks in both the x and y direction
    for i in range(1, len(y_steps)):
        for j in range(1, len(x_steps)):
            # compute the starting and ending (x, y)-coordinates
            # for the current block
            start_x = x_steps[j - 1]
            start_y = y_steps[i - 1]
            end_x = x_steps[j]
            end_y = y_steps[i]
            # extract the ROI using NumPy array slicing, compute the
            # mean of the ROI, and then draw a rectangle with the
            # mean RGB values over the ROI in the original image
            roi = image[start_y:end_y, start_x:end_x]
            (B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
            cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (B, G, R),
                          -1)
    # return the pixelated blurred image
    return image


if __name__ == '__main__':
    my_image = face_blur('input.jpg',
                         model=MODEL_PATH,
                         prototxt=PROTOTXT_PATH,
                         blur_type="simple")
    cv2.imwrite('output.jpg', my_image)

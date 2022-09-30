## fish-image-analysis

Automated pre-processing and analysis of fish images.

Detects magenta stained bones from alizarin red staining.

Rotate
Crop 
Measure (magenta, length, width)

<br>

---

<br>

## Getting started
* Install the required modules

```pip install -r requirements.txt```

* Place images to be analyzed in ./input/raw

* Run main.py

```python main.py```


<br>

---

<br>

## Results

`./input/crop`

Contains rotated and cropped images. Rotation attempts to align imaged vertically. Images are cropped to the same dimensions.

`./input/magenta`

Contains magenta threshold images. In alizarin red stained fish, the skeleton should be visible in these images.

`./input/magenta/magenta.xlsx`

Report showing the measured amount of 'magenta' in the thresholded images. Also reports the approximate length and width of each fish (in pixels).
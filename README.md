# Optical Character Recognition using Custom Machine-Readable Alphabets

## Overview

This project presents a novel approach to Optical Character Recognition (OCR) by designing a completely new set of machine-readable alphabets instead of recognizing traditional English characters directly.

The objective was to create symbols that are easy for computers to identify while remaining simple enough to be handwritten. Each symbol is represented using a combination of bounded regions and filled blobs, allowing efficient recognition through image processing techniques rather than computationally expensive deep learning models.

The system performs image preprocessing, character segmentation, feature extraction, and ASCII decoding to reconstruct text from handwritten symbols.

---

## Motivation

Traditional OCR systems often rely on neural networks and large datasets because standard alphabets were designed for human readability rather than machine recognition.

This project explores the reverse problem:

> Can we design a writing system that is inherently easy for machines to recognize?

To answer this question, we developed a custom alphabet system along with an OCR pipeline capable of decoding handwritten symbols into ASCII characters.

---

## Key Features

- Custom machine-readable alphabet design
- Representation of printable ASCII characters
- Image preprocessing and binarization
- Character segmentation using DBSCAN
- Feature extraction using Connected Component Labeling (CCL)
- Blob-based character encoding and decoding
- End-to-end OCR pipeline
- Experimental evaluation on multiple handwritten datasets

---

## Methodology

### 1. Custom Alphabet Design

Each symbol consists of:

- An outer boundary
- Multiple internal chambers
- Filled circles (blobs) placed within chambers

The presence or absence of blobs acts as a binary encoding mechanism.

#### Symbol Generation

For **n chambers**, the total number of possible symbols is:

\[
2^n
\]

Using this approach:

| Chambers | Unique Symbols |
|-----------|---------------|
| 6 | 126 |
| 9 | 1022 |

This allows direct mapping of symbols to ASCII characters.

---

### 2. Image Preprocessing

The input image is converted into a binary image using grayscale thresholding.

Steps:

1. Convert image to grayscale
2. Apply thresholding
3. Generate binary image
4. Remove unwanted noise

---

### 3. Character Segmentation

To separate individual symbols, DBSCAN clustering is applied on foreground pixels.

#### DBSCAN Parameters

| Parameter | Value |
|------------|---------|
| Epsilon | 7.5 |
| Min Samples | 1 |

Each cluster is treated as a candidate character.

---

### 4. Connected Component Analysis

For each segmented symbol:

- Connected Component Labeling (CCL) is performed.
- Internal regions are detected.
- Filled blobs are identified.
- Blob counts are extracted.

The resulting feature vector uniquely identifies the character.

---

### 5. ASCII Decoding

The extracted blob pattern is interpreted as a binary representation.

The binary value is converted into:

```
Blob Pattern → Binary Number → ASCII Value → Character
```

This enables reconstruction of complete text from handwritten symbols.

---

## Technology Stack

- Python
- OpenCV
- NumPy
- Scikit-Learn (DBSCAN)
- Connected Component Labeling
- Image Processing Techniques

---

## Experimental Results

The OCR system was tested on multiple handwritten datasets.

| Dataset | Letters Present | Letters Recovered | Correctly Recognized |
|----------|----------------|-------------------|----------------------|
| 1 | 30 | 30 | 30 |
| 2 | 32 | 32 | 32 |
| 3 | 32 | 32 | 32 |
| 4 | 32 | 32 | 31 |
| 5 | 5 | 5 | 5 |
| 6 | 40 | 40 | 39 |
| 7 | 1 | 1 | 1 |
| 8 | 9 | 9 | 9 |
| 9 | 20 | 20 | 20 |

### Error Analysis

#### Type-I Error

Failure to recover characters during clustering.

\[
\text{Type-I Error} =
\frac{\text{Letters Not Recovered}}
{\text{Total Letters}}
\times 100
\]

**Result:** 0%

#### Type-II Error

Incorrect recognition after successful recovery.

\[
\text{Type-II Error} =
\frac{\text{Incorrectly Recognized Letters}}
{\text{Recovered Letters}}
\times 100
\]

**Result:** ~0.995%

---

## Sample Pipeline

```text
Input Image
      │
      ▼
Image Binarization
      │
      ▼
DBSCAN Clustering
      │
      ▼
Character Segmentation
      │
      ▼
Connected Component Labeling
      │
      ▼
Blob Detection
      │
      ▼
Binary Encoding
      │
      ▼
ASCII Decoding
      │
      ▼
Recovered Text
```

---

## Challenges

- Sensitivity to non-uniform lighting conditions
- Recognition errors caused by broken boundaries
- Dependence on proper spacing between symbols
- Increased computational cost for large images
- Handwritten variations affecting blob placement

---

## Future Improvements

- Design more intuitive machine-readable symbols
- Improve robustness against image noise
- Reduce clustering time complexity
- Support real-time OCR from camera feeds
- Develop a hybrid machine learning-based recognition system
- Build a web application for interactive OCR demonstrations

---

## Learning Outcomes

Through this project, I gained hands-on experience in:

- Computer Vision
- Optical Character Recognition
- Image Processing
- Clustering Algorithms
- Connected Component Analysis
- Feature Engineering
- Pattern Recognition
- Algorithm Design

---

## Conclusion

This project demonstrates that carefully designed machine-readable symbols can significantly simplify the OCR process. By combining custom alphabet design, clustering, connected component analysis, and binary encoding, we successfully built an end-to-end OCR system capable of recognizing handwritten symbols with high accuracy.

The work highlights an alternative perspective on OCR: instead of making machines better at reading human-designed alphabets, we can also design alphabets that are inherently easier for machines to understand.

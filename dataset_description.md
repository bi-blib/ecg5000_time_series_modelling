# Dataset Description: ECG5000

## Dataset Title

ECG5000 Time-Series Classification Dataset

## Location

- **Dataset description and download:** https://www.timeseriesclassification.com/description.php?Dataset=ECG5000
- **Original ECG database:** https://physionet.org/content/chfdb/

The dataset is distributed through the UCR Time Series Classification Archive in formats such as text and ARFF files.

## Background & Motivation

ECG5000 is a benchmark dataset for classifying individual electrocardiogram heartbeats. It is commonly used to evaluate time-series classification methods for distinguishing normal cardiac activity from different types of abnormal heartbeats.

The original signal was obtained from the BIDMC Congestive Heart Failure Database on PhysioNet. Individual heartbeats were extracted from a long ECG recording and resampled to a common sequence length.

## Data Description

- The dataset contains 5,000 individual heartbeat signals.
- Every sample is a univariate time series.
- Each heartbeat contains 140 time steps.
- The predefined split contains:

  - 500 training samples
  - 4,500 test samples

- Each sample belongs to one of five heartbeat classes:

  1. Normal beat
  2. R-on-T premature ventricular contraction
  3. Premature ventricular contraction
  4. Supraventricular premature or ectopic beat
  5. Unclassified beat

- The dataset is strongly class-imbalanced, with normal and R-on-T beats occurring much more frequently than some of the other classes.

Typical data shapes are:

- Input: $$\(X \in \mathbb{R}^{N \times 140 \times 1}\)$$
- Labels: $$\(\mathbf{y} \in \left\lbrace 1,2,3,4,5 \right\rbrace^{N}\)$$

## Typical Modeling Task

- **Input:** one ECG heartbeat containing 140 measurements.
- **Target:** one of five heartbeat classes.
- **Task type:** univariate time-series classification.
- **Possible models:** 1D-CNN, LSTM, GRU, Transformer, ROCKET, or shapelet-based classifiers.

## Evaluation Metrics

Because the classes are imbalanced, accuracy alone may be insufficient. Suitable metrics include:

- Accuracy
- Macro-averaged precision
- Macro-averaged recall
- Macro-averaged F1-score
- Balanced accuracy
- Confusion matrix

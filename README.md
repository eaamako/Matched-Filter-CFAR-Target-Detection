# Matched-Filter-CFAR-Target-Detection
This Repository implements a matched filter Constant False Alarm Rate (CFAR) target detection algorithm for detecting targets in noisy radar signals.

The code uses a rectangular pulse as the reference signal, which is convolved with the noisy signal to calculate the matched filter output. The CFAR algorithm is then applied to the matched filter output to detect target peaks that exceed the threshold. 

The code also includes a function to calculate the CFAR threshold and window, which is used to set the detection threshold.

The CFAR algorithm is implemented by selecting a sliding window of samples from the matched filter output and comparing it to the noise level in the reference and guard cells. The code includes a main function that loops over different pulse parameters, calculates the matched filter output and CFAR threshold, and plots the detected target peaks. 



Finally, the code prints the total number of peaks detected and the probability of false alarms. The code also calculates the minimum number of guard cells required to avoid false alarms in cells adjacent to the CUT.

USAGE:
for detecting target peaks in a noisy received signal using a Matched Filter and Constant False Alarm Rate (CFAR) thresholding. The signal is assumed to be a pulse-compressed radar signal with a known pulse shape. The code sweeps over different pulse parameters (sweep bandwidth, pulse duration, and frequency shift), calculates the matched filter response, and applies the CFAR algorithm to detect the target peaks. 

The detected target peaks are plotted with the CFAR threshold and the signal power. The code also computes the probability of false alarms and determines the minimum number of guard cells needed to avoid false alarms in cells adjacent to the cell under test (CUT).

The code can be used in radar signal processing applications for target detection in noisy environments. It can be modified to handle different types of radar signals and pulse shapes, as well as to implement different detection algorithms and signal processing techniques.

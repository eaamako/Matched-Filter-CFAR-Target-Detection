

import numpy as np
import matplotlib.pyplot as plt

# Load the noisy received signal from the file
y = np.load('signal.npy')

# Define the pulse parameters
Fs = 10e6

global refLength, guardLength, P_fa
refLength = 500
guardLength = 80
P_fa = 0.00001

def rect(t):
    # rectangular pulse function
    x = np.zeros(len(t))
    x[abs(t) < 0.5] = 1
    return x

def calculate_MF(y, B, T, u, Fs):
    t = np.arange(-T/2, T/2,1/Fs)
    k = B/T  # Sweep rate in Hz/s
    x = rect(t/T)*np.exp(1j*u*k*t**2)
     # Construct the matched filter for the pulse
    h = np.conj(x[::-1])

        # Apply the matched filter to the noisy signal
    b = np.convolve(y, h, mode='same')
    
    return b
   # calculate the CFAR threshold and Window
    
def detector_CFAR_Threshold(b):
            
    # calculate the CFAR threshold and Window
    " Reference Cells = refLenght"
    " Guard Cells = guardLenght"


    CFAR_window = np.ones((refLength+guardLength)*2+1)
    CFAR_window[refLength+1:refLength+1+2*guardLength] = 0
    CFAR_window = CFAR_window/np.sum(CFAR_window)
    noiseLevel = np.convolve((np.abs(b))**2,CFAR_window, mode='same')

    # Adjust the noise level at the edges of the data 
    #  to fix the issue of "fall off the edge" in the CA-CFAR threshold calculation

    for i in range(refLength+guardLength):
        noiseLevel[i] = np.mean(noiseLevel[:2*(refLength+guardLength)-i])
        noiseLevel[-i-1] = np.mean(noiseLevel[-2*(refLength+guardLength)+i:])

   # Boost the threshold near the edges to account for the fact that fewer cells are being averaged
        alpha = 1  # Factor to boost the threshold by
        if i < guardLength:
            noiseLevel[i] *= alpha ** (guardLength-i)
            noiseLevel[-i-1] *= alpha ** (guardLength-i)

    # Compute the threshold scale factor based on the desired false alarm probability
    gamma = -np.log(P_fa)

    # Compute the CFAR threshold for each sample in the signal
    CFAR_Threshold = noiseLevel * gamma
       
    return CFAR_Threshold
   

def main():
            
    # Define the time axis
    N = len(y)  # Number of samples
    
    for B in [1e6, 2e6, 5e6, 8e6]:
        for T in [10e-6, 20e-6, 30e-6, 40e-6]:
            for u in [-1, 1]:
                k = B/T  # Sweep rate in Hz/s
                tau = 0    # Shift of pulse in seconds
                
                b = calculate_MF(y, B, T, u, Fs)
                
                CFAR_Threshold = detector_CFAR_Threshold(b)
              
                " code to detect number of target peaks and plot them"
                
                # Initialize count of detected peaks
                Detected_Target_Peaks = 0
                
                # Compute a list of indices that exceed the CFAR threshold
                H = np.arange(guardLength, len(b) - guardLength)
                
                # Find peak with the highest magnitude squared above the threshold
                pvalues = np.where(np.abs(b[H])**2 > CFAR_Threshold[H])[0]    
                pINDX= None
                
                if len(pvalues) > 0:
                    pINDX = H[pvalues[np.argmax(np.abs(b[H][pvalues])**2)]]
                
                 # Plot  CFAR Threshold Detector and  indicate target peak
                # initialize counter
                plot_count = 0
                
                if pINDX is not None:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(H, np.abs(b[H])**2,label='Signal')
                    ax.plot(H, np.abs(CFAR_Threshold[H]), 'r--', linewidth=2)
                    ax.scatter(pINDX, np.abs(b[pINDX])**2, marker='*', color='m')
                    ax.set_title(f'B={B/1e6} MHz, T={int(T/1e-6)} us, u={u}, TARGET Peak Detected')
                    ax.legend(['p', 'CFAR Threshold', 'Target peak'],fontsize=8)
                    ax.set_xlabel('n (Samples)')
                    ax.set_ylabel('Received Power (W)')
                    ax.set_xlim(0, 10000)
                    plt.grid()
                    plt.tight_layout()
                    plt.show()
                    plot_count += 1
                             
                
    print(f'Total number of peaks detected: {plot_count}')   
    print(f"Probability of false alarms is: {P_fa:.3%}")      # print probability of false alarm


     ## What are the minimum number of guard cells G needed to avoid false alarms in cells adjacent to the CUT?
                           
    T = 20e-6  # pulse duration
    # = np.ceil(B/T**3)  # minimum number of guard cells
    N = 581
    L = 500
    D = 1
    G = np.ceil((N + 2 * (L + 1) - (2 * T)) / (2 * (L + 1)))
    print(f"Minimum number of guard cells needed: {G}")
    
    ## What are the maximum number of reference cells R that can be used before potentially masking another pulse?  
    
    P = 1/Fs  # pulse repetition interval
    T = 1
    G1 = 80
    R = np.ceil((P + G1+ T + 1) / (2 * (G1+ T + 1)))
    print(f"Maximum number of reference cells: {R}")
    
    ## What is the minimum P_FA for which no missed detections occur?                 

    
if __name__ == '__main__':
    main()
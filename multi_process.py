## THIS IS A MODULE FOR MULTIPROCESSING UTILIZATION
import numpy as np

def maxFilter(spectrum, threshold):
    # find max intensity wavelength
    max_index = np.argmax(spectrum[:, 1])
    max_val = spectrum[max_index][0]
    max_intensity = spectrum[max_index][1]
    ret_val = max_val if max_intensity > threshold else 0
    return ret_val


def buildSpectra(step,    
                wavelen_arr,        
                images_arr,            
                threshold,                
                img_height,                
                img_width):
    
    wave_len_num = len(wavelen_arr)

    spectra_array = np.zeros((img_height, img_width, wave_len_num, 2))
    max_inten_spectra_array = np.zeros((img_height, img_width))
    
    for i in range(img_height):
        for j in range(img_width):
            spectrum = np.transpose(np.array([wavelen_arr, images_arr[:, i, j]]))
            spectra_array[i][j] = spectrum # a spectrum for each pixel
            max_inten_spectra_array[i][j] = maxFilter(spectrum, threshold)
            
    return ((step, max_inten_spectra_array), (step, spectra_array))   
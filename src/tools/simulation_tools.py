import numpy as np
from langchain.tools import tool


def calculate_diffraction_efficiency(
    wavelength_nm: float, 
    pixel_pitch_um: float, 
    fill_factor: float = 0.93,
    phase_depth: float = 2.0 * np.pi
) -> float:
    """
    Calculate the first-order diffraction efficiency for an LCoS-SLM.
    
    Based on scalar diffraction theory with fill factor and phase modulation depth.
    
    Parameters:
        wavelength_nm: Incident wavelength in nanometers (e.g., 532.0)
        pixel_pitch_um: SLM pixel pitch in micrometers (e.g., 3.74)
        fill_factor: Ratio of active area to total pixel area (default: 0.93)
        phase_depth: Maximum phase modulation in radians (default: 2π)
    
    Returns:
        Diffraction efficiency as a float between 0.0 and 1.0
    """
    if wavelength_nm <= 0 or pixel_pitch_um <= 0:
        raise ValueError("Wavelength and pixel pitch must be positive")
    
    if not 0 < fill_factor <= 1:
        raise ValueError("Fill factor must be in range (0, 1]")
    
    sinc_arg = np.pi * fill_factor
    sinc_value = np.sin(sinc_arg) / sinc_arg if sinc_arg != 0 else 1.0
    aperture_efficiency = sinc_value ** 2
    
    phase_efficiency = (np.sin(phase_depth / 2) / (phase_depth / 2)) ** 2
    
    wavelength_factor = min(1.0, (532.0 / wavelength_nm) ** 0.5)
    
    total_efficiency = aperture_efficiency * phase_efficiency * wavelength_factor
    
    return np.clip(total_efficiency, 0.0, 1.0)
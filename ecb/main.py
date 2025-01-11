from Crypto.Cipher import AES
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os
from datetime import datetime
import glob
import math

class ImageEncryptionAnalyzer:
    def __init__(self, key: bytes = b"aaaabbbbccccdddd", iv: bytes = b"1111222233334444"):
        self.key = key
        # CTR can be added here!
        self.mode = AES.MODE_CTR
        self.results_dir = self._create_results_directory()

    def _create_results_directory(self) -> str:
        """Create a timestamped directory for results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"results_{timestamp}"
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(os.path.join(results_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(results_dir, "plots"), exist_ok=True)
        return results_dir

    def save_plot(self, filename: str):
        """Save plot to results directory."""
        plt.savefig(os.path.join(self.results_dir, "plots", filename), 
                    dpi=300, bbox_inches='tight')
        plt.close()

    def init_cipher(self) -> AES:
        """Initialize AES cipher."""
        cipher = AES.new(self.key, self.mode)
        return cipher

    def process_file(self, filename: str) -> bytes:
        """Read file and return bytes."""
        with open(filename, "rb") as f:
            return f.read()

    def save_file(self, filename: str, data: bytes) -> None:
        """Save bytes to file."""
        filepath = os.path.join(self.results_dir, "images", os.path.basename(filename))
        with open(filepath, "wb") as f:
            f.write(data)

    def get_padding(self, block: bytes) -> int:
        """Calculate padding for AES block size."""
        return (len(block) % 16) * -1
    
    def encrypt_image(self, cipher: AES.MODE_ECB, input_path: str, output_path: str) -> None:
        """Encrypt image using AES-ECB."""
        # Read image as raw bytes
        with open(input_path, 'rb') as f:
            img_data = f.read()
        
        # Skip BMP header (54 bytes) if it's a BMP file
        if input_path.lower().endswith('.bmp'):
            header = img_data[:54]
            data = img_data[54:]
        else:
            header = b''
            data = img_data
        
        # Ensure data length is multiple of 16 (AES block size)
        padding_length = (16 - (len(data) % 16)) % 16
        padded_data = data + bytes([padding_length] * padding_length)
        
        # Encrypt the data
        encrypted_data = b''
        for i in range(0, len(padded_data), 16):
            block = padded_data[i:i+16]
            encrypted_block = cipher.encrypt(block)
            encrypted_data += encrypted_block
        
        # Write the encrypted image
        with open(output_path, 'wb') as f:
            f.write(header + encrypted_data[:len(data)])  # Remove padding before saving

    def create_modified_image(self, image_path: str, output_path: str) -> None:
        """Create a modified version of the image."""
        with open(image_path, 'rb') as f:
            data = bytearray(f.read())
        
        # If BMP, start after header
        start_pos = 54 if image_path.lower().endswith('.bmp') else 0
        
        # Modify every 16th byte (one byte per block)
        for i in range(start_pos, len(data), 16):
            if i < len(data):
                data[i] = (data[i] + 1) % 256
        
        with open(output_path, 'wb') as f:
            f.write(data)

    def calculate_npcr(self, image1_path: str, image2_path: str) -> float:
        """Calculate NPCR between two encrypted images."""
        with open(image1_path, 'rb') as f1, open(image2_path, 'rb') as f2:
            data1 = f1.read()
            data2 = f2.read()
        
        # Skip headers if BMP
        start_pos = 54 if image1_path.lower().endswith('.bmp') else 0
        data1 = data1[start_pos:]
        data2 = data2[start_pos:]
        
        # Count different bytes
        diff_count = sum(1 for a, b in zip(data1, data2) if a != b)
        total_bytes = len(data1)
        
        return (diff_count / total_bytes) * 100

    def calculate_uaci(self, image1_path: str, image2_path: str) -> float:
        """Calculate UACI between two encrypted images."""
        with open(image1_path, 'rb') as f1, open(image2_path, 'rb') as f2:
            data1 = f1.read()
            data2 = f2.read()
        
        # Skip headers if BMP
        start_pos = 54 if image1_path.lower().endswith('.bmp') else 0
        data1 = data1[start_pos:]
        data2 = data2[start_pos:]
        
        # Calculate absolute differences
        diff_sum = sum(abs(a - b) for a, b in zip(data1, data2))
        total_bytes = len(data1)
        
        return (diff_sum / (total_bytes * 255)) * 100

    def decrypt_image(self, cipher: AES.MODE_ECB, input_path: str, output_path: str) -> None:
        """Decrypt image using AES-ECB."""
        # Read the encrypted image
        img = np.array(Image.open(input_path))
        
        # Convert to bytes and decrypt
        decrypted = cipher.decrypt(img.tobytes())
        
        # Remove padding if present
        if len(decrypted) > 0:
            padding_length = decrypted[-1]
            if padding_length > 0:
                decrypted = decrypted[:-padding_length]
        
        # Convert back to numpy array
        decrypted_array = np.frombuffer(decrypted, dtype=np.uint8).reshape(img.shape)
        
        # Save decrypted image
        Image.fromarray(decrypted_array).save(output_path)
   
    
    def plot_correlation_analysis(self, image_path: str, title: str, samples: int = 5000) -> dict:
        """Plot and calculate pixel correlation in all directions."""
        img = np.array(Image.open(image_path))
        correlations = {}
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        plt.rcParams.update({'font.size': 12})
        directions = ['Horizontal', 'Vertical', 'Diagonal']
        
        for channel in range(3):
            channel_data = img[:, :, channel]
            height, width = channel_data.shape
            
            for direction_idx, direction in enumerate(directions):
                x_samples = []
                y_samples = []
                
                while len(x_samples) < samples:
                    i = np.random.randint(0, height - 1)
                    j = np.random.randint(0, width - 1)
                    
                    if direction == 'Horizontal':
                        if j < width - 1:
                            x_samples.append(channel_data[i, j])
                            y_samples.append(channel_data[i, j + 1])
                    elif direction == 'Vertical':
                        if i < height - 1:
                            x_samples.append(channel_data[i, j])
                            y_samples.append(channel_data[i + 1, j])
                    else:  # Diagonal
                        if i < height - 1 and j < width - 1:
                            x_samples.append(channel_data[i, j])
                            y_samples.append(channel_data[i + 1, j + 1])
                
                correlation = np.corrcoef(x_samples, y_samples)[0, 1]
                correlations[f"{['R', 'G', 'B'][channel]}_{direction}"] = correlation
                
                ax = axes[channel, direction_idx]
                ax.scatter(x_samples, y_samples, c=['red', 'green', 'blue'][channel], 
                          alpha=0.1, s=1)
                ax.set_xlabel("Pixel Value")
                ax.set_ylabel("Adjacent Pixel Value")
                ax.set_title(f"{['R', 'G', 'B'][channel]} {direction}\nr = {correlation:.4f}")
                
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        self.save_plot(f"correlation_{title.lower().replace(' ', '_')}.png")
        
        return correlations
    
    def plot_rgb_histograms(self, image_path: str, title: str) -> None:
        """Plot RGB channel histograms."""
        img = np.array(Image.open(image_path))
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        plt.rcParams.update({'font.size': 12})
        colors = ['red', 'green', 'blue']

        for idx, (ax, color) in enumerate(zip(axes, colors)):
            hist, bins = np.histogram(img[:, :, idx].ravel(), bins=256, range=(0, 256))
            ax.bar(bins[:-1], hist, color=color, alpha=0.7, width=1)
            ax.set_title(f"{color.capitalize()} Channel")
            ax.set_xlabel("Pixel Intensity")
            ax.set_ylabel("Frequency")

        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        self.save_plot(f"histogram_{title.lower().replace(' ', '_')}.png")
    
    def calculate_entropy(self, image_path: str) -> dict[str, float]:
        """Calculate entropy for each RGB channel."""
        img = np.array(Image.open(image_path))
        entropy = {}
        
        for channel, color in enumerate(['R', 'G', 'B']):
            channel_data = img[:, :, channel].flatten()
            hist = np.histogram(channel_data, bins=256, range=(0, 256))[0]
            prob = hist / np.sum(hist)
            entropy[color] = -np.sum(prob * np.log2(prob + 1e-10))
            
        return entropy
    
    def calculate_psnr(self,image1_path, image2_path):
        # Load the images
        img1 = np.array(Image.open(image1_path), dtype=np.float32)
        img2 = np.array(Image.open(image2_path), dtype=np.float32)
        
        # Check if dimensions match
        if img1.shape != img2.shape:
            raise ValueError("Images must have the same dimensions for PSNR calculation.")
        
        # Calculate Mean Squared Error (MSE)
        mse = np.mean((img1 - img2) ** 2)
        
        # Handle special case where MSE is zero (identical images)
        if mse == 0:
            return float('inf')  # Infinite PSNR (perfect match)
        
        # Calculate PSNR
        max_pixel_value = 255.0  # For 8-bit images
        psnr = 10 * math.log10((max_pixel_value ** 2) / mse)
        
        return psnr

def main():
    # Initialize the analyzer
    analyzer = ImageEncryptionAnalyzer()

    print("Starting image encryption analysis...")

    # Convert PNG to BMP if needed
    if not os.path.exists("baboon.bmp"):
        Image.open("baboon.png").save("baboon.bmp", format="BMP")

    # Process original image
    original_image_path = "baboon.bmp"
    encrypted_image_path = os.path.join(analyzer.results_dir, "images", "encrypted_baboon.bmp")
    modified_image_path = os.path.join(analyzer.results_dir, "images", "modified_baboon.bmp")
    encrypted_modified_image_path = os.path.join(analyzer.results_dir, "images", "encrypted_modified_baboon.bmp")
    
    # Create a fresh cipher for each encryption
    cipher1 = analyzer.init_cipher()
    cipher2 = analyzer.init_cipher()

    # Create modified image first
    print("Creating modified image...")
    analyzer.create_modified_image(original_image_path, modified_image_path)
    
    # Encrypt both images
    print("Encrypting original image...")
    analyzer.encrypt_image(cipher1, original_image_path, encrypted_image_path)
    
    print("Encrypting modified image...")
    analyzer.encrypt_image(cipher2, modified_image_path, encrypted_modified_image_path)
    
    #Analysis
    print("\nAnalyzing original image...")
    original_corr = analyzer.plot_correlation_analysis("baboon.bmp", 
                                                     "Original Image Correlation")
    analyzer.plot_rgb_histograms("baboon.bmp", "Original Image Histograms")

    print("\nAnalyzing encrypted image...")
    encrypted_corr = analyzer.plot_correlation_analysis(
        encrypted_image_path,
        "Encrypted Image Correlation"
    )
    analyzer.plot_rgb_histograms(
        encrypted_image_path,
        "Encrypted Image Histograms"
    )
    # Add merged histogram plots
    print("\nGenerating merged histograms...")
    analyzer.plot_rgb_histograms("baboon.bmp", "Original Image")
    analyzer.plot_rgb_histograms(
        encrypted_image_path,
        "Encrypted Image"
    )

    # Calculate NPCR and UACI
    npcr_value = analyzer.calculate_npcr(encrypted_image_path, encrypted_modified_image_path)
    uaci_value = analyzer.calculate_uaci(encrypted_image_path, encrypted_modified_image_path)
    

    # Calculate PSNR
    psnr = analyzer.calculate_psnr(original_image_path, encrypted_image_path)

    # Calculate entropy
    original_entropy = analyzer.calculate_entropy("baboon.bmp")
    encrypted_entropy = analyzer.calculate_entropy(
        encrypted_modified_image_path
    )
    
    # Save results to file
    with open(os.path.join(analyzer.results_dir, "analysis_results.txt"), "w") as f:
        f.write("Image Encryption Analysis Results\n")
        f.write("================================\n\n")
        
        f.write("1. Differential Analysis\n")
        f.write(f"NPCR: {npcr_value:.4f}% (Expected: >99.5%)\n")
        f.write(f"UACI: {uaci_value:.4f}% (Expected: ~33.4633%)\n\n")
        f.write(f"PSNR: {psnr:.4f} dB\n\n")        

        f.write("2. Original Image Correlation\n")
        for key, value in original_corr.items():
            f.write(f"{key}: {value:.4f}\n")
        f.write("\n")
        
        f.write("3. Encrypted Image Correlation\n")
        for key, value in encrypted_corr.items():
            f.write(f"{key}: {value:.4f}\n")
        f.write("\n")
        
        f.write("4. Original Image Entropy\n")
        for channel, value in original_entropy.items():
            f.write(f"{channel}: {value:.4f}\n")
        f.write("\n")
        
        f.write("5. Encrypted Image Entropy\n")
        for channel, value in encrypted_entropy.items():
            f.write(f"{channel}: {value:.4f}\n")
            
    print(f"\nAnalysis complete! Results saved in {analyzer.results_dir}/")

if __name__ == "__main__":
    main()
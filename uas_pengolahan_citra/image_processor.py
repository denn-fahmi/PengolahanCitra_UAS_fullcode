import cv2
import numpy as np

def rgb_to_grayscale(img):
    """Mengubah citra RGB ke Grayscale."""
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def rgb_to_binary(img, threshold_val):
    """Mengubah citra RGB/Grayscale ke Biner dengan threshold manual."""
    gray = rgb_to_grayscale(img)
    _, binary = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)
    return binary

def histogram_equalization(img):
    """Melakukan perataan histogram (Histogram Equalization)."""
    if len(img.shape) == 2:
        return cv2.equalizeHist(img)
    else:
        # Untuk citra warna, konversi ke YCrCb agar tidak merusak keseimbangan warna (chrominance)
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])  # Equalize komponen Y (luminance)
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

def contrast_stretching(img, low_pct=2, high_pct=98):
    """Melakukan Contrast Stretching berbasis persentil."""
    if len(img.shape) == 2:
        low, high = np.percentile(img, [low_pct, high_pct])
        stretched = np.clip(img, low, high)
        if high > low:
            stretched = ((stretched - low) / (high - low) * 255).astype(np.uint8)
        else:
            stretched = stretched.astype(np.uint8)
        return stretched
    else:
        # Lakukan stretching pada channel L di ruang warna LAB agar warna tetap natural
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        low, high = np.percentile(l_channel, [low_pct, high_pct])
        stretched_l = np.clip(l_channel, low, high)
        if high > low:
            stretched_l = ((stretched_l - low) / (high - low) * 255).astype(np.uint8)
        else:
            stretched_l = stretched_l.astype(np.uint8)
        lab[:, :, 0] = stretched_l
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def adjust_brightness(img, value):
    """Mengatur kecerahan citra (Brightness Adjustment)."""
    # Menggunakan HSV Value channel untuk citra warna, atau add biasa untuk grayscale
    if len(img.shape) == 3:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        # Menghindari overflow dengan cv2.add / cv2.subtract atau numpy clip
        v = cv2.add(v, value) if value >= 0 else cv2.subtract(v, abs(value))
        final_hsv = cv2.merge((h, s, v))
        return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    else:
        return cv2.add(img, value) if value >= 0 else cv2.subtract(img, abs(value))

def sharpen_image(img, strength=1.0):
    """Mempertajam citra (Sharpening) menggunakan metode Unsharp Masking."""
    # Blurring citra terlebih dahulu
    blurred = cv2.GaussianBlur(img, (5, 5), 1.0)
    # Formula: original + strength * (original - blurred)
    sharpened = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def mean_filter(img, kernel_size=3):
    """Mean / Average Filter untuk mereduksi noise."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.blur(img, (kernel_size, kernel_size))

def median_filter(img, kernel_size=3):
    """Median Filter untuk mereduksi noise (sangat baik untuk salt-and-pepper)."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    # Median filter hanya menerima kernel ganjil dalam format integer
    return cv2.medianBlur(img, kernel_size)

def gaussian_filter(img, kernel_size=3, sigma=1.0):
    """Gaussian Filter untuk blurring yang halus."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)

def sobel_edge(img, kernel_size=3):
    """Deteksi tepi Sobel."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    gray = rgb_to_grayscale(img)
    # Hitung gradien x dan y
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)
    # Gabungkan magnitudonya
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
    return magnitude

def canny_edge(img, t_lower=50, t_upper=150):
    """Deteksi tepi Canny."""
    gray = rgb_to_grayscale(img)
    return cv2.Canny(gray, t_lower, t_upper)

def prewitt_edge(img):
    """Deteksi tepi Prewitt menggunakan konvolusi kernel manual."""
    gray = rgb_to_grayscale(img)
    # Kernel Prewitt
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    # Filter citra
    img_prewittx = cv2.filter2D(gray, -1, kernelx)
    img_prewitty = cv2.filter2D(gray, -1, kernely)
    # Magnitude perkiraan
    magnitude = cv2.addWeighted(img_prewittx, 0.5, img_prewitty, 0.5, 0)
    return magnitude

def threshold_segmentation(img, thresh_val=127, method="Manual"):
    """Segmentasi citra menggunakan thresholding (Manual atau Otsu)."""
    gray = rgb_to_grayscale(img)
    if method == "Otsu":
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
    return thresh

def kmeans_segmentation(img, k=3):
    """Segmentasi citra menggunakan K-Means Clustering (Machine Learning)."""
    # K-means butuh citra warna 3D diubah menjadi array 2D pixel
    pixel_values = img.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    
    # Kriteria penghentian algoritma (maksimal 100 iterasi atau akurasi epsilon 0.2)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    
    # Jalankan K-Means
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Konversi center kembali ke uint8
    centers = np.uint8(centers)
    
    # Petakan piksel ke warna centroid masing-masing cluster
    segmented_pixels = centers[labels.flatten()]
    
    # Reshape kembali ke dimensi citra asli
    segmented_img = segmented_pixels.reshape(img.shape)
    return segmented_img

def watershed_segmentation(img):
    """Segmentasi citra menggunakan algoritma Watershed."""
    # Pastikan citra berupa citra warna RGB/BGR
    if len(img.shape) == 2:
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_color = img.copy()
        
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    
    # Binarisasi Otsu terbalik (background hitam, objek putih)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Menghilangkan noise dengan opening morfologi
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Area background yang pasti (dilasi)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Area foreground yang pasti (menggunakan distance transform)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # Area perbatasan yang tidak diketahui (unknown)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Pelabelan komponen terhubung
    _, markers = cv2.connectedComponents(sure_fg)
    
    # Tambahkan 1 ke label agar sure background berlabel 1 (bukan 0)
    markers = markers + 1
    # Beri label 0 untuk area yang tidak diketahui
    markers[unknown == 255] = 0
    
    # Jalankan Watershed
    markers = cv2.watershed(img_color, markers)
    
    # Tandai batas segmentasi dengan warna merah
    img_color[markers == -1] = [0, 0, 255]
    return img_color

def detect_faces(img, scale_factor=1.1, min_neighbors=5):
    """Deteksi wajah Haar Cascade (Bonus Fitur)."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = rgb_to_grayscale(img)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)
    
    img_copy = img.copy()
    for (x, y, w, h) in faces:
        # Gambar kotak biru di sekeliling wajah yang terdeteksi
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
    return img_copy, len(faces)

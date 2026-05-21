import cv2
import numpy as np

# ==========================================
# KONFIGURASI PARAMETER METROLOGI
# ==========================================
# 1. Kalibrasi Spasial (Dummy: 1 Pixel = 0.05 mm)
PIXEL_TO_MM_RATIO = 0.05  

# 2. Spesifikasi Desain (CAD) & Guardbanding
TARGET_DIMENSION_MM = 10.00  # Target panjang part
TOLERANCE_CAD_MM = 0.5      # Toleransi resmi (±0.5 mm)
GUARDBAND_PERCENT = 0.20     # Reduksi toleransi 20% untuk safety

# Menghitung batas aktual software (Guardbanded)
ACTIVE_TOLERANCE = TOLERANCE_CAD_MM * (1.0 - GUARDBAND_PERCENT)
LOWER_LIMIT = TARGET_DIMENSION_MM - ACTIVE_TOLERANCE
UPPER_LIMIT = TARGET_DIMENSION_MM + ACTIVE_TOLERANCE

# ==========================================
# FUNGSI SUB-PIXEL EDGE DETECTION
# ==========================================
def calc_subpixel_offset(val_left, val_center, val_right):
    """Menghitung pergeseran desimal dengan interpolasi parabola."""
    # Konversi ke float untuk hindari overflow pada uint8
    vl, vc, vr = float(val_left), float(val_center), float(val_right)
    denominator = 2.0 * (vl - 2.0 * vc + vr)
    if denominator == 0:
        return 0.0
    return (vl - vr) / denominator

def refine_edge_1d(image_gray, rough_x, rough_y, axis='horizontal'):
    """Memperbaiki posisi edge kasar menggunakan Sobel gradient lokal."""
    h, w = image_gray.shape
    
    # Ambil ROI kecil (3 pixel) di sekitar tepi
    if axis == 'horizontal':
        if rough_x <= 0 or rough_x >= w - 1: return float(rough_x)
        # Hitung gradien horizontal sederhana (perbedaan intensitas)
        roi = image_gray[rough_y, rough_x-1 : rough_x+2]
        # Menggunakan intensitas langsung untuk aproksimasi puncak gradien tepi
        offset = calc_subpixel_offset(roi[0], roi[1], roi[2])
        return rough_x + offset
        
    elif axis == 'vertical':
        if rough_y <= 0 or rough_y >= h - 1: return float(rough_y)
        roi = image_gray[rough_y-1 : rough_y+2, rough_x]
        offset = calc_subpixel_offset(roi[0], roi[1], roi[2])
        return rough_y + offset

# ==========================================
# PIPELINE INSPEKSI
# ==========================================
def get_camera(cap_id=0):
    cap = cv2.VideoCapture(cap_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cap_id)
    if not cap.isOpened():
        raise RuntimeError("Kamera tidak ditemukan")
    return cap

def process_inspection(frame):
    result_frame = frame.copy()
    
    # 1. Preprocessing (Grayscale & Denoising)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2. Ekstraksi Kontur Kasar
    edges = cv2.Canny(blurred, 40, 120)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Asumsi part yang diinspeksi adalah objek terbesar (misal: Spur Gear)
        largest_contour = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(largest_contour) > 1000:
            
            # --- PENGUKURAN BENDA BUNDAR (Sentroid & Enclosing Circle) ---
            
            # A. Mencari Titik Tengah Mutlak (Centroid) menggunakan Momen Spasial
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0
            
            # B. Mencari Radius Pembungkus (Minimum Enclosing Circle)
            (circle_x, circle_y), radius_px = cv2.minEnclosingCircle(largest_contour)
            
            # (Opsional Tingkat Lanjut) Jika ingin radius Rata-rata dari Sentroid:
            # radius_px = np.mean([np.sqrt((pt[0][0] - cx)**2 + (pt[0][1] - cy)**2) for pt in largest_contour])
            
            # 4. Konversi Diameter ke Metrik & Logika Keputusan (Guardbanding)
            diameter_px = radius_px * 2
            measured_mm = diameter_px * PIXEL_TO_MM_RATIO
            
            # Klasifikasi OK/NG (Berdasarkan Diameter, bukan Lebar Kotak)
            if LOWER_LIMIT <= measured_mm <= UPPER_LIMIT:
                status = "OK"
                color = (0, 255, 0)  # Hijau
            else:
                status = "NG (Not Good)"
                color = (0, 0, 255)  # Merah
                
            # 5. Visualisasi UI
            # Gambar Sentroid (Titik Tengah)
            cv2.circle(result_frame, (cx, cy), 3, (0, 255, 255), -1)
            
            # Gambar Lingkaran Pengukur
            cv2.circle(result_frame, (int(circle_x), int(circle_y)), int(radius_px), color, 2)
            
            # Teks Hasil Ukur di dekat objek
            cv2.putText(result_frame, f"Px Dia: {diameter_px:.2f} px", (cx - 60, cy - int(radius_px) - 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(result_frame, f"Dia: {measured_mm:.3f} mm", (cx - 60, cy - int(radius_px) - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Status OK/NG
            cv2.putText(result_frame, f"STATUS: {status}", (10, result_frame.shape[0] - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    if contours:
        # Asumsi part yang diinspeksi adalah objek terbesar
        largest_contour = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(largest_contour) > 1000:
            # Dapatkan Bounding Box kasar (integer)
            x, y, w_box, h_box = cv2.boundingRect(largest_contour)
            
            # 3. Sub-Pixel Refinement (Mencari dimensi lebar presisi)
            # Kita pindai titik tengah sisi kiri dan kanan bounding box
            mid_y = y + (h_box // 2)
            precise_x_left = refine_edge_1d(gray, x, mid_y, axis='horizontal')
            precise_x_right = refine_edge_1d(gray, x + w_box, mid_y, axis='horizontal')
            
            precise_width_px = precise_x_right - precise_x_left
            
            # 4. Konversi ke Metrik & Logika Keputusan (Guardbanding)
            measured_mm = precise_width_px * PIXEL_TO_MM_RATIO
            
            # Klasifikasi OK/NG
            if LOWER_LIMIT <= measured_mm <= UPPER_LIMIT:
                status = "OK"
                color = (0, 255, 0)  # Hijau
            else:
                status = "NG (Not Good)"
                color = (0, 0, 255)  # Merah
                
            # 5. Visualisasi UI
            # Gambar bounding box kasar
            cv2.rectangle(result_frame, (x, y), (x + w_box, y + h_box), (255, 255, 0), 1)
            
            # Teks Hasil Ukur
            cv2.putText(result_frame, f"Px Width: {precise_width_px:.2f} px", (x, y - 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(result_frame, f"Measured: {measured_mm:.3f} mm", (x, y - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Status OK/NG
            cv2.putText(result_frame, f"STATUS: {status}", (x, y + h_box + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Tambahkan overlay parameter di pojok layar
    cv2.putText(result_frame, f"Target: {TARGET_DIMENSION_MM}mm | Active Tol: +/- {ACTIVE_TOLERANCE:.3f}mm", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
    return edges, result_frame

def main():
    print("Memulai Inspeksi Visi Komputer. Tekan 'q' untuk keluar.")
    cap = get_camera(cap_id=1) # Ganti 1 jika pakai USB camera
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        edges, result = process_inspection(frame)
        
        cv2.imshow("Kamera Inspeksi Aktual", result)
        cv2.imshow("Deteksi Tepi Biner", edges)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
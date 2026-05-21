import cv2
import numpy as np
from collections import deque

# -----------------------------------------
# KONFIGURASI PARAMETER METROLOGI
# -----------------------------------------

# Spesifikasi desain (CAD) & guardbanding
TARGET_DIMENSION_MM = 25.50
TOLERANCE_CAD_MM = 0.5
GUARDBAND_PERCENT = 0.20
ARUCO_SIZE_MM = 20.00

# Menghitung margin guardbanded
ACTIVE_TOLERANCE = TOLERANCE_CAD_MM * (1.0 - GUARDBAND_PERCENT)
LOWER_LIMIT = TARGET_DIMENSION_MM - ACTIVE_TOLERANCE
UPPER_LIMIT = TARGET_DIMENSION_MM + ACTIVE_TOLERANCE

# Variabel global
current_ratio = 0.05
measurement_history = deque(maxlen = 10)

# -----------------------------------------
# FUNGSI ARUCO
# -----------------------------------------
def get_dynamic_ratio(frame):
    # Memakai dict ArUco 4 x 4
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()

    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(frame)

    if ids is not None and len(corners) > 0:
        # Tampilkan kotak ArUco
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Hitung jarak pixel antara dua sudut
        marker_corners = corners[0][0]
        dist_px = np.linalg.norm(marker_corners[0] - marker_corners[1])

        if (dist_px > 0):
            return ARUCO_SIZE_MM / float(dist_px)
    
    return None

# -----------------------------------------
# FUNGSI SUB-PIXEL EDGE DETECTION
# -----------------------------------------
def calc_subpixel_offset(val_left, val_center, val_right):
    vl, vc, vr  = float(val_left), float(val_center), float(val_right)
    denominator = 2.0 * (vl - 2.0 * vc + vr)
    if denominator == 0:
        return 0.0
    return (vl - vr) / denominator

def refine_edge_1d(image_gray, rough_x, rough_y, axis = 'horizontal'):
    h, w = image_gray.shape
    if axis == 'horizontal':
        if rough_x <= 0 or rough_x >= w - 1: 
            return float(rough_x)
        roi = image_gray[rough_y, rough_x - 1 : rough_x + 2]
        offset = calc_subpixel_offset(roi[0], roi[1], roi[2])
        return rough_x + offset
    return None
    
# -----------------------------------------
# PIPELINE INSPEKSI UTAMA
# -----------------------------------------
def get_camera(cap_id = 1):
    cap = cv2.VideoCapture(cap_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cap_id)
    if not cap.isOpened():
        raise RuntimeError("Kamera tidak ditemukan, sir")
    
    # Kalo bisa matikan autofocus
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    return cap

def process_inspection(frame):
    global current_ratio
    result_frame = frame.copy()

    # 1. Update rasio mm/px dari ArUco
    ratio = get_dynamic_ratio(frame)
    if ratio is not None:
        current_ratio = ratio
        cv2.putText(result_frame, f"ArUco OK: 1px = {current_ratio:.4f}mm", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(result_frame, "ArUco NOT DETECTED", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    
    # 2. Preprocessing (grayscale & denoising)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Ekstraksi kontur 
    edges = cv2.Canny(blurred, 40, 120)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # menari kontur terbesar (abaikan objek ArUco)
        valid_contours = [c for c in contours if cv2.contourArea(c) > 1500]

        if valid_contours:
            largest_contour = max(valid_contours, key = cv2.contourArea)

            # Pengaturan tahan rotasi (min area rect), benda bisa ditaruh miring
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            box = box.astype(np.intp)

            # ekstrak lebar & tinggi dari rect
            (cx, cy), (w_px, h_px), angle = rect

            # ambil sisi terpanjang sebagai dimensi utama
            primary_dim_px = max(w_px, h_px)

            # 4. Sub-pixel refinement
            x, y, w_box, h_box = cv2.boundingRect(largest_contour)
            mid_y = y + (h_box // 2)
            precise_x_left = refine_edge_1d(gray, x, mid_y, 'horizontal')
            precise_x_right = refine_edge_1d(gray, x + w_box, mid_y, 'horizontal')

            if precise_x_left is not None and precise_x_right is not None:
                refined_width_px = precise_x_right - precise_x_left
            else:
                refined_width_px = primary_dim_px
            
            # 5. Konversi metrik dan temporal smoothing
            raw_mm = refined_width_px * current_ratio
            measurement_history.append(raw_mm)

            # Hitung rata2 agar angka tidak flickering
            smoothed_mm = sum(measurement_history) / len(measurement_history)

            # 6. Klasifikasi OK/NG berdasarkan guardbanded limits
            if LOWER_LIMIT <= smoothed_mm <= UPPER_LIMIT:
                status = "OK"
                color = (0, 255, 0) # green
            else:
                status = "NG"
                color = (0, 0, 255) # red

            # 7. Visualisasi UI
            # Tampilkan bounding box & hasil pengukuran (light blue)
            cv2.drawContours(result_frame, [box], 0, (255, 255, 0), 2)

            # Midpoint
            cv2.circle(result_frame, (int(cx), int(cy)), 4, (0, 0, 255), -1)

            # Teks hasil pengukuran
            cv2.putText(result_frame, f"Measured: {smoothed_mm: .2f} mm" , (int(cx) - 80, int(cy) - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Status OK/NG
            cv2.putText(result_frame, f"Status: {status}", (10, result_frame.shape[0] - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)
            
            # Info parameter desain & toleransi
            cv2.putText(result_frame, f"Target: {TARGET_DIMENSION_MM}mm | Active Tol: +/- {ACTIVE_TOLERANCE:.2f}mm", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    return edges, result_frame

def main():
    print("Memulai inspeksi metrologi. Tekan 'q' untuk keluar.")
    cap = get_camera(cap_id = 1)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame kamera, sir")
            break

        edges, result_frame = process_inspection(frame)

        cv2.imshow("Kamera Inspeksi", result_frame)
        cv2.imshow("Sistem deteksi tepi", edges)

        # Tekan 'q' untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
        
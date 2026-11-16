# Air Canvas

Menggambar di udara memakai gerakan jari. Program membaca tangan lewat webcam (MediaPipe Hand Landmarker), lalu coretannya muncul di atas video secara langsung. Tidak perlu mouse atau layar sentuh.

## Fitur

- Menggambar, memilih warna, dan menghapus hanya dengan gestur jari
- Mendukung dua tangan sekaligus, tiap tangan bekerja sendiri-sendiri
- Garis dihaluskan supaya tidak bergetar
- Undo sampai 15 langkah
- Simpan hasil sebagai PNG transparan
- Efek: glow, warna pelangi, simetri cermin, dan latar papan gelap

## Kebutuhan

- Python 3.10 atau lebih baru
- Webcam
- File model `hand_landmarker.task` (sudah ada di repo ini)

## Instalasi

```bash
git clone https://github.com/faizroyan12/Air-Canvas.git
cd Air-Canvas

python -m venv env
env\Scripts\activate        # Windows (cmd)
# source env/bin/activate   # Linux / macOS

pip install -r requirements.txt
```

Di PowerShell, aktifkan environment dengan `env\Scripts\Activate.ps1`.

## Menjalankan

```bash
python video.py
```

Jendela kamera terbuka dalam mode layar penuh. Tekan `F` untuk beralih ke mode jendela.

## Gestur

| Gestur | Fungsi |
| --- | --- |
| Telunjuk saja | Menggambar |
| Telunjuk + jari tengah | Pilih warna (arahkan ujung telunjuk ke kotak warna di bar atas) |
| Telunjuk + tengah + manis | Penghapus |

Kelingking dibiarkan tertekuk untuk ketiga gestur di atas. Satu tangan menggambar sambil tangan lain memilih warna juga bisa.

## Tombol keyboard

| Tombol | Fungsi |
| --- | --- |
| `C` | Bersihkan kanvas |
| `Z` | Undo |
| `S` | Simpan gambar sebagai PNG transparan |
| `+` / `-` | Ubah ketebalan kuas |
| `G` | Efek glow |
| `R` | Warna pelangi |
| `M` | Simetri cermin |
| `B` | Latar papan gelap |
| `F` | Layar penuh / jendela |
| `Q` | Keluar |

## Pengaturan

Beberapa nilai bisa diubah di bagian konfigurasi paling atas `video.py`:

- `SMOOTHING`: kehalusan garis (naikkan ke 0.7 kalau terasa lambat)
- `ERASER_THICKNESS`: ukuran penghapus
- `COLORS`: daftar warna di toolbar
- `UNDO_LIMIT`: jumlah langkah undo

## Tips

- Pastikan ruangan cukup terang dan tangan terlihat jelas di kamera.
- Kalau kamera salah, program mencoba kamera nomor 1 dulu lalu nomor 0.
- Jauhkan kedua tangan satu sama lain supaya tidak tertukar saat dideteksi.

## Teknologi

- [OpenCV](https://opencv.org/)
- [MediaPipe](https://ai.google.dev/edge/mediapipe)
- NumPy

import os
import requests

def download_dataset():
    # Daftar 20 gambar buah-buahan dari Unsplash
    urls = {
        "01_apel.jpg": "https://images.unsplash.com/photo-1560806887-1e4cd0b6cbd6?w=500&auto=format&fit=crop&q=60",
        "02_pisang.jpg": "https://images.unsplash.com/photo-1571771894821-ce9b6c11b08e?w=500&auto=format&fit=crop&q=60",
        "03_jeruk.jpg": "https://images.unsplash.com/photo-1611080626919-7cf5a9dbab5b?w=500&auto=format&fit=crop&q=60",
        "04_stroberi.jpg": "https://images.unsplash.com/photo-1464965911861-746a04b4bca6?w=500&auto=format&fit=crop&q=60",
        "05_anggur.jpg": "https://images.unsplash.com/photo-1537640538966-79f369143f8f?w=500&auto=format&fit=crop&q=60",
        "06_mangga.jpg": "https://images.unsplash.com/photo-1553279768-865429fa0078?w=500&auto=format&fit=crop&q=60",
        "07_nanas.jpg": "https://images.unsplash.com/photo-1550258987-190a2d41a8ba?w=500&auto=format&fit=crop&q=60",
        "08_semangka.jpg": "https://images.unsplash.com/photo-1587049352846-4a222e784d38?w=500&auto=format&fit=crop&q=60",
        "09_minuman.jpg": "https://images.unsplash.com/photo-1527661591475-527312dd65f5?w=500&auto=format&fit=crop&q=60",
        "10_pisang2.jpg": "https://images.unsplash.com/photo-1603052875302-d376b7c0638a?w=500&auto=format&fit=crop&q=60",
        "11_lemon.jpg": "https://images.unsplash.com/photo-1590502593747-42a996133562?w=500&auto=format&fit=crop&q=60",
        "12_alpukat.jpg": "https://images.unsplash.com/photo-1523049673857-eb18f1d7b578?w=500&auto=format&fit=crop&q=60",
        "13_pir.jpg": "https://images.unsplash.com/photo-1514756331096-242fdeb70d4a?w=500&auto=format&fit=crop&q=60",
        "14_nasigoreng.jpg": "https://images.unsplash.com/photo-1595908129746-57ca1a63dd4d?w=500&auto=format&fit=crop&q=60",
        "15_ramen.jpg": "https://images.unsplash.com/photo-1526318896980-cf78c088247c?w=500&auto=format&fit=crop&q=60",
        "16_cat.jpg": "https://images.unsplash.com/photo-1589883661923-6476cb0ae9f2?w=500&auto=format&fit=crop&q=60",
        "17_strawberry.jpg": "https://images.unsplash.com/photo-1601004890684-d8cbf643f5f2?w=500&auto=format&fit=crop&q=60",
        "18_jeruk_bali.jpg": "https://images.unsplash.com/photo-1577234286642-fc512a5f8f11?w=500&auto=format&fit=crop&q=60",
        "19_dataex.jpg": "https://images.unsplash.com/photo-1528825871115-3581a5387919?w=500&auto=format&fit=crop&q=60",
        "20_tas.jpg": "https://images.unsplash.com/photo-1622560480605-d83c853bc5c3?w=500&auto=format&fit=crop&q=60",
    }

    output_dir = "dataset"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Membuat direktori '{output_dir}'...")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }

    print("Memulai pengunduhan dataset 20 gambar buah...")
    for idx, (filename, url) in enumerate(urls.items(), 1):
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            print(f"[{idx}/20] Berkas '{filename}' sudah ada, melewati...")
            continue
        
        try:
            print(f"[{idx}/20] Mengunduh {filename}...", end="", flush=True)
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code == 200:
                with open(filepath, "wb") as f:
                    f.write(response.content)
                print(" Berhasil!")
            else:
                print(f" Gagal (Status Code: {response.status_code})")
        except Exception as e:
            print(f" Error: {e}")
            
    print("\nPengunduhan selesai. Gambar yang tersedia di folder 'dataset':")
    print(os.listdir(output_dir))

if __name__ == "__main__":
    download_dataset()

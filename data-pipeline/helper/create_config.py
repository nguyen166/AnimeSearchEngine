import json
import os
import re
from urllib.parse import urlparse

def extract_anime_id(url: str) -> str:
    """
    Tự động lấy anime_id từ URL.
    Ví dụ: https://vuighe.cam/naruto/ -> naruto
    """
    # Xóa dấu gạch chéo cuối nếu có
    url = url.strip().rstrip('/')
    
    # Lấy phần cuối cùng của URL
    path_parts = url.split('/')
    anime_id = path_parts[-1]
    
    # Nếu user lỡ copy link tập phim (ví dụ: .../naruto/tap-1), hãy cắt bỏ phần 'tap-1'
    if "tap-" in anime_id:
        anime_id = path_parts[-2]
        
    return anime_id

def clean_base_url(url: str) -> str:
    """
    Đảm bảo URL base sạch đẹp để nối chuỗi
    """
    url = url.strip().rstrip('/')
    # Nếu URL kết thúc bằng 'tap-xxx', cắt bỏ nó đi để lấy root
    if re.search(r'/tap-\d+$', url):
        url = re.sub(r'/tap-\d+$', '', url)
    return url

def main():
    print("="*50)
    print("🛠️  CÔNG CỤ TẠO CONFIG CRAWLER TỰ ĐỘNG")
    print("="*50)

    # 1. Nhập URL
    while True:
        url_input = input("👉 Nhập link Anime (VD: https://vuighe.cam/chu-thuat-hoi-chien-phan-2/): ").strip()
        if url_input.startswith("http"):
            break
        print("❌ URL không hợp lệ! Phải bắt đầu bằng http hoặc https.")

    # 2. Nhập số tập
    while True:
        try:
            start_ep = int(input("👉 Từ tập số: "))
            end_ep = int(input("👉 Đến tập số: "))
            if start_ep > 0 and end_ep >= start_ep:
                break
            print("❌ Số tập không hợp lệ (Phải > 0 và 'Đến tập' >= 'Từ tập')")
        except ValueError:
            print("❌ Vui lòng nhập số nguyên.")

    # 3. Xử lý dữ liệu
    anime_id = extract_anime_id(url_input)
    base_url = clean_base_url(url_input)
    
    print(f"\n✅ Đã nhận diện Anime ID: {anime_id}")
    print(f"✅ Base URL: {base_url}")

    episodes_list = []
    for i in range(start_ep, end_ep + 1):
        # Quy tắc link của VuiGhe: base_url + /tap-i
        ep_url = f"{base_url}/tap-{i}"
        episodes_list.append({
            "episode": i,
            "url": ep_url
        })

    # 4. Tạo cấu trúc JSON
    config_data = {
        "output_dir": "./data/raw_videos",
        "headless": True,
        "delay_between_episodes": 5,
        "anime": [
            {
                "anime_id": anime_id,
                "season": "",  # Để trống hoặc bạn có thể input thêm nếu muốn
                "episodes": episodes_list
            }
        ]
    }

    # 5. Lưu file
    output_filename = f"config_{anime_id}.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)

    print("\n" + "="*50)
    print(f"🎉 Đã tạo xong file config: {output_filename}")
    print(f"📂 Tổng số tập: {len(episodes_list)}")
    print("="*50)
    print(f"\n🚀 Để chạy crawler, hãy dùng lệnh:")
    print(f"   python run_crawler.py --config {output_filename}")

if __name__ == "__main__":
    main()
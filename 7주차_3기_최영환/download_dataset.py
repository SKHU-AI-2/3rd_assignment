import requests
import os
import my_api

ACCESS_KEY = my_api.ACCESS_KEY
SEARCH_URL = "https://api.unsplash.com/search/photos"

def search_photos(query, page=1, per_page=30):
    # API 요청 파라미터
    params = {
        "query": query,
        "page": page,
        "per_page": per_page,
        "client_id": ACCESS_KEY
    }
    
    response = requests.get(SEARCH_URL, params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch photos: {response.status_code}, {response.text}")
        return None

def download_photo(photo_url, save_path):
    response = requests.get(photo_url, stream=True)
    if response.status_code == 200:
        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        print(f"Photo saved to {save_path}")
    else:
        print("Failed to download photo.")

def download_photos(query, max_photos, save_dir):
    total_downloaded = 0
    page = 1
    per_page = 30

    while total_downloaded < max_photos:
        result = search_photos(query, page=page, per_page=per_page)
        if not result or "results" not in result:
            break

        for photo in result["results"]:
            photo_url = photo["urls"]["small"]
            save_path = os.path.join(save_dir, f"{total_downloaded}.jpg")
            download_photo(photo_url, save_path)
            total_downloaded += 1

            if total_downloaded >= max_photos:
                break

        page += 1  # 다음 페이지로 이동

    print(f"Total photos downloaded: {total_downloaded}")

max_photos = 500

query = "person sitting on chair"
save_dir = "data1/sit"
os.makedirs(save_dir, exist_ok=True)
download_photos(query, max_photos, save_dir=save_dir)

query = "person standing"
save_dir = "data1/stand"
os.makedirs(save_dir, exist_ok=True)
download_photos(query, max_photos, save_dir=save_dir)

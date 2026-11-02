import os
import subprocess
import sys

def install_and_import(package):
    try:
        __import__(package)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def download_youtube_video(url, output_path="sample_video.mp4"):
    print("Setting up dependencies...")
    install_and_import("yt_dlp")
    import yt_dlp
    
    print(f"Downloading video from {url}...")
    
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': output_path,
        'merge_output_format': 'mp4',
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print(f"Successfully downloaded to {output_path}")
    except Exception as e:
        print(f"Error downloading video: {e}")

if __name__ == "__main__":
    # Example video: A public domain or standard action clip can be used here.
    # For now, we use a placeholder URL. Please replace it with a valid YouTube URL
    # that contains weapons/people for testing (e.g., a movie scene or CCTV clip).
    default_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ" # Replace with an actual CCTV/weapon clip URL
    
    url = input(f"Enter YouTube URL to download (default: test video): ")
    if not url.strip():
        url = default_url
        
    download_youtube_video(url, "sample_test_video.mp4")

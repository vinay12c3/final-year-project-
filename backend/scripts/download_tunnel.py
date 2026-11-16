"""Download cloudflared, used to share a conversation room over the internet (about 40 MB, one time)."""
from signvision import tunnel

if __name__ == "__main__":
    if tunnel.is_downloaded():
        print(f"cloudflared is already in {tunnel.binary_path()}")
    else:
        tunnel.download()
        print(f"saved to {tunnel.binary_path()}")

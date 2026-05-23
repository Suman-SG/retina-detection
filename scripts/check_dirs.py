import os

root = "data"
print("📁 Listing all folders under /data:")
for root, dirs, files in os.walk(root):
    print("DIR:", root, "FILES:", len(files))
print("\n✅ DONE!")
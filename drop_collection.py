"""
Script to drop Milvus collection (for dimension change)
"""
from pymilvus import connections, utility

# Connect to Milvus
connections.connect(
    alias="default",
    host="localhost",
    port="19530"
)

collection_name = "anime_frames"

# Drop collection if exists
if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)
    print(f"✅ Dropped collection: {collection_name}")
else:
    print(f"⚠️ Collection {collection_name} does not exist")

# Disconnect
connections.disconnect("default")
print("✅ Done! Collection will be recreated with CLIP (512 dim) when API starts.")

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

class VectorDB:
    def __init__(self):
        # Initialize a local, in-memory vector database (runs entirely inside your script)
        self.client = QdrantClient(":memory:")
        # Create a storage collection named "satellite_archive"
        # ResNet-50 outputs vectors of size 2048
        self.client.create_collection(
            collection_name="satellite_archive",
            vectors_config=VectorParams(size=2048, distance=Distance.COSINE),
        )

    def add_image_to_archive(self, image_id, vector, metadata):
        # Insert a vector into the database along with its file path info
        self.client.upsert(
            collection_name="satellite_archive",
            points=[
                PointStruct(id=image_id, vector=vector.tolist(), payload=metadata)
            ]
        )

    def search_similar(self, query_vector, top_k=10):
        # Look through millions of vectors in a millisecond using Cosine Similarity math
        search_result = self.client.search(
            collection_name="satellite_archive",
            query_vector=query_vector.tolist(),
            limit=top_k
        )
        return search_result
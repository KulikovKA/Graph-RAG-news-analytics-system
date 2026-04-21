import os
import sys
from dotenv import load_dotenv
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# Fix path to allow importing from src if needed
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

# Load configuration
load_dotenv(os.path.join(PROJECT_ROOT, '.env.db'))

class GraphManager:
    def __init__(self):
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "mysecretpassword")
        self.driver = None

    def connect(self):
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self.driver.verify_connectivity()
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Neo4j at {self.uri}: {e}")

    def check_connection(self):
        with self.driver.session() as session:
            result = session.run("RETURN 1")
            return result.single()[0] == 1

    def setup_schema(self):
        """Create unique constraints for nodes."""
        constraints = [
            "CREATE CONSTRAINT person_id_unique IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT doc_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE"
        ]
        with self.driver.session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                    print(f"Applied: {constraint}")
                except Exception as e:
                    print(f"Warning: Could not apply constraint '{constraint}': {e}")

    def clear_all(self):
        """Delete all nodes and relationships (debug only)."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("Graph cleared.")

    def close(self):
        if self.driver:
            self.driver.close()

class VectorManager:
    def __init__(self):
        self.host = os.getenv("QDRANT_HOST", "localhost")
        self.port = int(os.getenv("QDRANT_PORT", 6333))
        self.client = None

    def connect(self):
        try:
            self.client = QdrantClient(host=self.host, port=self.port)
            # check connection by getting collections
            self.client.get_collections()
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Qdrant at {self.host}:{self.port}: {e}")

    def init_collection(self, collection_name, vector_size=768):
        """Initialize a Qdrant collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        exists = any(c.name == collection_name for c in collections)
        
        if not exists:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            print(f"Collection '{collection_name}' created (Size: {vector_size}, Metric: Cosine).")
        else:
            print(f"Collection '{collection_name}' already exists.")

if __name__ == "__main__":
    print("Initalizing infrastructure components...")
    
    try:
        # 1. Neo4j
        gm = GraphManager()
        gm.connect()
        if gm.check_connection():
            print("Neo4j status: Connected")
        gm.setup_schema()
        
        # 2. Qdrant
        vm = VectorManager()
        vm.connect()
        print("Qdrant status: Ready")
        vm.init_collection("news_segments", vector_size=768) # Default for Multilingual-E5
        
        print("\nInfrastructure initialization successful!")
    except ConnectionError as ce:
        print(f"\nCRITICAL ERROR: {ce}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {e}")
        sys.exit(1)
    finally:
        if 'gm' in locals():
            gm.close()

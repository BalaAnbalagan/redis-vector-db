# Redis Vector Database with OpenAI Embeddings

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BalaAnbalagan/redis-vector-db/blob/main/redis_vector_database.ipynb)
[![View Notebook](https://img.shields.io/badge/View-Notebook-orange)](https://github.com/BalaAnbalagan/redis-vector-db/blob/main/redis_vector_database.ipynb)

A practical implementation of semantic search using Redis as a vector database, demonstrating how modern AI applications store and search through embedded text using vector similarity.

---

## What This Project Does

This project shows how to build a semantic search system that understands *meaning* rather than just matching keywords. Instead of searching for exact words, you can search by concepts and get relevant results even if they use different terminology.

**Example:** Searching for "space exploration" finds articles about NASA, rockets, and astronauts - even if those exact words aren't in your query.

---

## The Problem We're Solving

Traditional databases search for exact matches. If you search for "automobile," you won't find articles about "cars" or "vehicles." Vector databases solve this by converting text into numerical representations (vectors) that capture meaning, allowing us to find semantically similar content.

---

## How It Works

### 1. Text Embeddings

We convert text into vectors (arrays of numbers) using OpenAI's `text-embedding-3-small` model. Similar concepts produce similar vectors.

```
"cat" → [0.23, 0.67, -0.12, ..., 0.45]  (1536 numbers)
"kitten" → [0.24, 0.66, -0.11, ..., 0.44]  (very similar!)
"car" → [-0.31, 0.02, 0.87, ..., -0.23]  (very different)
```

### 2. Vector Storage in Redis

Redis stores these vectors efficiently and provides fast similarity search through its RediSearch module. Think of it as a specialized database optimized for finding "similar" items rather than exact matches.

### 3. Similarity Search

When you search, your query is converted to a vector, then Redis finds the closest matching vectors using cosine similarity - a mathematical way to measure how "close" two vectors are.

### 4. Hybrid Search

You can combine vector similarity with traditional filters:
- Find articles about "art" (semantic) that mention "Leonardo da Vinci" (exact text)
- This gives you the best of both worlds

---

## Project Architecture

```
User Query
    ↓
OpenAI Embeddings API (converts text → vector)
    ↓
Redis Cloud (stores vectors, performs similarity search)
    ↓
Results (ranked by similarity)
```

**Components:**
- **OpenAI API**: Generates 1536-dimensional embeddings from text
- **Redis Cloud**: Free cloud-hosted Redis with RediSearch module
- **Wikipedia Dataset**: 2,500 pre-embedded articles for searching
- **Python Notebook**: Interactive demonstration of all concepts

---

## Getting Started

### Prerequisites

- Python 3.9 or higher
- OpenAI API key ([get one here](https://platform.openai.com/api-keys))
- Redis Cloud account (free tier, no credit card needed)

### Installation

1. **Clone this repository**
   ```bash
   git clone https://github.com/BalaAnbalagan/redis-vector-db.git
   cd redis-vector-db
   ```

2. **Install Python dependencies**
   ```bash
   pip install redis openai pandas numpy wget
   ```

3. **Set up Redis Cloud** (takes 5 minutes)
   - Sign up at [redis.com/try-free](https://redis.com/try-free/)
   - Create a free database (30MB)
   - Enable the "Search and query" module (this is RediSearch)
   - Note your connection details (host, port, password)

4. **Configure credentials**
   ```bash
   # Copy the template
   cp redis_config.json.example redis_config.json

   # Edit with your credentials
   nano redis_config.json
   ```

   Add your Redis Cloud and OpenAI credentials:
   ```json
   {
     "redis_host": "your-redis-host.redis-cloud.com",
     "redis_port": 12345,
     "redis_password": "your-redis-password",
     "openai_api_key": "sk-your-openai-key"
   }
   ```

5. **Run the notebook**
   ```bash
   jupyter notebook redis_vector_database.ipynb
   ```

---

## Understanding the Code

### Step 1: Loading Pre-Embedded Data

The notebook downloads 25,000 Wikipedia articles that have already been converted to vectors. We use only 2,500 to fit in Redis Cloud's free tier.

```python
# Each article has both title and content embeddings
article_df = pd.read_csv('wikipedia_articles_embedded.csv')
article_df = article_df.head(2500)  # Use subset for free tier
```

**Why pre-embedded?** Creating embeddings for 25,000 articles would cost ~$0.30 and take time. The OpenAI Cookbook provides these pre-computed.

### Step 2: Creating a Search Index

We tell Redis how to index our vectors:

```python
# Define vector fields
VectorField("title_vector", "FLAT", {
    "TYPE": "FLOAT32",
    "DIM": 1536,  # OpenAI embedding size
    "DISTANCE_METRIC": "COSINE"  # How to measure similarity
})
```

**COSINE distance** measures the angle between vectors. Smaller angles = more similar content.

### Step 3: Loading Data into Redis

We convert Python lists to binary format and store them:

```python
# Convert vector to bytes for Redis storage
embedding_bytes = np.array(vector, dtype=np.float32).tobytes()
redis_client.hset(key, mapping=document)
```

**Why bytes?** Redis stores binary data more efficiently than text representations of numbers.

### Step 4: Semantic Search

Find similar content by converting your query to a vector:

```python
# Your search query
query = "modern art in Europe"

# OpenAI converts it to a vector
query_vector = openai.Embedding.create(
    input=query,
    model="text-embedding-3-small"
)

# Redis finds similar vectors (KNN = K-Nearest Neighbors)
results = redis_client.ft("embeddings-index").search(
    f"*=>[KNN 10 @title_vector $vector AS score]",
    {"vector": query_vector}
)
```

**Results you might get:**
1. Museum of Modern Art (Score: 0.875)
2. Renaissance art (Score: 0.864)
3. Pop art (Score: 0.860)

### Step 5: Hybrid Search

Combine semantic and exact matching:

```python
# Find articles about "art" that mention "Leonardo da Vinci"
results = search_redis(
    query="Art",
    hybrid_fields='@text:"Leonardo da Vinci"'  # Exact text filter
)
```

This ensures results are semantically about art AND contain that specific phrase.

---

## Key Concepts Explained

### What is a Vector Database?

A vector database specializes in storing and searching high-dimensional vectors. Unlike traditional databases that search for exact matches, vector databases find similar items using mathematical distance metrics.

**Traditional Database:**
```sql
SELECT * FROM articles WHERE title = 'Cat'  -- Only finds "Cat"
```

**Vector Database:**
```
Find articles similar to [0.23, 0.67, ...]  -- Finds "Cat", "Kitten", "Feline"
```

### What are Embeddings?

Embeddings are numerical representations of text that capture semantic meaning. Words or sentences with similar meanings have similar embeddings.

**How they're created:**
- Large neural networks (like GPT) learn to encode text
- Each dimension captures different aspects of meaning
- OpenAI's model uses 1536 dimensions
- The specific numbers are learned from billions of text examples

### Why Use Redis?

Redis is extremely fast (in-memory storage) and the RediSearch module adds:
- Vector indexing (FLAT or HNSW algorithms)
- Similarity search (cosine, L2, or inner product)
- Hybrid queries (combine vectors with filters)
- Real-time indexing and search

### Distance Metrics

**Cosine Similarity:**
- Measures the angle between vectors
- Range: -1 (opposite) to 1 (identical)
- Good for text embeddings
- Ignores magnitude, focuses on direction

**Why we use it:** Text embeddings care more about meaning (direction) than absolute values (magnitude).

---

## Example Use Cases

### 1. Semantic Search
Search for "renewable energy" and find articles about solar panels, wind turbines, and hydroelectric power - even if they don't contain those exact words.

### 2. Recommendation Systems
"Users who liked this article also liked..." based on content similarity, not just tags or categories.

### 3. Question Answering
Store your documentation as vectors, then find the most relevant sections when someone asks a question.

### 4. Content Deduplication
Find similar or duplicate articles even if the wording is different.

### 5. Clustering and Classification
Group similar content together or classify new content based on similarity to labeled examples.

---

## Project Structure

```
redis-vector-db/
├── redis_vector_database.ipynb    # Main Jupyter notebook (with outputs)
├── redis_config.json              # Your credentials (NOT in Git)
├── redis_config.json.example      # Template for credentials
├── .gitignore                     # Protects credentials
└── README.md                      # This file
```

---

## Dataset Details

**Source:** OpenAI Cookbook - Pre-embedded Wikipedia articles

**Full Dataset:**
- 25,000 articles from Wikipedia
- Each has title and content embeddings
- Size: ~700MB compressed
- Model: text-embedding-3-small

**This Project Uses:**
- 2,500 articles (subset)
- Fits in Redis Cloud free tier (30MB)
- Sufficient for learning all concepts

**Fields per Article:**
- `id`: Unique identifier
- `title`: Article title
- `text`: Article content
- `url`: Wikipedia source
- `title_vector`: 1536-dimensional embedding of title
- `content_vector`: 1536-dimensional embedding of content

---

## Performance Characteristics

**Embedding Generation:**
- Model: text-embedding-3-small
- Speed: ~1000 tokens/second
- Cost: $0.00002 per 1K tokens
- Dimension: 1536

**Redis Search:**
- Index type: FLAT (exact nearest neighbor)
- Search time: <100ms for 2,500 documents
- Memory: ~15MB for this dataset
- Alternative: HNSW (approximate, faster for large datasets)

**Trade-offs:**
- FLAT: Exact results, slower for large datasets
- HNSW: Approximate results (99%+ accuracy), much faster

---

## Extending This Project

### Add More Data
```python
# Embed your own text
new_embedding = openai.Embedding.create(
    input="Your custom text",
    model="text-embedding-3-small"
)

# Store in Redis
redis_client.hset(f"doc:{new_id}", mapping={
    "text": "Your custom text",
    "vector": np.array(new_embedding).tobytes()
})
```

### Try Different Search Parameters
```python
# Change K (number of results)
results = search_redis(query, k=50)

# Search different fields
results = search_redis(query, vector_field="content_vector")

# Add multiple filters
hybrid_fields = '(@category:{technology}) (@date:[2023 2024])'
```

### Implement Your Own Use Case
- Product search
- Document retrieval
- Code search
- Image search (with CLIP embeddings)

---

## Common Questions

**Q: Why not just use keyword search?**
A: Keyword search misses synonyms, related concepts, and different phrasings. "Automobile" won't match "car." Embeddings understand semantic similarity.

**Q: How are embeddings different from word vectors like Word2Vec?**
A: Modern embeddings (like OpenAI's) are contextual - the same word has different embeddings based on surrounding text. They also work on entire sentences/paragraphs.

**Q: Why Redis instead of specialized vector databases like Pinecone or Weaviate?**
A: Redis is familiar to many developers, offers a free tier, and provides additional features (caching, pub/sub, etc.). It's a good starting point for learning.

**Q: Can I use this for production?**
A: The free tier is for learning. For production, consider Redis Enterprise, or scale to larger tiers with HNSW indexing for performance.

**Q: What about data privacy?**
A: Your data is sent to OpenAI's API for embedding. For sensitive data, consider self-hosted embedding models or OpenAI's Azure offering with data privacy guarantees.

---

## Learning Resources

**Vector Databases:**
- [Pinecone Learning Center](https://www.pinecone.io/learn/) - Great explanations
- [What is a Vector Database?](https://www.youtube.com/watch?v=klTvEwg3oJ4) - Video intro

**Embeddings:**
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/) - Visual explanation

**Redis:**
- [Redis Vector Similarity Docs](https://redis.io/docs/stack/search/reference/vectors/)
- [RediSearch Quick Start](https://redis.io/docs/stack/search/quick_start/)

**OpenAI:**
- [OpenAI Cookbook](https://cookbook.openai.com/) - Many examples
- [Embeddings API Reference](https://platform.openai.com/docs/api-reference/embeddings)

---

## Troubleshooting

**Redis Connection Failed**
- Check your `redis_config.json` has correct host, port, and password
- Verify internet connection (Redis Cloud is remote)
- Test with: `redis-cli -h your-host -p your-port -a your-password ping`

**OpenAI API Errors**
- Verify API key is valid: check [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- Check billing: ensure you have credits
- Rate limits: free tier has limits, wait or upgrade

**Out of Memory (Redis)**
- Reduce dataset size: use fewer articles
- Free tier is 30MB - our 2,500 articles fit comfortably
- Consider paid tier for larger datasets

**Slow Search**
- Expected for FLAT index with many documents
- For >10K documents, switch to HNSW index
- HNSW is approximate but much faster

---

## Security Note

**Credentials are NOT in this repository.**

The file `redis_config.json` contains your actual credentials and is excluded from Git via `.gitignore`. To use this project:

1. Copy `redis_config.json.example` to `redis_config.json`
2. Add your own credentials
3. Never commit `redis_config.json` to version control

See [SECURITY_SETUP.md](SECURITY_SETUP.md) for details.

---

## License

This project is for educational purposes. The Wikipedia data is from the OpenAI Cookbook examples.

---

## Contributing

This is a learning project. Feel free to:
- Try different embedding models
- Add visualization of vectors (t-SNE, UMAP)
- Implement different distance metrics
- Compare FLAT vs HNSW performance
- Add your own dataset

---

## Acknowledgments

- **OpenAI** - For the embeddings API and cookbook examples
- **Redis** - For RediSearch and vector similarity features
- **Wikipedia** - For the article content
- **OpenAI Cookbook** - For pre-computed embeddings dataset

---

## Contact

For questions about this implementation:
- Check [FINAL_STATUS.md](FINAL_STATUS.md) for setup status
- See [SUBMISSION_GUIDE.md](SUBMISSION_GUIDE.md) for usage
- Review notebook comments for code explanations

**Happy Learning!** 🚀

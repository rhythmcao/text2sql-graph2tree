FROM huggingface/transformers-pytorch-gpu:latest
COPY requirements.txt /workspace
ENV NLTK_DATA=/root/nltk_data STANZA_RESOURCES_DIR=/root/stanza_resources EMBEDDINGS_ROOT=/root/.embeddings
RUN pip3 install torch==1.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html \
    && pip3 install dgl-cu101 -f https://data.dgl.ai/wheels/repo.html \
    && pip3 install -r requirements.txt \
    && python3 -c "import stanza; stanza.download('en')" \
    && python3 -c "from embeddings import GloveEmbedding, KazumaCharEmbedding; wemb, cemb = GloveEmbedding('common_crawl_48', d_emb=300), KazumaCharEmbedding()" \
    && python3 -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')" \
    && rm -rf $NLTK_DATA/corpora/stopwords.zip $STANZA_RESOURCES_DIR/en/default.zip $EMBEDDINGS_ROOT/glove/common_crawl_48.zip

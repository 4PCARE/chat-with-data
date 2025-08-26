import csv
import json
import os

import pandas as pd
from dotenv import load_dotenv
from pymilvus import (Collection, CollectionSchema, DataType, FieldSchema,
                      connections, utility)
from sentence_transformers import SentenceTransformer, models
import torch

load_dotenv()
neural_seek_url = os.getenv("NEURAL_SEEK_URL", None)
neural_seek_api_key = os.getenv("NEURAL_SEEK_API_KEY", None)


def create_field_schema(schema, EMBEDDINGS_DIMENSION, TEXT_MAX_LENGTH):
    """Create field schemas for the collection."""
    final_schema = [FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)]
    for key in schema:
        if schema[key] == DataType.FLOAT_VECTOR:
            curr_schema = FieldSchema(name=key, dtype=schema[key], dim=EMBEDDINGS_DIMENSION)
        elif schema[key] == DataType.VARCHAR:
            curr_schema = FieldSchema(name=key, dtype=schema[key], max_length=TEXT_MAX_LENGTH)
        elif schema[key] == DataType.FLOAT:
            curr_schema = FieldSchema(name=key, dtype=schema[key])
        else:
            pass
        final_schema.append(curr_schema)
    return final_schema

def create_collection_schema(fields, description="Search promotional events"):
    """Create a collection schema with the provided fields."""
    return CollectionSchema(fields=fields, description=description, enable_dynamic_field=True)

def initialize_collection(collection_name, schema, using='default'):
    """Initialize a collection with the given name and schema."""
    return Collection(name=collection_name, schema=schema, using=using)

def manage_collection(collection_name, schema, ID_MAX_LENGTH=50000, EMBEDDINGS_DIMENSION=1024, TEXT_MAX_LENGTH=50000):
    """Manage the creation or replacement of a collection."""
    print("Existing collections:", utility.list_collections())
    if collection_name in utility.list_collections():
        utility.drop_collection(collection_name)
        print("Dropped old collection")

    # Ensure collection is dropped
    existing_collections = utility.list_collections()
    print(f"Existing collections after drop operation: {existing_collections}")

    fields = create_field_schema(schema, EMBEDDINGS_DIMENSION, TEXT_MAX_LENGTH)
    print("Fields for new collection:", fields)

    schema = create_collection_schema(fields)
    collection = initialize_collection(collection_name, schema)
    print(f"Initialized new collection: {collection_name}")
    return collection

def get_model(model_name='airesearch/wangchanberta-base-att-spm-uncased', max_seq_length=768, condition=True):
    if condition:
        device = torch.device("cpu")
        word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),pooling_mode='cls') # We use a [CLS] token as representation
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model], cache_folder='./tmp', device=device)
    return model

def batch_insert(collection, embeds, text_to_encode_list, item_description_list, category_list, notification_date_list, batch_size):
    """
    Insert data into the collection in smaller batches.

    :param collection: The collection object to insert data into.
    :param embeds: Embeddings array.
    :param text_to_encode_list: List of text to encode.
    :param item_description_list: List of item descriptions.
    :param category_list: List of categories.
    :param batch_size: The maximum number of records to insert in one batch.
    """
    for i in range(0, len(embeds), batch_size):
        batch_embeds = embeds[i:i+batch_size]
        batch_text_to_encode = text_to_encode_list[i:i+batch_size]
        batch_item_description = item_description_list[i:i+batch_size]
        batch_category_list = category_list[i:i+batch_size]
        print(len(batch_embeds))
        collection.insert([batch_embeds, batch_text_to_encode, batch_item_description, batch_category_list, notification_date_list])
        
def get_retrival(model_embedder, question, collection, limit=3):
    query_encode = list(model_embedder.encode([question]))
    collection.load()

    documents = collection.search(
                                data=query_encode, 
                                anns_field="embeddings", 
                                param={"metric_type": "IP", "params": {}},       
    output_fields=['query', 'sql'], 
    limit=limit)

    df_retrival = pd.DataFrame()
    for hit in documents[0]:
        _df = pd.DataFrame(hit.entity.__dict__).T.iloc[-1:,:]
        df_retrival = pd.concat([df_retrival, _df], axis=0)
    df_retrival.reset_index(drop=True, inplace=True)

    return df_retrival

def get_retrival_v2(
    model_embedder, 
    question, 
    collection, 
    output_fields=['text_to_encode'], 
    limit=3, 
    score_threshold=0.7, 
    mode='limit'  # options: 'limit', 'score', 'hybrid'
):
    """
    mode = 'limit'   -> Return top N based on similarity score
    mode = 'score'   -> Return all hits with score >= threshold
    mode = 'hybrid'  -> Fetch more, filter by score, and limit final output
    """
    # Encode the query
    if model_embedder is not None:     
        query_encode = list(model_embedder.encode([question]))
    else:
        query_encode = get_embeddings([[question]])
    
    collection.load()

    # Fetch larger batch in hybrid mode to ensure good results after filtering
    search_limit = limit if mode == 'limit' else max(limit * 3, 10)

    documents = collection.search(
        data=query_encode, 
        anns_field="embeddings", 
        param={"metric_type": "COSINE", "params": {}}, 
        consistency_level="Strong",      
        output_fields=output_fields, 
        limit=search_limit
    )

    # Collect and optionally filter results
    results = []
    for hit in documents[0]:
        hit_data = hit.entity.__dict__.copy()
        hit_data['score'] = hit.score
        results.append(hit_data)

    df_retrieval = pd.DataFrame(results)

    if mode == 'limit':
        # Just return top N
        df_retrieval = df_retrieval.nlargest(limit, 'score').reset_index(drop=True)
    elif mode == 'score':
        # Return all results passing the threshold
        df_retrieval = df_retrieval[df_retrieval['score'] >= score_threshold].reset_index(drop=True)
    elif mode == 'hybrid':
        # Filter by score, then pick top N
        df_retrieval = df_retrieval[df_retrieval['score'] >= score_threshold]
        df_retrieval = df_retrieval.nlargest(limit, 'score').reset_index(drop=True)
    else:
        raise ValueError("Invalid mode. Choose from 'limit', 'score', or 'hybrid'.")

    return df_retrieval
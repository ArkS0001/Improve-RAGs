It sounds like your RAG system is “mixing” German and English documents because the embeddings from your PDF chunks aren’t being clearly separated by language. In other words, the vector space might be too “collided” across languages, so a German query ends up matching an English PDF. Here are a few approaches to address this issue:

• **Separate or tag your documents by language:**  
 – When indexing, detect the language of each PDF (or its chunks) and store that as metadata. Then, during retrieval, filter or re-rank results based on the language tag so that a German query only considers German documents.  
 
• **Use language-specific or fine-tuned models:**  
 – Although many Sentence Transformers are multilingual, some aren’t optimal for every language. Consider using an embedding model that’s been specifically fine-tuned for German (or for cross-lingual tasks). For example, models like “danielheinz/e5-base-sts-en-de” or “intfloat/multilingual-e5-large” have been reported to perform well for German.  
 
• **Apply semantic specialization or normalization tweaks:**  
 – Sometimes general-purpose models exhibit a “textual similarity bias” where superficial (lexical) similarities override deeper semantic differences. Techniques like semantic specialization (e.g. using approaches similar to EMU) or ensuring proper L2 normalization across embeddings can help reduce such collisions.  
 
• **Preprocessing and re-indexing:**  
 – Check that your text extraction and chunking from PDFs preserve language cues. Inconsistent preprocessing might cause documents to “blur together.” Reprocessing with a focus on language integrity can improve the separability in your embedding space.

By combining language metadata filtering with either a dedicated German embedding model or fine-tuning your existing model on German data, you should be able to mitigate these collisions and get more accurate retrieval from your German PDFs.

These steps can help ensure that the retrieval step in your RAG pipeline is more language-aware and that the correct documents are returned for a German query.

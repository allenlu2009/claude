## FEATURE:

Compute the perplexity of common public models using commom dataset

## EXAMPLES:

It compute the perplexity of phi3 and phi4 models using wikitext2 dataset

## DOCUMENTATION:

[List out any documentation (web pages, sources for an MCP server like Crawl4AI RAG, etc.) that will need to be referenced during development]

## OTHER CONSIDERATIONS:

I am using desktop GPU RTX 3060, there is 12GB memory.  Make sure to use flash attention whenever it applies.  If evaluate more than 1 models, do it sequentially and clear the memory before loading a new model.
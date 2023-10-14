def get_all_sentences(ds, lang):
    for item in ds:
        yield item["translation"][lang]

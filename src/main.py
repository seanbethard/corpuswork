from komatsu.nl.utils.Utils import prepare_openwebtext_archive
from komatsu.nl.utils.Utils import KomatsuPostgres
import os

if __name__ == "__main__":
    archives = []
    for archive in os.listdir('openwebtext'):
        if archive.endswith('_data'):
            archives.append(os.path.join('openwebtext', archive))

    sentences = prepare_openwebtext_archive(archives)
    db = KomatsuPostgres()
    db.copy_sentences(db)

from __future__ import absolute_import, division, print_function

import sys
import random
import pickle
import logging
import logging.handlers
import numpy as np
import csv
import scipy.sparse as sp
import torch


# Dataset names.
from sklearn.feature_extraction.text import TfidfTransformer

ML1M = 'ml1m'
LASTFM = 'lastfm'
BEAUTY = 'beauty'
CD = 'cd'
CELL = 'cell'
CLOTH = 'cloth'
AMAZON_DATASETS = [CD, CELL, CLOTH, BEAUTY]
# Dataset directories.
DATASET_DIR = {
    ML1M: '../../datasets/ml1m',
    LASTFM: '../../datasets/lastfm',
    BEAUTY: '../../datasets/Amazon_Beauty',
    CD: '../../datasets/Amazon_CDs',
    CELL: '../../datasets/Amazon_Cellphones',
    CLOTH: '../../datasets/Amazon_Clothing',
}

# Model result directories.
TMP_DIR = {
    ML1M: 'tmp/ml1m',
    LASTFM: 'tmp/lastfm',
    BEAUTY: 'tmp/Amazon_Beauty',
    CD: 'tmp/Amazon_CDs',
    CELL: 'tmp/Amazon_Cellphones',
    CLOTH: 'tmp/Amazon_Clothing',
}

# Label files.
LABELS = {
    ML1M: (TMP_DIR[ML1M] + '/train_label.pkl', TMP_DIR[ML1M] + '/test_label.pkl'),
    LASTFM: (TMP_DIR[LASTFM] + '/train_label.pkl', TMP_DIR[LASTFM] + '/test_label.pkl'),
    BEAUTY: (TMP_DIR[BEAUTY] + '/train_label.pkl', TMP_DIR[BEAUTY] + '/test_label.pkl'),
    CD: (TMP_DIR[CD] + '/train_label.pkl', TMP_DIR[CD] + '/test_label.pkl'),
    CELL: (TMP_DIR[CELL] + '/train_label.pkl', TMP_DIR[CELL] + '/test_label.pkl'),
    CLOTH: (TMP_DIR[CLOTH] + '/train_label.pkl', TMP_DIR[CLOTH] + '/test_label.pkl'),
}

#ML1M ENTITIES
MOVIE = 'movie'
ACTOR = 'actor'
DIRECTOR = 'director'
PRODUCTION_COMPANY = 'production_company'
EDITOR = 'editor'
WRITTER = 'writter'
CINEMATOGRAPHER = 'cinematographer'
COMPOSER = 'composer'

#LASTFM ENTITIES
SONG = 'song'
ARTIST = 'artist'
ENGINEER = 'engineer'
PRODUCER = 'producer'
RELATED_SONG = 'related_song'

#AMAZON ENTITIES
RELATED_PRODUCT = 'related_product'
BRAND = 'brand'
WORD = 'word'
#SHARED ENTITIES
USER = 'user'
CATEGORY = 'category'
PRODUCT = 'product'



# Entities
ENTITY_LIST = {
    ML1M: [
        USER,
        MOVIE,
        ACTOR,
        DIRECTOR,
        PRODUCER,
        PRODUCTION_COMPANY,
        CATEGORY,
        EDITOR,
        WRITTER,
        CINEMATOGRAPHER,
    ],
    LASTFM: [
        USER,
        SONG,
        ARTIST,
        ENGINEER,
        PRODUCER
    ],
}

#ML1M RELATIONS
WATCHED = 'watched'
DIRECTED_BY = 'directed_by'
PRODUCED_BY_COMPANY = 'produced_by_company'
STARRING = 'starring'
EDITED_BY = 'edited_by'
WROTE_BY = 'wrote_by'
CINEMATOGRAPHY = 'cinematography'
COMPOSED_BY = 'composed_by'

#LASTFM RELATIONS
LISTENED = 'listened'
MIXED_BY = 'mixed_by'
FEATURED_BY = 'featured_by'
SANG_BY = 'sang_by'
ALTERNATIVE_VERSION_OF = 'alternative_version_of'
ORIGINAL_VERSION_OF = "original_version_of"
RELATED_TO = 'related_to'

#AMAZON RELATIONS
PURCHASE = 'purchase'
PRODUCED_BY = 'produced_by'
ALSO_BOUGHT = 'also_bought'
ALSO_VIEWED = 'also_viewed'
BOUGHT_TOGETHER = 'bought_together'
MENTION = 'mentions'
DESCRIBED_AS = 'described_as'
#SHARED RELATIONS
BELONG_TO = 'belong_to'
PRODUCED_BY_PRODUCER = 'produced_by_producer'
SELF_LOOP = 'self_loop'

RELATION_LIST = {
    ML1M: {
        0: "http://dbpedia.org/ontology/cinematography",
        1: "http://dbpedia.org/property/productionCompanies",
        2: "http://dbpedia.org/property/composer",
        3: "http://purl.org/dc/terms/subject",
        4: "http://dbpedia.org/ontology/openingFilm",
        5: "http://www.w3.org/2000/01/rdf-schema",
        6: "http://dbpedia.org/property/story",
        7: "http://dbpedia.org/ontology/series",
        8: "http://www.w3.org/1999/02/22-rdf-syntax-ns",
        9: "http://dbpedia.org/ontology/basedOn",
        10: "http://dbpedia.org/ontology/starring",
        11: "http://dbpedia.org/ontology/country",
        12: "http://dbpedia.org/ontology/wikiPageWikiLink",
        13: "http://purl.org/linguistics/gold/hypernym",
        14: "http://dbpedia.org/ontology/editing",
        15: "http://dbpedia.org/property/producers",
        16: "http://dbpedia.org/property/allWriting",
        17: "http://dbpedia.org/property/notableWork",
        18: "http://dbpedia.org/ontology/director",
        19: "http://dbpedia.org/ontology/award",
    },
    LASTFM: {
        0: "http://rdf.freebase.com/ns/common.topic.notable_types",
        1: "http://rdf.freebase.com/ns/music.recording.releases",
        2: "http://rdf.freebase.com/ns/music.recording.artist",
        3: "http://rdf.freebase.com/ns/music.recording.engineer",
        4: "http://rdf.freebase.com/ns/music.recording.producer",
        5: "http://rdf.freebase.com/ns/music.recording.canonical_version",
        6: "http://rdf.freebase.com/ns/music.recording.song",
        7: "http://rdf.freebase.com/ns/music.single.versions",
        8: "http://rdf.freebase.com/ns/music.recording.featured_artists",
    },
}

KG_RELATION = {
    ML1M: {
        USER: {
            WATCHED: MOVIE,
        },
        ACTOR: {
            STARRING: MOVIE,
        },
        DIRECTOR: {
            DIRECTED_BY: MOVIE,
        },
        MOVIE: {
            WATCHED: USER,
            PRODUCED_BY_COMPANY: PRODUCTION_COMPANY,
            PRODUCED_BY_PRODUCER: PRODUCER,
            EDITED_BY: EDITOR,
            WROTE_BY: WRITTER,
            CINEMATOGRAPHY: CINEMATOGRAPHER,
            BELONG_TO: CATEGORY,
            DIRECTED_BY: DIRECTOR,
            STARRING: ACTOR,
            COMPOSED_BY: COMPOSER,
        },
        PRODUCTION_COMPANY: {
            PRODUCED_BY_COMPANY: MOVIE,
        },
        COMPOSER: {
            COMPOSED_BY: MOVIE,
        },
        PRODUCER: {
            PRODUCED_BY_PRODUCER: MOVIE,
        },
        WRITTER: {
            WROTE_BY: MOVIE,
        },
        EDITOR: {
            EDITED_BY: MOVIE,
        },
        CATEGORY: {
            BELONG_TO: MOVIE,
        },
        CINEMATOGRAPHER: {
            CINEMATOGRAPHY: MOVIE,
        },
    },
    LASTFM: {
        USER: {
            LISTENED: SONG,
        },
        ARTIST: {
            SANG_BY: SONG,
            FEATURED_BY: SONG,
        },
        ENGINEER: {
            MIXED_BY: SONG,
        },
        SONG: {
            LISTENED: USER,
            PRODUCED_BY_PRODUCER: PRODUCER,
            SANG_BY: ARTIST,
            FEATURED_BY: ARTIST,
            MIXED_BY: ENGINEER,
            BELONG_TO: CATEGORY,
            RELATED_TO: RELATED_SONG,
            ORIGINAL_VERSION_OF: RELATED_SONG,
            ALTERNATIVE_VERSION_OF: RELATED_SONG,
        },
        PRODUCER: {
            PRODUCED_BY_PRODUCER: SONG,
        },
        CATEGORY: {
            BELONG_TO: SONG,
        },
        RELATED_SONG: {
            RELATED_TO: SONG,
            ORIGINAL_VERSION_OF: SONG,
            ALTERNATIVE_VERSION_OF: SONG,
        }
    },
    BEAUTY: {
        USER: {
            PURCHASE: PRODUCT,
            MENTION: WORD,
        },
        WORD: {
            MENTION: USER,
            DESCRIBED_AS: PRODUCT,
        },
        PRODUCT: {
            PURCHASE: USER,
            DESCRIBED_AS: PRODUCT,
            PRODUCED_BY: BRAND,
            BELONG_TO: CATEGORY,
            ALSO_BOUGHT: RELATED_PRODUCT,
            ALSO_VIEWED: RELATED_PRODUCT,
            BOUGHT_TOGETHER: RELATED_PRODUCT,
        },
        BRAND: {
            PRODUCED_BY: PRODUCT,
        },
        CATEGORY: {
            BELONG_TO: PRODUCT,
        },
        RELATED_PRODUCT: {
            ALSO_BOUGHT: PRODUCT,
            ALSO_VIEWED: PRODUCT,
            BOUGHT_TOGETHER: PRODUCT,
        }
    },
    CD: {
        USER: {
            PURCHASE: PRODUCT,
        },
        PRODUCT: {
            PURCHASE: USER,
            DESCRIBED_AS: PRODUCT,
            PRODUCED_BY: BRAND,
            BELONG_TO: CATEGORY,
            ALSO_BOUGHT: RELATED_PRODUCT,
            ALSO_VIEWED: RELATED_PRODUCT,
            BOUGHT_TOGETHER: RELATED_PRODUCT,
        },
        BRAND: {
            PRODUCED_BY: PRODUCT,
        },
        CATEGORY: {
            BELONG_TO: PRODUCT,
        },
        RELATED_PRODUCT: {
            ALSO_BOUGHT: PRODUCT,
            ALSO_VIEWED: PRODUCT,
            BOUGHT_TOGETHER: PRODUCT,
        }
    },
    CELL: {
        USER: {
            PURCHASE: PRODUCT,
            MENTION: WORD,
        },
        WORD: {
            MENTION: USER,
            DESCRIBED_AS: PRODUCT,
        },
        PRODUCT: {
            PURCHASE: USER,
            DESCRIBED_AS: WORD,
            PRODUCED_BY: BRAND,
            BELONG_TO: CATEGORY,
            ALSO_BOUGHT: RELATED_PRODUCT,
            ALSO_VIEWED: RELATED_PRODUCT,
            BOUGHT_TOGETHER: RELATED_PRODUCT,
        },
        BRAND: {
            PRODUCED_BY: PRODUCT,
        },
        CATEGORY: {
            BELONG_TO: PRODUCT,
        },
        RELATED_PRODUCT: {
            ALSO_BOUGHT: PRODUCT,
            ALSO_VIEWED: PRODUCT,
            BOUGHT_TOGETHER: PRODUCT,
        }
    },
    CLOTH: {
        USER: {
            PURCHASE: PRODUCT,
        },
        PRODUCT: {
            PURCHASE: USER,
            PRODUCED_BY: BRAND,
            BELONG_TO: CATEGORY,
            ALSO_BOUGHT: RELATED_PRODUCT,
            ALSO_VIEWED: RELATED_PRODUCT,
            BOUGHT_TOGETHER: RELATED_PRODUCT,
        },
        BRAND: {
            PRODUCED_BY: PRODUCT,
        },
        CATEGORY: {
            BELONG_TO: PRODUCT,
        },
        RELATED_PRODUCT: {
            ALSO_BOUGHT: PRODUCT,
            ALSO_VIEWED: PRODUCT,
            BOUGHT_TOGETHER: PRODUCT,
        }
    },
}


#0 is reserved to the main relation, 1 to mention
PATH_PATTERN = {
    ML1M: {
        0: ((None, USER), (WATCHED, MOVIE), (WATCHED, USER), (WATCHED, MOVIE)),
        2: ((None, USER), (WATCHED, MOVIE), (CINEMATOGRAPHY, CINEMATOGRAPHER), (CINEMATOGRAPHY, MOVIE)),
        3: ((None, USER), (WATCHED, MOVIE), (PRODUCED_BY_COMPANY, PRODUCTION_COMPANY), (PRODUCED_BY_COMPANY, MOVIE)),
        4: ((None, USER), (WATCHED, MOVIE), (COMPOSED_BY, COMPOSER), (COMPOSED_BY, MOVIE)),
        5: ((None, USER), (WATCHED, MOVIE), (BELONG_TO, CATEGORY), (BELONG_TO, MOVIE)),
        6: ((None, USER), (WATCHED, MOVIE), (BELONG_TO, CATEGORY), (BELONG_TO, MOVIE)),
        7: ((None, USER), (WATCHED, MOVIE), (STARRING, ACTOR), (STARRING, MOVIE)),
        8: ((None, USER), (WATCHED, MOVIE), (EDITED_BY, EDITOR), (EDITED_BY, MOVIE)),
        9: ((None, USER), (WATCHED, MOVIE), (PRODUCED_BY_PRODUCER, PRODUCER), (PRODUCED_BY_PRODUCER, MOVIE)),
        10: ((None, USER), (WATCHED, MOVIE), (WROTE_BY, WRITTER), (WROTE_BY, MOVIE)),
        11: ((None, USER), (WATCHED, MOVIE), (DIRECTED_BY, DIRECTOR), (DIRECTED_BY, MOVIE)),
    },
    LASTFM: {
        0: ((None, USER), (LISTENED, SONG), (LISTENED, USER), (LISTENED, SONG)),
        2: ((None, USER), (LISTENED, SONG), (BELONG_TO, CATEGORY), (BELONG_TO, SONG)),
        3: ((None, USER), (LISTENED, SONG), (RELATED_TO, RELATED_SONG), (RELATED_TO, SONG)),
        4: ((None, USER), (LISTENED, SONG), (SANG_BY, ARTIST), (SANG_BY, SONG)),
        5: ((None, USER), (LISTENED, SONG), (MIXED_BY, ENGINEER), (MIXED_BY, SONG)),
        6: ((None, USER), (LISTENED, SONG), (PRODUCED_BY_PRODUCER, PRODUCER), (PRODUCED_BY_PRODUCER, SONG)),
        7: ((None, USER), (LISTENED, SONG), (ORIGINAL_VERSION_OF, RELATED_SONG), (ORIGINAL_VERSION_OF, SONG)),
        8: ((None, USER), (LISTENED, SONG), (RELATED_TO, RELATED_SONG), (RELATED_TO, SONG)),
        9: ((None, USER), (LISTENED, SONG), (ALTERNATIVE_VERSION_OF, RELATED_SONG), (ALTERNATIVE_VERSION_OF, SONG)),
        10: ((None, USER), (LISTENED, SONG), (FEATURED_BY, ARTIST), (FEATURED_BY, SONG)),
    },
    BEAUTY: {
        0: ((None, USER), (PURCHASE, PRODUCT), (PURCHASE, USER), (PURCHASE, PRODUCT)),
        1: ((None, USER), (MENTION, WORD), (DESCRIBED_AS, PRODUCT)),  # Len 3 need a special treatment in kg_env
        2: ((None, USER), (PURCHASE, PRODUCT), (PRODUCED_BY, BRAND), (PRODUCED_BY, PRODUCT)),
        3: ((None, USER), (PURCHASE, PRODUCT), (BELONG_TO, CATEGORY), (BELONG_TO, PRODUCT)),
        4: ((None, USER), (PURCHASE, PRODUCT), (ALSO_BOUGHT, RELATED_PRODUCT), (ALSO_BOUGHT, PRODUCT)),
        5: ((None, USER), (PURCHASE, PRODUCT), (ALSO_VIEWED, RELATED_PRODUCT), (ALSO_VIEWED, PRODUCT)),
        6: ((None, USER), (PURCHASE, PRODUCT), (BOUGHT_TOGETHER, RELATED_PRODUCT), (BOUGHT_TOGETHER, PRODUCT)),
        7: ((None, USER), (PURCHASE, PRODUCT), (DESCRIBED_AS, WORD), (DESCRIBED_AS, PRODUCT)),
        8: ((None, USER), (MENTION, WORD), (MENTION, USER), (PURCHASE, PRODUCT)),
    },
    CD: {
        0: ((None, USER), (PURCHASE, PRODUCT), (PURCHASE, USER), (PURCHASE, PRODUCT)),
        1: ((None, USER), (MENTION, WORD), (DESCRIBED_AS, PRODUCT)),#Len 3 need a special treatment in kg_env
        2: ((None, USER), (PURCHASE, PRODUCT), (PRODUCED_BY, BRAND), (PRODUCED_BY, PRODUCT)),
        3: ((None, USER), (PURCHASE, PRODUCT), (BELONG_TO, CATEGORY), (BELONG_TO, PRODUCT)),
        4: ((None, USER), (PURCHASE, PRODUCT), (ALSO_BOUGHT, RELATED_PRODUCT), (ALSO_BOUGHT, PRODUCT)),
        5: ((None, USER), (PURCHASE, PRODUCT), (ALSO_VIEWED, RELATED_PRODUCT), (ALSO_VIEWED, PRODUCT)),
        6: ((None, USER), (PURCHASE, PRODUCT), (BOUGHT_TOGETHER, RELATED_PRODUCT), (BOUGHT_TOGETHER, PRODUCT)),
        7: ((None, USER), (PURCHASE, PRODUCT), (DESCRIBED_AS, WORD), (DESCRIBED_AS, PRODUCT)),
        8: ((None, USER), (MENTION, WORD), (MENTION, USER), (PURCHASE, PRODUCT)),
    },
    CELL: {
        0: ((None, USER), (PURCHASE, PRODUCT), (PURCHASE, USER), (PURCHASE, PRODUCT)),
        1: ((None, USER), (MENTION, WORD), (DESCRIBED_AS, PRODUCT)),#Len 3 need a special treatment in kg_env
        2: ((None, USER), (PURCHASE, PRODUCT), (PRODUCED_BY, BRAND), (PRODUCED_BY, PRODUCT)),
        3: ((None, USER), (PURCHASE, PRODUCT), (BELONG_TO, CATEGORY), (BELONG_TO, PRODUCT)),
        4: ((None, USER), (PURCHASE, PRODUCT), (ALSO_BOUGHT, RELATED_PRODUCT), (ALSO_BOUGHT, PRODUCT)),
        5: ((None, USER), (PURCHASE, PRODUCT), (ALSO_VIEWED, RELATED_PRODUCT), (ALSO_VIEWED, PRODUCT)),
        6: ((None, USER), (PURCHASE, PRODUCT), (BOUGHT_TOGETHER, RELATED_PRODUCT), (BOUGHT_TOGETHER, PRODUCT)),
        7: ((None, USER), (PURCHASE, PRODUCT), (DESCRIBED_AS, WORD), (DESCRIBED_AS, PRODUCT)),
        8: ((None, USER), (MENTION, WORD), (MENTION, USER), (PURCHASE, PRODUCT)),
    },
    CLOTH: {
        0: ((None, USER), (PURCHASE, PRODUCT), (PURCHASE, USER), (PURCHASE, PRODUCT)),
        #1: ((None, USER), (MENTION, WORD), (DESCRIBED_AS, PRODUCT)),  # Len 3 need a special treatment in kg_env
        2: ((None, USER), (PURCHASE, PRODUCT), (PRODUCED_BY, BRAND), (PRODUCED_BY, PRODUCT)),
        3: ((None, USER), (PURCHASE, PRODUCT), (BELONG_TO, CATEGORY), (BELONG_TO, PRODUCT)),
        4: ((None, USER), (PURCHASE, PRODUCT), (ALSO_BOUGHT, RELATED_PRODUCT), (ALSO_BOUGHT, PRODUCT)),
        5: ((None, USER), (PURCHASE, PRODUCT), (ALSO_VIEWED, RELATED_PRODUCT), (ALSO_VIEWED, PRODUCT)),
        6: ((None, USER), (PURCHASE, PRODUCT), (BOUGHT_TOGETHER, RELATED_PRODUCT), (BOUGHT_TOGETHER, PRODUCT)),
        7: ((None, USER), (PURCHASE, PRODUCT), (DESCRIBED_AS, WORD), (DESCRIBED_AS, PRODUCT)),
        8: ((None, USER), (MENTION, WORD), (MENTION, USER), (PURCHASE, PRODUCT)),
    },
}

RELATION_NAMES = {
    ML1M: {0: WATCHED, 2: CINEMATOGRAPHY, 3: PRODUCED_BY_COMPANY, 4: COMPOSED_BY, 5: BELONG_TO,  6: BELONG_TO, 7: STARRING, 8: EDITED_BY, 9: PRODUCED_BY_PRODUCER, 10: WROTE_BY, 11: DIRECTED_BY},
    LASTFM: {0: LISTENED, 2: BELONG_TO, 3: RELATED_TO, 4: SANG_BY, 5: MIXED_BY, 6: PRODUCED_BY_PRODUCER, 7: ORIGINAL_VERSION_OF, 8: RELATED_TO, 9: ALTERNATIVE_VERSION_OF, 10: FEATURED_BY},
    BEAUTY: {0: PURCHASE, 1: DESCRIBED_AS, 2: PRODUCED_BY, 3: BELONG_TO, 4: ALSO_BOUGHT, 5: ALSO_VIEWED, 6: MENTION, 7: DESCRIBED_AS, 8: PURCHASE},
    #CD: {1: PURCHASE, 2: PRODUCED_BY, 3: BELONG_TO, 4: ALSO_BOUGHT, 5: ALSO_VIEWED, 6: BOUGHT_TOGETHER},
    CELL: {0: PURCHASE, 1: DESCRIBED_AS, 2: PRODUCED_BY, 3: BELONG_TO, 4: ALSO_BOUGHT, 5: ALSO_VIEWED, 6: MENTION, 7: DESCRIBED_AS, 8: PURCHASE},
    CLOTH: {0: PURCHASE, 1: DESCRIBED_AS, 2: PRODUCED_BY, 3: BELONG_TO, 4: ALSO_BOUGHT, 5: ALSO_VIEWED, 6: MENTION, 7: DESCRIBED_AS, 8: PURCHASE},
}

#Used to read from KG completion the triple (e_h, e_t, r) the number represent the relation number used in the kg, the values is the entity tail
TAIL_ENTITY_NAMES = {
    ML1M: {0: CINEMATOGRAPHER, 1: PRODUCTION_COMPANY, 2: COMPOSER, 3: CATEGORY, 8: CATEGORY, 10: ACTOR, 14: EDITOR, 15: PRODUCER, 16: WRITTER, 18: DIRECTOR},
    LASTFM: {0: CATEGORY, 1: RELATED_SONG, 2: ARTIST, 3: ENGINEER, 4: PRODUCER, 5: RELATED_SONG, 6: RELATED_SONG, 7: RELATED_SONG, 8: ARTIST},
}

RELATION_TO_ENTITY = {
    ML1M: {WATCHED: USER, CINEMATOGRAPHY: CINEMATOGRAPHER, PRODUCED_BY_COMPANY: PRODUCTION_COMPANY, COMPOSED_BY: COMPOSER, BELONG_TO: CATEGORY, STARRING: ACTOR, EDITED_BY: EDITOR, PRODUCED_BY_PRODUCER: PRODUCER, WROTE_BY: WRITTER, DIRECTED_BY: DIRECTOR},
    LASTFM: {LISTENED: USER, BELONG_TO: CATEGORY, RELATED_TO: RELATED_SONG, SANG_BY: ARTIST, MIXED_BY: ENGINEER, PRODUCED_BY_PRODUCER: PRODUCER, ORIGINAL_VERSION_OF: RELATED_SONG, RELATED_TO: RELATED_SONG, ALTERNATIVE_VERSION_OF: RELATED_SONG, FEATURED_BY: ARTIST},
    CELL: {PURCHASE: USER, MENTION: WORD, DESCRIBED_AS: WORD, PRODUCED_BY: BRAND, BELONG_TO: CATEGORY, ALSO_BOUGHT: RELATED_PRODUCT, ALSO_VIEWED: RELATED_PRODUCT, BOUGHT_TOGETHER: RELATED_PRODUCT}
}

MAIN_PRODUCT_INTERACTION = {
    ML1M: (MOVIE, WATCHED),
    LASTFM: (SONG, LISTENED),
    BEAUTY: (PRODUCT, PURCHASE),
    CD: (PRODUCT, PURCHASE),
    CELL: (PRODUCT, PURCHASE),
    CLOTH: (PRODUCT, PURCHASE),
}
def get_relations_names(dataset_name):
    return RELATION_NAMES[dataset_name].values()

def get_user2gender(dataset_name):
    file = open(DATASET_DIR[dataset_name] + "/mappings/uid2gender.txt", 'r')
    csv_reader = csv.reader(file, delimiter='\n')
    uid_gender = {}
    gender2name = {-1: "Overall", 0: "Male", 1: "Female"}
    uid_mapping = get_uid_to_kgid_mapping(dataset_name) #1->0
    for row in csv_reader:
        row = row[0].strip().split('\t')
        if dataset_name == "ml1m":
            uid_gender[uid_mapping[int(row[0])]] = 0 if row[1] == 'M' else 1
        else:
            uid_gender[uid_mapping[int(row[0])]] = 0 if row[1] == 'm' else 1
    return uid_gender, gender2name

# Returns a dict that maps the user uid with his age
def get_user2age():
    file = open("../../datasets/ml1m/mappings/uid2age.txt", 'r')
    csv_reader = csv.reader(file, delimiter='\n')
    uid_age = {}
    age2name = {1:  "Under 18", 18:  "18-24", 25:  "25-34", 35:  "35-44", 45:  "45-49", 50:  "50-55", 56:  "56+"}
    for row in csv_reader:
        row = row[0].strip().split('\t')
        uid_age[int(row[0])] = int(row[1])
    return uid_age, age2name

# Returns a dict that maps the user uid with his occupation
def get_user2occupation():
    file = open("../../datasets/ml1m/mappings/uid2occupation.txt", 'r')
    csv_reader = csv.reader(file, delimiter='\n')
    uid_occ = {}
    occ2name = {0:  "other", 1:  "academic/educator",  2:  "artist",  3:  "clerical/admin",  4:  "college/grad student",  5:  "customer service",  6:  "doctor/health care",  7:  "executive/managerial",  8:  "farmer",  9:  "homemaker", 10:  "K-12 student", 11:  "lawyer", 12:  "programmer", 13:  "retired", 14:  "sales/marketing", 15:  "scientist", 16:  "self-employed", 17:  "technician/engineer", 18:  "tradesman/craftsman", 19:  "unemployed", 20:  "writer"}
    for row in csv_reader:
        row = row[0].strip().split('\t')
        uid_occ[int(row[0])] = int(row[1])
    return uid_occ, occ2name



def get_entities(dataset_name):
    return list(KG_RELATION[dataset_name].keys())

def get_entities_without_user(dataset_name):
    ans = list(KG_RELATION[dataset_name].keys())
    ans.remove('user')
    return ans


def get_knowledge_derived_relations(dataset_name):
    main_entity, main_relation = MAIN_PRODUCT_INTERACTION[dataset_name]
    ans = list(KG_RELATION[dataset_name][main_entity].keys())
    ans.remove(main_relation)
    if dataset_name in AMAZON_DATASETS: ans.remove("described_as")
    return ans

def get_tail_entity_name(dataset, relationship_id):
    return TAIL_ENTITY_NAMES[dataset][relationship_id]

def get_dataset_relations(dataset_name, entity_head):
    return list(KG_RELATION[dataset_name][entity_head].keys())


def get_entity_tail(dataset_name, relation):
    entity_head, _ = MAIN_PRODUCT_INTERACTION[dataset_name]
    return KG_RELATION[dataset_name][entity_head][relation]


def compute_tfidf_fast(vocab, docs):
    """Compute TFIDF scores for all vocabs.

    Args:
        docs: list of list of integers, e.g. [[0,0,1], [1,2,0,1]]

    Returns:
        sp.csr_matrix, [num_docs, num_vocab]
    """
    # (1) Compute term frequency in each doc.
    data, indices, indptr = [], [], [0]
    for d in docs:
        term_count = {}
        for term_idx in d:
            if term_idx not in term_count:
                term_count[term_idx] = 1
            else:
                term_count[term_idx] += 1
        indices.extend(term_count.keys())
        data.extend(term_count.values())
        indptr.append(len(indices))
    tf = sp.csr_matrix((data, indices, indptr), dtype=int, shape=(len(docs), len(vocab)))

    # (2) Compute normalized tfidf for each term/doc.
    transformer = TfidfTransformer(smooth_idf=True)
    tfidf = transformer.fit_transform(tf)
    return tfidf


def get_logger(logname):
    logger = logging.getLogger(logname)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(levelname)s]  %(message)s')
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    fh = logging.handlers.RotatingFileHandler(logname, mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_dataset(dataset, dataset_obj):
    dataset_file = TMP_DIR[dataset] + '/dataset.pkl'
    with open(dataset_file, 'wb') as f:
        pickle.dump(dataset_obj, f)


def load_dataset(dataset):
    dataset_file = TMP_DIR[dataset] + '/dataset.pkl'
    dataset_obj = pickle.load(open(dataset_file, 'rb'))
    return dataset_obj


def save_labels(dataset, labels, mode='train'):
    if mode == 'train':
        label_file = LABELS[dataset][0]
    elif mode == 'test':
        label_file = LABELS[dataset][1]
    else:
        raise Exception('mode should be one of {train, test}.')
    with open(label_file, 'wb') as f:
        pickle.dump(labels, f)


def load_labels(dataset, mode='train'):
    if mode == 'train':
        label_file = LABELS[dataset][0]
        # CHANGED
    elif mode == 'test':
        label_file = LABELS[dataset][1]
    else:
        raise Exception('mode should be one of {train, test}.')
    user_products = pickle.load(open(label_file, 'rb'))
    return user_products


def save_embed(dataset, embed):
    embed_file = '{}/transe_embed.pkl'.format(TMP_DIR[dataset])
    pickle.dump(embed, open(embed_file, 'wb'))


def load_embed(dataset):
    embed_file = '{}/transe_embed.pkl'.format(TMP_DIR[dataset])
    print('Load embedding:', embed_file)
    embed = pickle.load(open(embed_file, 'rb'))
    return embed

#Receive paths in form (score, prob, [path]) return the last relationship
def get_path_pattern(path):
    return path[-1][-1][0]
    '''
    0: ((None, USER), (WATCHED, MOVIE), (CINEMATOGRAPHY, CINEMATOGRAPHER), (CINEMATOGRAPHY, MOVIE)),
    1: ((None, USER), (WATCHED, MOVIE), (PRODUCED_BY_COMPANY, PRODUCTION_COMPANY), (PRODUCED_BY_COMPANY, MOVIE)),
    8: ((None, USER), (WATCHED, MOVIE), (BELONG_TO, CATEGORY), (BELONG_TO, MOVIE)),
    10: ((None, USER), (WATCHED, MOVIE), (STARRING, ACTOR), (STARRING, MOVIE)),
    14: ((None, USER), (WATCHED, MOVIE), (EDITED_BY, EDITOR), (EDITED_BY, MOVIE)),
    15: ((None, USER), (WATCHED, MOVIE), (PRODUCED_BY_PRODUCER, PRODUCER), (PRODUCED_BY_PRODUCER, MOVIE)),
    16: ((None, USER), (WATCHED, MOVIE), (WROTE_BY, WRITTER), (WROTE_BY, MOVIE)),
    18: ((None, USER), (WATCHED, MOVIE), (DIRECTED_BY, DIRECTOR), (DIRECTED_BY, MOVIE)),
    '''


def get_pid_to_kgid_mapping(dataset_name):
    file = open(DATASET_DIR[dataset_name] + "/mappings/product_mappings.txt", "r")
    reader = csv.reader(file, delimiter='\t')
    dataset_pid2kg_pid = {}
    next(reader, None)
    for row in reader:
        if dataset_name == "ml1m" or dataset_name == "lastfm":
            dataset_pid2kg_pid[int(row[1])] = int(row[0])
        else:
            dataset_pid2kg_pid[row[1]] = int(row[0])
    file.close()
    return dataset_pid2kg_pid

def get_uid_to_kgid_mapping(dataset_name):
    dataset_uid2kg_uid = {}
    with open(DATASET_DIR[dataset_name] + "/mappings/user_mappings.txt", 'r') as file:
        reader = csv.reader(file, delimiter="\t")
        next(reader, None)
        for row in reader:
            if dataset_name == "ml1m" or dataset_name == "lastfm":
                uid_review = int(row[1])
            else:
                uid_review = row[1]
            uid_kg = int(row[0])
            dataset_uid2kg_uid[uid_review] = uid_kg
    return dataset_uid2kg_uid

def save_kg(dataset, kg):
    kg_file = TMP_DIR[dataset] + '/kg.pkl'
    pickle.dump(kg, open(kg_file, 'wb'))


def load_kg(dataset):
    kg_file = TMP_DIR[dataset] + '/kg.pkl'
    # CHANGED
    kg = pickle.load(open(kg_file, 'rb'))
    return kg

def shuffle(arr):
    for i in range(len(arr) - 1, 0, -1):
        # Pick a random index from 0 to i
        j = random.randint(0, i + 1)

        # Swap arr[i] with the element at random index
        arr[i], arr[j] = arr[j], arr[i]
    return arr
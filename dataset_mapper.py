import argparse
import json
import os
import re
import shutil
import string
from os.path import exists

import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
import numpy as np
from easydict import EasyDict as edict
from models.PGPR.utils import get_tail_entity_name, RELATION_NAMES
from myutils import *
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

#Generate the mapping from the KG Completation of KGAT completion to a PGPR readable dataset
class LastFmDatasetMapper(object):
    def __init__(self, args):
        self.args = args
        self.generate_user_attributes_mappings()
        self.generate_kg_entities()
        self.generate_kg_relations()
        self.generate_train_test_split()

    def generate_train_test_split(self):
        dataset_name = self.args.dataset
        uid_review_tuples = {}
        dataset_size = 0
        valid_movies = get_valid_products(dataset_name)
        print("Loading reviews...")
        removed_reviews = 0
        with open(DATASET_DIR[dataset_name] + "/ratings.dat", 'r', encoding='latin-1') as reviews_file:
            reader = csv.reader(reviews_file, delimiter=',')
            next(reader, None)
            for row in reader:
                uid = int(row[0])
                if uid not in uid_review_tuples:
                    uid_review_tuples[uid] = []
                if int(row[3]) not in valid_movies:
                    removed_reviews += 1
                    continue
                uid_review_tuples[uid].append((row[0], row[3], row[4]))
                dataset_size += 1
        print("Discarted {} reviews for product not in the kg completion".format(removed_reviews))
        reviews_file.close()
        train_size = 0.8
        print("Performing split {}/{}...".format(train_size * 100, 100 - train_size * 100))
        for uid, reviews in uid_review_tuples.items():
            reviews.sort(key=lambda x: int(x[-1]))  # sorting from recent to older

        train = []
        test = []
        discarted_users = 0
        total_users = 0
        th = 5
        test_sizes = []
        for uid, reviews in uid_review_tuples.items():  # python dict are sorted, 1...nuser
            total_users += 1
            if len(reviews) < th:
                discarted_users += 1
                continue
            n_elements_test = int(len(reviews) * train_size)
            train.append(reviews[:n_elements_test])
            test.append(reviews[n_elements_test:])
            test_sizes.append(len(test[-1]))
        print("Users: {} Discarted {}/{} with < {} interactions".format(total_users-discarted_users, discarted_users, total_users, th))
        print("Average test size: {}".format(np.array(test_sizes).mean()))
        print("Writing train...")
        with open(DATASET_DIR[dataset_name] + "/train.txt", 'w+') as file:
            for user_reviews in train:
                for review in user_reviews:
                    s = ' '.join(review)
                    file.writelines(s)
                    file.write("\n")
        file.close()

        print("Writing test...")
        with open(DATASET_DIR[dataset_name] + "/test.txt", 'w+') as file:
            for user_reviews in test:
                for review in user_reviews:
                    s = ' '.join(review)
                    file.writelines(s)
                    file.write("\n")
        file.close()
        print("Zipping train and test...")
        zip_file(DATASET_DIR[dataset_name] + "/train.txt")
        zip_file(DATASET_DIR[dataset_name] + "/test.txt")
        print("Loading reviews.. DONE")

    def generate_kg_relations(self):
        dataset_name = args.dataset
        mappings = get_all_entity_mappings(dataset_name)
        product = 'song'
        no_of_movies = len(mappings[product])+1
        movie_id_entity = edict(
            sang_by=([[] for _ in range(no_of_movies)], DATASET_DIR[dataset_name] + '/relations/sang_by_s_a.txt'),
            featured_by = ([[] for _ in range(no_of_movies)],DATASET_DIR[dataset_name] + '/relations/featured_by_s_a.txt'),
            belong_to = ([[] for _ in range(no_of_movies)], DATASET_DIR[dataset_name] + '/relations/belong_to_s_ca.txt'),
            mixed_by = ([[] for _ in range(no_of_movies)], DATASET_DIR[dataset_name] + '/relations/mixed_by_s_e.txt'),
            related_to = ([[] for _ in range(no_of_movies)], DATASET_DIR[dataset_name] + '/relations/related_to_s_rs.txt'),
            alternative_version_of = ([[] for _ in range(no_of_movies)],DATASET_DIR[dataset_name] + '/relations/alternative_version_of_s_rs.txt'),
            original_version_of = ([[] for _ in range(no_of_movies)],DATASET_DIR[dataset_name] + '/relations/orginal_version_of_s_rs.txt'),
            produced_by_producer=([[] for _ in range(no_of_movies)], DATASET_DIR[dataset_name] + '/relations/produced_by_producer_s_pr.txt'),
        )
        relations_path = DATASET_DIR[dataset_name] + "/relations/"
        if not os.path.isdir(relations_path):
            os.makedirs(relations_path)

        print("Inserting relations inside buckets...\n")
        with open(KG_COMPLETATION_DATASET_DIR[dataset_name] + '/kg_final.txt', 'r') as file:
            csv_reader = csv.reader(file, delimiter=' ')
            invalid = 0
            for row in csv_reader:
                row[0] = int(row[0])
                if row[0] not in mappings[product]:
                    invalid += 1
                    continue
                head = mappings[product][row[0]][0] #id of the movie in the kg
                relation = int(row[1])
                tail = int(row[2])

                if relation not in SELECTED_RELATIONS[dataset_name]: continue
                tail_entity_name = get_tail_entity_name(dataset_name, relation)
                relation_name = RELATION_NAMES[LASTFM][relation]
                if tail not in mappings[tail_entity_name]: continue
                kg_id_tail = mappings[tail_entity_name][tail]
                movie_id_entity[relation_name][0][head].append(kg_id_tail)
        file.close()
        #print(invalid)
        for relation_name in movie_id_entity.keys():
            relationship_filename = movie_id_entity[relation_name][1]
            associated_entity_list = movie_id_entity[relation_name][0]
            print("Populating " + relationship_filename + "...\n")
            with open(relationship_filename, 'w+') as file:
                for entitylist_for_movie in associated_entity_list:
                    s = ' '.join([str(entitity) for entitity in entitylist_for_movie])
                    file.writelines(s)
                    file.write("\n")
            zip_file(relationship_filename)

    #Generate mappings from uid to sensible attributes for gender, age and occupation
    def generate_user_attributes_mappings(self):
        dataset_name = self.args.dataset
        uid_attributes = {}
        #user_id, country, age, gender, playcount, registered_unixtime
        with open(DATASET_DIR[dataset_name] + "/users.dat", 'r') as file:
            csv_reader = csv.reader(file, delimiter=',')
            next(csv_reader, None)
            for row in csv_reader:
                uid = row[0]
                country = row[1]
                age = row[2]
                gender = row[3]
                uid_attributes[uid] = [country, age, gender]
        file.close()

        if not os.path.exists(DATASET_DIR[dataset_name] + "/mappings/"):
            os.makedirs(DATASET_DIR[dataset_name] + "/mappings/")

        # Write user_occupation mapping
        with open(DATASET_DIR[dataset_name] + "/mappings/uid2country.txt", 'w+') as file:
            for uid, attributes in uid_attributes.items():
                country = attributes[0]
                file.write(uid + "\t" + country + "\n")
        file.close()

        # Write user_age mapping
        with open(DATASET_DIR[dataset_name] + "/mappings/uid2age_map.txt", 'w+') as file:
            for uid, attributes in uid_attributes.items():
                age = int(attributes[1])
                if age == -1:
                    age_range = -1
                elif age < 18:
                    age_range = 1
                elif age >= 18 and age <= 24:
                    age_range = 18
                elif age >= 25 and age <= 34:
                    age_range = 25
                elif age >= 35 and age <= 44:
                    age_range = 35
                elif age >= 45 and age <= 49:
                    age_range = 45
                elif age >= 50 and age <= 55:
                    age_range = 50
                else:
                    age_range = 56
                #{1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44", 45: "45-49", 50: "50-55", 56: "56+"}
                file.write(uid + "\t" + str(age_range) + "\n")
        file.close()

        #Write user_gender mapping
        with open(DATASET_DIR[dataset_name] + "/mappings/uid2gender.txt", 'w+') as file:
            for uid, attributes in uid_attributes.items():
                gender = attributes[2]
                file.write(uid + "\t" + gender + "\n")
        file.close()
    def get_valid_users(self, args):
        dataset_name = self.args.dataset
        valid_users = set()
        users_file = open(DATASET_DIR[dataset_name] + "/users.dat", "r")
        reader = csv.reader(users_file)
        next(reader, None)
        for row in reader:
            uid = int(row[0])
            valid_users.add(uid)
        return valid_users

    def generate_kg_entities(self):
        dataset_name = self.args.dataset
        #Creates a dict of sets to store all the extracted entitities for every differnt type
        kg_entities = edict(
            user=(set(), 'user.txt'),
            song=(set(), 'song.txt'),
            artist=(set(), 'artist.txt'),
            engineer=(set(), 'engineer.txt'),
            producer=(set(), 'producer.txt'),
            category=(set(), 'category.txt'),
            related_song=(set(), 'related_song.txt'),
        )
        entity_path = DATASET_DIR[dataset_name] + "/entities/"
        if not os.path.isdir(entity_path):
            os.makedirs(entity_path)

        lastid2name = {}
        with open(DATASET_DIR[dataset_name] + "/tracks.txt") as file:
            reader = csv.reader(file, delimiter=",")
            next(reader, None)
            for row in reader:
                track_id = int(row[0])
                lastid2name[track_id] = row[1]
        file.close()

        file = open(KG_COMPLETATION_DATASET_DIR[dataset_name] + "/item_list.txt", "r")
        csv_reader = csv.reader(file, delimiter=' ')
        dbid2lastid = {}
        lastid2dbid = {}
        lastid2freebase = {}
        next(csv_reader, None)
        for row in csv_reader:
            last_id = int(row[0])
            if last_id not in lastid2name: continue
            dbid2lastid[int(row[1])] = int(row[0])
            lastid2dbid[int(row[0])] = int(row[1])
            lastid2freebase[int(row[0])] = row[2]
        file.close()

        kgid2freebase = {}
        file = open(KG_COMPLETATION_DATASET_DIR[dataset_name] + "/entity_list.txt", "r")
        csv_reader = csv.reader(file, delimiter=' ')
        next(csv_reader, None)
        for row in csv_reader:
            kgid2freebase[int(row[1])] = row[0]
        file.close()

        with open(KG_COMPLETATION_DATASET_DIR[dataset_name] + "/kg_final.txt", 'r') as file:
            csv_reader = csv.reader(file, delimiter=' ')
            invalid = 0
            count = 0
            for row in csv_reader:
                head = int(row[0])
                relation = int(row[1])
                tail = int(row[2])
                if head not in dbid2lastid:
                    invalid += 1
                    continue

                movie_id = head #OCCHIO
                tail_name = get_tail_entity_name(dataset_name, relation) #Retriving what is the tail of that relation
                kg_entities['song'][0].add(movie_id)
                kg_entities[tail_name][0].add(tail)
        file.close()
        #print(invalid, count)

        review_uid_kg_uid = {}
        valid_users = self.get_valid_users(args)
        # Write user entity
        with open(KG_COMPLETATION_DATASET_DIR[dataset_name] + "/user_list.txt", 'r') as file:
            csv_reader = csv.reader(file, delimiter=' ')
            next(csv_reader, None)
            with open(entity_path + "/user.txt", 'w+') as file:
                for row in csv_reader:
                    review_uid = int(row[0])
                    kg_uid = int(row[1])
                    if review_uid in kg_entities.user[0] or review_uid not in valid_users: continue
                    kg_entities.user[0].add(review_uid)
                    review_uid_kg_uid[review_uid] = kg_uid
                    file.writelines(str(kg_uid))
                    file.write("\n")
        file.close()
        zip_file(entity_path + "user.txt")

        with open(DATASET_DIR[dataset_name] + "/mappings/user_mappings.txt", 'w+') as file:
            header = ["kgid", "lastfmid"]
            file.write(' '.join(header) + "\n")
            for review_id, kg_id in review_uid_kg_uid.items():
                file.write('\t'.join([str(kg_id), str(review_id), "\n"]))
        file.close()

        #Populate movie entity file (Done by itself due to is different structure)
        new_id2old_id = {}
        with open(entity_path + "/song.txt", 'w+') as file:
            for idx, movie in enumerate(kg_entities['song'][0]):
                new_id2old_id[idx] = int(movie)
                file.write(str(idx) + "\n")
        file.close()
        zip_file(entity_path + "song.txt")

        # newId (0...n), oldId(movilandID), entityId(jointkgentityid), trackname, freebase id
        with open(DATASET_DIR[dataset_name] + "/mappings/product_mappings.txt", 'w+') as file:
            header = ["kgid", "lastfmid", "kgcompletionid", "trackname", "freebaseid"]
            file.write(' '.join(header) + "\n")
            for new_id, db_id in new_id2old_id.items():
                last_id = dbid2lastid[db_id]
                track_name = lastid2name[last_id]
                freebase_id = lastid2freebase[last_id]
                file.write('\t'.join([str(new_id), str(last_id), str(db_id), track_name, freebase_id, "\n"]))
        file.close()

        #Populating other entities
        for entity_name in get_entities_without_user(dataset_name):
            if entity_name == 'song': continue
            new_id2old_id = {}
            filename = entity_path + entity_name + '.txt'
            #Populate entities
            with open(filename, 'w+') as file:
                for idx, entity in enumerate(kg_entities[entity_name][0]):
                    new_id2old_id[idx] = int(entity)
                    file.write(str(idx) + "\n")
            file.close()

            # newId (0...n), entityId(jointkgentityid), entityNameDBPEDIA
            with open(DATASET_DIR[dataset_name] + "/mappings/" + entity_name + 'id2dbid.txt', 'w+') as file:
                header = ["kg_id", "kg_completion_id", "freebase_id"]
                file.write(' '.join(header) + "\n")
                for new_id, old_id in new_id2old_id.items():
                    entity_dblink = kgid2freebase[old_id]
                    file.write(str(new_id) + '\t' + str(old_id) + '\t' + entity_dblink + "\n")
            file.close()

            # Zip entities
            zip_file(filename)

#Generate the mapping from the KG Completation of Joint-KG to a PGPR readable dataset
class ML1MDatasetMapper(object):
    def __init__(self, args):
        self.args = args
        if not exists("datasets/ml1m/joint-kg/dataset.dat"):
            self.unify_dataset()
        #self.generate_dbpid_mlpid_mapping()
        #self.generate_kg_entities()
        #self.generate_kg_relations()
        #self.generate_user_attributes_mappings()
        self.generate_train_test_split()

    # Function usefull to join all the knowledge in a unique KG (needed for KTUP type datasets)
    def unify_dataset(self):
        dataset_name = self.dataset
        selected_relationship = SELECTED_RELATIONS[dataset_name]
        print("Unifying dataset from joint-kg Knowledge graph completation for {}...".format(dataset_name))
        with open(KG_COMPLETATION_DATASET_DIR[dataset_name] + "/dataset.dat", 'w+', newline='\n') as dataset_file:
            print("Loading joint-kg train...")
            with open(KG_COMPLETATION_DATASET_DIR[dataset_name] + "/kg/train.dat") as joint_kg_train:
                csv_reader = csv.reader(joint_kg_train, delimiter='\t')
                for row in csv_reader:
                    relation = int(row[2])
                    if relation not in selected_relationship: continue
                    dataset_file.writelines('\t'.join(row))
                    dataset_file.write("\n")
            joint_kg_train.close()
            print("Loading joint-kg valid...")
            with open(KG_COMPLETATION_DATASET_DIR[dataset_name] + "/kg/valid.dat") as joint_kg_valid:
                csv_reader = csv.reader(joint_kg_valid, delimiter='\t')
                for row in csv_reader:
                    relation = int(row[2])
                    if relation not in selected_relationship: continue
                    dataset_file.writelines('\t'.join(row))
                    dataset_file.write("\n")
            joint_kg_valid.close()
            print("Loading joint-kg test...")
            with open(KG_COMPLETATION_DATASET_DIR[dataset_name] + "/kg/test.dat") as joint_kg_test:
                csv_reader = csv.reader(joint_kg_test, delimiter='\t')
                for row in csv_reader:
                    relation = int(row[2])
                    if relation not in selected_relationship: continue
                    dataset_file.writelines('\t'.join(row))
                    dataset_file.write("\n")
            joint_kg_test.close()
            print("Unifying dataset from joint-kg Knowledge graph completation... DONE")
        dataset_file.close()

    def generate_dbpid_mlpid_mapping(self):
        dataset_name = self.args.dataset
        file = open(DATASET_DIR[dataset_name] + "/joint-kg/i2kg_map.tsv", "r")
        dburl_to_mlid = {}
        reader = csv.reader(file, delimiter="\t")
        for row in reader:
            mlid = int(row[0])
            name = row[1]
            dburl = row[2]
            dburl_to_mlid[dburl] = [mlid, name]
        file.close()

        file = open(DATASET_DIR[dataset_name] + "/joint-kg/kg/e_map.dat", "r", encoding='latin-1')
        fileo = open(DATASET_DIR[dataset_name] + "/mappings/product_mappings.txt", "w+")
        writer = csv.writer(fileo, delimiter="\t")
        header = ["mlid", "dbid", "name", "dburl"]
        writer.writerow(header)
        reader = csv.reader(file, delimiter="\t")
        for row in reader:
            dbid = int(row[0])
            dburl = row[1]
            if dburl not in dburl_to_mlid: continue
            mlid = dburl_to_mlid[dburl][0]
            name = dburl_to_mlid[dburl][1]
            writer.writerow([mlid, dbid, name, dburl])
        file.close()
        fileo.close()

    def generate_train_test_split(self):
        dataset_name = self.args.dataset
        uid_review_tuples = {}
        dataset_size = 0
        valid_movies = get_valid_products(dataset_name)
        print("Loading reviews...")
        with open(DATASET_DIR[dataset_name] + "/ratings.dat", 'r', encoding='latin-1') as reviews_file:
            reader = csv.reader(reviews_file, delimiter='\n')
            for row in reader:
                row = ''.join(row).strip().split("::")
                if int(row[1]) not in valid_movies: continue
                if row[0] not in uid_review_tuples:
                    uid_review_tuples[row[0]] = []
                uid_review_tuples[row[0]].append((row[0], row[1], row[2], row[3]))
                dataset_size += 1
        reviews_file.close()
        train_size = 0.8
        print("Performing split {}/{}...".format(train_size*100, 100-train_size*100))
        for uid, reviews in uid_review_tuples.items():
            reviews.sort(key=lambda x: int(x[3])) #sorting from recent to older

        train = []
        test = []
        total_users = 0
        discarted_users = 0
        th = 10
        test_sizes = []
        for uid, reviews in uid_review_tuples.items():  # python dict are sorted, 1...nuser
            total_users += 1
            if len(reviews) < th:
                discarted_users += 1
                continue
            n_elements_test = int(len(reviews) * train_size)
            train.append(reviews[:n_elements_test])
            test.append(reviews[n_elements_test:])
            test_sizes.append(len(test[-1]))
        print("Users: {} Discarted {}/{} with < {} interactions".format(total_users-discarted_users, discarted_users, total_users, th))
        print("Average test size: {}".format(np.array(test_sizes).mean()))
        print("Writing train...")
        with open(DATASET_DIR[dataset_name] + "/train10.txt", 'w+') as file:
            for user_reviews in train:
                for review in user_reviews:
                    s = ' '.join(review)
                    file.writelines(s)
                    file.write("\n")
        file.close()

        print("Writing test...")
        with open(DATASET_DIR[dataset_name] + "/test10.txt", 'w+') as file:
            for user_reviews in test:
                for review in user_reviews:
                    s = ' '.join(review)
                    file.writelines(s)
                    file.write("\n")
        file.close()
        print("Zipping train and test...")
        zip_file(DATASET_DIR[dataset_name] + "/train10.txt")
        zip_file(DATASET_DIR[dataset_name] + "/test10.txt")
        print("Loading reviews.. DONE")

    #Generate mappings from uid to sensible attributes for gender, age and occupation
    def generate_user_attributes_mappings(self):
        dataset_name = self.args.dataset
        users_id = []
        genders = []
        ages = []
        occupations = []
        with open(DATASET_DIR[dataset_name] + "/users.dat", 'r') as file:
            csv_reader = csv.reader(file, delimiter='\n')
            for row in csv_reader:
                attributes = row[0].strip().split('::')
                users_id.append(attributes[0])
                genders.append(attributes[1])
                ages.append(attributes[2])
                occupations.append(attributes[3])
        file.close()

        #Write user_gender mapping
        with open(DATASET_DIR[dataset_name] + "/mappings/uid2gender.txt", 'w+') as file:
            for user, gender in zip(users_id, genders):
                file.write(user + "\t" + gender + "\n")
        file.close()

        # Write user_occupation mapping
        with open(DATASET_DIR[dataset_name] + "/mappings/uid2occupation.txt", 'w+') as file:
            for user, occupation in zip(users_id, occupations):
                file.write(user + "\t" + occupation + "\n")
        file.close()

        # Write user_age mapping
        with open(DATASET_DIR[dataset_name] + "/mappings/uid2age_map.txt", 'w+') as file:
            for user, age in zip(users_id, ages):
                file.write(user + "\t" + age + "\n")
        file.close()

    def generate_kg_entities(self):
        dataset_name = self.args.dataset
        #Creates a dict of sets to store all the extracted entitities for every differnt type
        kg_entities = edict(
            user=(set(), 'user.txt'),
            movie=(set(), 'movie.txt'),
            actor=(set(), 'actor.txt'),
            director=(set(), 'director.txt'),
            producer=(set(), 'producer.txt'),
            production_company=(set(), 'production_company.txt'),
            category=(set(), 'category.txt'),
            editor=(set(), 'editor.txt'),
            writter=(set(), 'writter.txt'),
            cinematographer=(set(), 'cinematographer.txt'),
            composer=(set(), 'composer.txt'),
        )
        entity_path = DATASET_DIR[dataset_name] + "/entities/"
        if not os.path.isdir(entity_path):
            os.makedirs(entity_path)

        file = open(DATASET_DIR[dataset_name] + "/mappings/product_mappings.txt", "r")
        reader = csv.reader(file, delimiter='\n')
        db_pid2ml_pid = {}
        ml_pid2db_pid = {}
        ml_pid2metada = {}
        next(reader, None)
        for i, row in enumerate(reader):
            row = row[0].strip().split("\t")
            db_pid2ml_pid[int(row[1])] = int(row[0])
            ml_pid2db_pid[int(row[0])] = int(row[1])
            ml_pid2metada[int(row[0])] = [row[2], row[3]]
        file.close()
        kg_entities['movie'][0] = set(ml_pid2db_pid.keys())
        with open(KG_COMPLETATION_DATASET_DIR[dataset_name] + "/dataset.dat", 'r') as file:
            csv_reader = csv.reader(file, delimiter='\t')
            for row in csv_reader:
                head = int(row[0])
                tail = row[1]
                relation = int(row[2])
                if head not in db_pid2ml_pid: continue

                #movie_id = db_pid2ml_pid[head]
                tail_name = get_tail_entity_name(dataset_name, relation) #Retriving what is the tail of that relation

                #kg_entities['movie'][0].add(movie_id)
                kg_entities[tail_name][0].add(tail)
        file.close()

        # Write user entity
        with open(DATASET_DIR[dataset_name] + "/users.dat", 'r') as file:
            csv_reader = csv.reader(file, delimiter='\n')
            for row in csv_reader:
                row = row[0].strip().split('::')
                uid = int(row[0])
                kg_entities.user[0].add(uid)

        new_id2old_id = {}
        with open(entity_path + "/user.txt", 'w+') as file:
            for idx, u in enumerate(kg_entities.user[0]):
                new_id2old_id[idx] = int(u)
                file.writelines(str(idx))
                file.write("\n")
        file.close()

        zip_file(entity_path + "/user.txt")

        with open(DATASET_DIR[dataset_name] + "/mappings/user_mappings.txt", 'w+') as file:
            header = ["kg_id", "ml1m_id"]
            file.write('\t'.join(header) + "\n")
            for new_id, old_id in new_id2old_id.items():
                file.write(str(new_id) + '\t' + str(old_id) + "\n")
        file.close()

        #Populate movie entity file (Done by itself due to is different structure)
        new_id2old_id = {}
        with open(entity_path + "/movie.txt", 'w+') as file:
            for idx, movie in enumerate(kg_entities['movie'][0]):
                new_id2old_id[idx] = int(movie)
                file.write(str(idx) + "\n")
        file.close()

        # newId (0...n), oldId(movilandID), entityId(jointkgentityid), entityNameDBPEDIA
        with open(DATASET_DIR[dataset_name] + "/mappings/product_mappings.txt", 'w+') as file:
            header = ["kg_id", "ml1m_id", "db_id", "name", "dbpedia_url"]
            file.write('\t'.join(header) + "\n")
            for new_id, old_id in new_id2old_id.items():
                entity_id = ml_pid2db_pid[old_id]
                file.write("\t".join([str(new_id), str(old_id), str(entity_id), ml_pid2metada[old_id][0], ml_pid2metada[old_id][1] + "\n"]))
        file.close()

        zip_file(entity_path + "/movie.txt")

        #Retrive the dblink associated to the entity id in the kg completion
        entity_id2dblink = {}
        entity_file = open(DATASET_DIR[dataset_name] + "/joint-kg/kg/e_map.dat", "r", encoding='latin-1')
        reader = csv.reader(entity_file, delimiter="\t")
        for row in reader:
            eid = int(row[0])
            dblink = row[1]
            entity_id2dblink[eid] = dblink

        #Populating other entities
        for entity_name in get_entities_without_user(dataset_name):
            if entity_name == 'movie': continue
            new_id2old_id = {}
            filename = entity_path + entity_name + '.txt'
            #Populate entities
            with open(filename, 'w+') as file:
                for idx, entity in enumerate(kg_entities[entity_name][0]):
                    new_id2old_id[idx] = int(entity)
                    file.write(str(idx) + "\n")
            file.close()

            # newId (0...n), entityId(jointkgentityid), entityNameDBPEDIA
            with open(DATASET_DIR[dataset_name] + "/mappings/" + entity_name + 'id2dbid.txt', 'w+') as file:
                header = ["kgid", "dbid", "dblink"]
                file.write("\t".join(header) + "\n")
                for new_id, old_id in new_id2old_id.items():
                    entity_dblink = entity_id2dblink[old_id]
                    file.write(str(new_id) + '\t' + str(old_id) + '\t' + entity_dblink + "\n")
            file.close()

            # Zip entities
            zip_file(filename)

    def generate_kg_relations(self):
        dataset_name = args.dataset
        mappings = get_all_entity_mappings(dataset_name)

        no_of_movies = len(mappings['movie'])+1
        movie_id_entity = edict(
            production_company=([[] for _ in range(no_of_movies)], DATASET_DIR[dataset_name] + '/relations/produced_by_company_m_pc.txt'),
            composer=([[] for _ in range(no_of_movies)], DATASET_DIR[dataset_name] + '/relations/composed_by_m_c.txt'),
            category=([[] for _ in range(no_of_movies)], DATASET_DIR[dataset_name] + '/relations/belong_to_m_ca.txt'),
            director=([[] for _ in range(no_of_movies)], DATASET_DIR[dataset_name] + '/relations/directed_by_m_d.txt'),
            actor=([[] for _ in range(no_of_movies)], DATASET_DIR[dataset_name] + '/relations/starring_m_a.txt'),
            cinematographer=([[] for _ in range(no_of_movies)], DATASET_DIR[dataset_name] + '/relations/cinematography_m_ci.txt'),
            editor=([[] for _ in range(no_of_movies)], DATASET_DIR[dataset_name] + '/relations/edited_by_m_ed.txt'),
            producer=([[] for _ in range(no_of_movies)], DATASET_DIR[dataset_name] + '/relations/produced_by_producer_m_pr.txt'),
            writter=([[] for _ in range(no_of_movies)], DATASET_DIR[dataset_name] + '/relations/wrote_by_m_w.txt'),
        )
        relations_path = DATASET_DIR[dataset_name] + "/relations/"
        if not os.path.isdir(relations_path):
            os.makedirs(relations_path)
        invalid = 0
        print("Inserting relations inside buckets...\n")
        with open(KG_COMPLETATION_DATASET_DIR[dataset_name] + '/dataset.dat', 'r') as file:
            csv_reader = csv.reader(file, delimiter='\n')
            for row in csv_reader:
                row = row[0].strip().split("\t")
                db_pid = int(row[0])
                if db_pid not in mappings['movie']:
                    invalid += 1
                    continue
                head = mappings['movie'][db_pid][0] #id of the movie in the kg
                tail = int(row[1])
                relation = int(row[2])

                if relation not in SELECTED_RELATIONS[dataset_name]:
                    invalid += 1
                    continue
                tail_entity_name = get_tail_entity_name(dataset_name, relation)
                if tail not in mappings[tail_entity_name]:
                    invalid += 1
                    continue
                kg_id_tail = mappings[tail_entity_name][tail]
                movie_id_entity[tail_entity_name][0][head].append(kg_id_tail)
        file.close()

        print("Invalid relationships:", invalid)
        for entitity_name in get_entities_without_user(dataset_name):
            if entitity_name == 'movie': continue
            relationship_filename = movie_id_entity[entitity_name][1]
            associated_entity_list = movie_id_entity[entitity_name][0]
            print("Populating " + relationship_filename + "...\n")
            with open(relationship_filename, 'w+') as file:
                for entitylist_for_movie in associated_entity_list:
                    s = ' '.join([str(entitity) for entitity in entitylist_for_movie])
                    file.writelines(s)
                    file.write("\n")
            zip_file(relationship_filename)


#Generate the mapping from the Amazon E-commerce dataset to a PGPR readable dataset
#This dataset is very heavy and result without using words may have very low ndcg values
#Please manually set the thresholds for discarding user product and words
class Amazon2018DatasetMapper(object):
    def __init__(self, args):
        self.args = args
        if args.use_words:
            self.train_test_split_with_words()
        else:
            self.train_test_split()
        self.generate_entities_mappings()
        self.generate_relations()


    def get_mapping_entity(self, relation):
        relation2entity = {"belong_to": "category", "produced_by": "brand", "also_bought": "rproduct", "also_viewed": "rproduct"}
        entity = relation2entity[relation]
        mapping = {}
        filei = open(DATASET_DIR[self.args.dataset] + "/mappings/" + entity + "id2dbid.txt", 'r')
        reader = csv.reader(filei, delimiter="\t")
        next(reader, None)
        for row in reader:
            mapping[row[1]] = row[0]
        return mapping

    def generate_entities_mappings(self, valid_products):
        dataset_name = args.dataset
        raw_review_file = {CD: "CDs_and_Vinyl", BEAUTY: "Beauty", CLOTH: "Clothing_Shoes_and_Jewelry", CELL: "Cell_Phones_and_Accessories"}


        filei = open(DATASET_DIR[dataset_name] + "/raw/2018_meta_" + raw_review_file[dataset_name] + ".json")
        data = []
        for line in filei:
            line = json.loads(line)
            if line["asin"] not in valid_products: continue
            data.append(line)
        filei.close()
        entities = {"brand": set(), "category": set(), "rproduct": set()}
        products = []
        product_metadata = {}
        #similar_item = set()
        for row in data:
            pid = row["asin"]
            products.append([pid, row["title"]])
            if pid not in product_metadata:
                product_metadata[pid] = {"produced_by": set(), "belong_to": set(), "also_bought": set(), "also_viewed": set()}
            mc = row['main_cat']
            entities["category"].add(mc)
            product_metadata[pid]["belong_to"].add(mc)
            for c in row['category']:
                entities["category"].add(c)
                product_metadata[pid]["belong_to"].add(c)
            for av in row["also_view"]:
                entities["rproduct"].add(av)
                product_metadata[pid]["also_viewed"].add(av)
            entities["brand"].add(row["brand"])
            product_metadata[pid]["produced_by"].add(row["brand"])
            #similar_item.add(row["similar_item"])
            for ab in row["also_buy"]:
                entities["rproduct"].add(ab)
                product_metadata[pid]["also_bought"].add(ab)

        ensure_dir(DATASET_DIR[dataset_name] + "/mappings/")
        fileo = open(DATASET_DIR[dataset_name] + "/mappings/product_mappings.txt", "w+")
        writer = csv.writer(fileo, delimiter="\t")
        header = ["kg_id", "amazon_id", "name"]
        writer.writerow(header)
        for kgid, pid_name in enumerate(products):
            amazon_id = pid_name[0]
            name = pid_name[1]
            writer.writerow([kgid, amazon_id, name])
        fileo.close()

        ensure_dir(DATASET_DIR[dataset_name] + "/entities/")
        fileo = open(DATASET_DIR[dataset_name] + "/entities/product.txt", "w+")
        writer = csv.writer(fileo)
        for kgid in range(len(products)):
            writer.writerow([kgid])
        fileo.close()
        zip_file(DATASET_DIR[dataset_name] + "/entities/product.txt")

        #Other entities
        for entity, values in entities.items():
            fileo = open(DATASET_DIR[dataset_name] + "/mappings/" + entity + "id2dbid.txt", "w+")
            writer = csv.writer(fileo, delimiter="\t")
            header = ["kg_id", "amazon_id"]
            writer.writerow(header)
            entities_values = []
            for kgid, value in enumerate(values):
                amazon_id = value
                entities_values.append(kgid)
                writer.writerow([kgid, amazon_id])
            fileo.close()

            ensure_dir(DATASET_DIR[dataset_name] + "/entities/")
            fileo = open(DATASET_DIR[dataset_name] + "/entities/" + entity + ".txt", "w+")
            writer = csv.writer(fileo)
            for kgid in entities_values:
                writer.writerow([kgid])
            fileo.close()
            zip_file(DATASET_DIR[dataset_name] + "/entities/" + entity + ".txt")


        #Relationships
        relation_filename = {"also_bought": "also_bought_p_p.txt", "also_viewed": "also_viewed_p_p.txt", "produced_by": "brand_p_b.txt",
         "belong_to": "category_p_c.txt"}
        for relation, filename in relation_filename.items():
            mapping = self.get_mapping_entity(relation)
            ensure_dir(DATASET_DIR[dataset_name] + "/relations")
            fileo = open(DATASET_DIR[dataset_name] + "/relations/" + filename, mode='w+')
            writer = csv.writer(fileo, delimiter=" ")
            for pid_name in products:
                pid = pid_name[0]
                entity_kgid = [mapping[x] for x in product_metadata[pid][relation]]
                writer.writerow(entity_kgid)
            fileo.close()
            zip_file(DATASET_DIR[dataset_name] + "/relations/" +  filename)


    def generate_relations(self):
        dataset_name = args.dataset
        original_relations_filenames = ["also_bought_p_p.txt.gz", "also_viewed_p_p.txt.gz", "bought_together_p_p.txt.gz", "brand_p_b.txt.gz", "category_p_c.txt.gz"]
        relations_filenames = ['.'.join(x.split('.')[:-1]) for x in original_relations_filenames]
        #invalid_products = get_invalid_products(dataset_name, True)
        ensure_dir(DATASET_DIR[dataset_name] + "/relations")
        for idx, relation_filename in enumerate(relations_filenames):
            filei = gzip.open(DATASET_DIR[dataset_name] + "/old/" + original_relations_filenames[idx], mode='rt')
            fileo = open(DATASET_DIR[dataset_name] + "/relations/" + relation_filename, mode='w+')
            reader = csv.reader(filei)
            writer = csv.writer(fileo)
            for i, row in enumerate(reader):
                #if i in invalid_products: continue
                writer.writerow(row)
            fileo.close()
            filei.close()
            zip_file(DATASET_DIR[dataset_name] + "/relations/" +  relation_filename)

    def tokenize(self, text):
        tokens = nltk.word_tokenize(text)
        stems = []
        for item in tokens:
            stems.append(PorterStemmer().stem(item))
        return stems

    def preprocessor(self, text):
        text = re.sub(r'[^a-zA-Z ]', '', text)
        return text.lower()

    def train_test_split_with_words(self):
        sp = spacy.load('en_core_web_lg')

        dataset_name = args.dataset
        raw_review_file = {CD: "CDs_and_Vinyl", BEAUTY: "Beauty", CLOTH: "Clothing_Shoes_and_Jewelry", CELL: "Cell_Phones_and_Accessories"}
        f = open(DATASET_DIR[dataset_name] + '/raw/' + raw_review_file[dataset_name] + '_5_2018.json')
        data = []
        for line in f:
            data.append(json.loads(line))
        f.close()

        valid_reviews = {}
        original_dataset_products, original_dataset_users = {}, set()
        no_of_reviews = 0
        for x in data:
            uid = x['reviewerID']
            pid = x['asin']
            rating = str(x['overall'])
            timestamp = str(x['unixReviewTime'])
            if uid not in valid_reviews:
                valid_reviews[uid] = []
            valid_reviews[uid].append([uid, pid, rating, timestamp])
            original_dataset_users.add(uid)
            if pid not in original_dataset_products:
                original_dataset_products[pid] = 0
            original_dataset_products[pid] += 1
            no_of_reviews += 1

        no_of_products, no_of_users = len(original_dataset_products), len(original_dataset_users)
        print("Original dataset has {} users and {} products with {} reviews and sparsity of {}".format(
            no_of_users, no_of_products,  no_of_reviews,  no_of_reviews/(no_of_users*no_of_products)))

        th_products = 5
        print("Removing product that appear less than {} times in reviews...".format(th_products))
        for pid in list(original_dataset_products.keys()):
            if original_dataset_products[pid] < th_products:
                del original_dataset_products[pid]
        valid_products = original_dataset_products.keys()
        print("Removed {}/{} products, valid products: {}".format(no_of_products-len(valid_products), no_of_products, len(valid_products)))

        th_users = 15
        no_of_removed_reviews = 0
        no_of_removed_users = 0
        print("Removing users with less than {} reviews...".format(th_users))
        for uid in list(valid_reviews.keys()):
            user_valid_reviews = valid_reviews[uid]
            if len(user_valid_reviews) < th_users:
                no_of_removed_reviews += len(user_valid_reviews)
                del valid_reviews[uid]
                no_of_removed_users += 1
                #If user is removed reduce the count of the items present in their reviews
                for review in user_valid_reviews:
                    pid = review[1]
                    if pid not in original_dataset_products: continue
                    original_dataset_products[pid] -= 1
                    if original_dataset_products[pid] == 0:
                        del original_dataset_products[pid]
                continue
            pid_removed_valid_reviews = []
            no_user_review = len(user_valid_reviews)
            for i in range(len(user_valid_reviews)):
                review = user_valid_reviews[i]
                pid = review[1]
                if pid in valid_products:
                    pid_removed_valid_reviews.append(review)
                else:
                    no_user_review -= 1
                    no_of_removed_reviews += 1
                    if no_user_review < 10:
                        no_of_removed_reviews += no_user_review
                        for review in valid_reviews[uid]:
                            pid = review[1]
                            if pid not in original_dataset_products: continue
                            original_dataset_products[pid] -= 1
                            if original_dataset_products[pid] == 0:
                                del original_dataset_products[pid]
                        del valid_reviews[uid] #Remove user
                        break
            if len(pid_removed_valid_reviews) < 10: continue
            valid_reviews[uid] = pid_removed_valid_reviews
        valid_users = valid_reviews.keys()
        valid_products = original_dataset_products.keys()
        no_of_reviews_left = no_of_reviews-no_of_removed_reviews
        print("After preprocessing are left {} users, {} products, {} valid reviews and sparsity of {}.".format(len(valid_users), len(valid_products),
              no_of_reviews_left,no_of_reviews_left/(len(valid_users)*len(valid_products))))
        train_size = 0.7
        print("Preparing train/test split of size {}/{}".format(train_size*100, 100-train_size*100))
        train = []
        user_train_pids = {}
        test = []
        test_sizes = []
        for uid, reviews in valid_reviews.items():  # python dict are sorted, 1...nuser
            n_elements_test = int(len(reviews) * train_size)
            train.append(reviews[:n_elements_test])
            user_train_pids[uid] = [review[1] for review in reviews[:n_elements_test]]
            test.append(reviews[n_elements_test:])
            test_sizes.append(len(test[-1]))
        print("Average Test Size: ", np.array(test_sizes).mean())

        print("Extracting vocabulary")
        user_product_words = {uid: {} for uid in valid_users}
        vocab = {}
        for x in data:
            uid = x['reviewerID']
            if uid not in valid_users: continue
            pid = x['asin']
            if pid not in valid_products or pid not in user_train_pids[uid]: continue

            for feature in ['reviewText', 'summary']:
                if feature not in x: continue
                text_review = x[feature]
                tokens = sp(text_review)
                for token in tokens:
                    token_preprocessed = self.preprocessor(token.lemma_)
                    if token_preprocessed != '':
                        if token_preprocessed not in vocab:
                            vocab[token_preprocessed] = 0
                        vocab[token_preprocessed] += 1
                        if pid not in user_product_words[uid]:
                            user_product_words[uid][pid] = []
                        user_product_words[uid][pid].append(token_preprocessed)

        #word_th = 5
        #for word in list(vocab.keys()):
        #    if vocab[word] < word_th:
        #        del vocab[word]

        print("Saving vocab")
        fileo = open(DATASET_DIR[dataset_name] + "/mappings/vocabid2dbid.txt", "w+")
        writer = csv.writer(fileo, delimiter="\t")
        header = ["kg_id", "word"]
        writer.writerow(header)
        vocab_mapping = {}
        entities_values = []
        for kgid, value in enumerate(vocab):
            amazon_id = value
            entities_values.append(kgid)
            vocab_mapping[amazon_id] = str(kgid)
            writer.writerow([kgid, amazon_id])
        fileo.close()

        ensure_dir(DATASET_DIR[dataset_name] + "/entities/")
        fileo = open(DATASET_DIR[dataset_name] + "/entities/vocab.txt", "w+")
        writer = csv.writer(fileo)
        for kgid in entities_values:
            writer.writerow([kgid])
        fileo.close()
        zip_file(DATASET_DIR[dataset_name] + "/entities/vocab.txt")

        print("Writing train...")
        with open(DATASET_DIR[dataset_name] + "/train.txt", 'w+') as file:
            for user_reviews in train:
                for review in user_reviews:
                    uid, pid = review[0], review[1]
                    s = ' '.join(review)
                    if uid in user_product_words:
                        if pid in user_product_words[uid]:
                            s = s + ' '.join([vocab_mapping[word] for word in user_product_words[uid][pid] if word in vocab_mapping])
                    file.writelines(s)
                    file.write("\n")
        file.close()

        print("Writing test...")
        with open(DATASET_DIR[dataset_name] + "/test.txt", 'w+') as file:
            for user_reviews in test:
                for review in user_reviews:
                    s = ' '.join(review)
                    file.writelines(s)
                    file.write("\n")
        file.close()
        print("Zipping train and test...")
        zip_file(DATASET_DIR[dataset_name] + "/train.txt")
        zip_file(DATASET_DIR[dataset_name] + "/test.txt")
        print("Loading reviews.. DONE")


        print("Generating User and Product mappings between original dataset and kg new ids...")
        self.generate_user_mappings(valid_users)
        print("Generating User and Product mappings between original dataset and kg new ids... DONE")
        self.generate_entities_mappings(valid_products)

    def train_test_split(self):
        dataset_name = args.dataset
        raw_review_file = {CD: "CDs_and_Vinyl", BEAUTY: "Beauty", CLOTH: "Clothing_Shoes_and_Jewelry", CELL: "Cell_Phones_and_Accessories"}
        f = open(DATASET_DIR[dataset_name] + '/raw/' + raw_review_file[dataset_name] + '_5_2018.json')
        data = []
        for line in f:
            data.append(json.loads(line))
        f.close()

        valid_reviews = {}
        original_dataset_products, original_dataset_users = {}, set()
        no_of_reviews = 0
        for x in data:
            uid = x['reviewerID']
            pid = x['asin']
            rating = str(x['overall'])
            timestamp = str(x['unixReviewTime'])
            if uid not in valid_reviews:
                valid_reviews[uid] = []
            valid_reviews[uid].append([uid, pid, rating, timestamp])
            original_dataset_users.add(uid)
            if pid not in original_dataset_products:
                original_dataset_products[pid] = 0
            original_dataset_products[pid] += 1
            no_of_reviews += 1

        no_of_products, no_of_users = len(original_dataset_products), len(original_dataset_users)
        print("Original dataset has {} users and {} products with {} reviews and sparsity of {}".format(
            no_of_users, no_of_products,  no_of_reviews,  no_of_reviews/(no_of_users*no_of_products)))

        th_products = 10
        print("Removing product that appear less than {} times in reviews...".format(th_products))
        for pid in list(original_dataset_products.keys()):
            if original_dataset_products[pid] < th_products:
                del original_dataset_products[pid]
        valid_products = original_dataset_products.keys()
        print("Removed {}/{} products, valid products: {}".format(no_of_products-len(valid_products), no_of_products, len(valid_products)))

        th_users = 15
        no_of_removed_reviews = 0
        no_of_removed_users = 0
        print("Removing users with less than {} reviews...".format(th_users))
        for uid in list(valid_reviews.keys()):
            user_valid_reviews = valid_reviews[uid]
            if len(user_valid_reviews) < th_users:
                no_of_removed_reviews += len(user_valid_reviews)
                del valid_reviews[uid]
                no_of_removed_users += 1
                #If user is removed reduce the count of the items present in their reviews
                for review in user_valid_reviews:
                    pid = review[1]
                    if pid not in original_dataset_products: continue
                    original_dataset_products[pid] -= 1
                    if original_dataset_products[pid] == 0:
                        del original_dataset_products[pid]
                continue
            pid_removed_valid_reviews = []
            no_user_review = len(user_valid_reviews)
            for i in range(len(user_valid_reviews)):
                review = user_valid_reviews[i]
                pid = review[1]
                if pid in valid_products:
                    pid_removed_valid_reviews.append(review)
                else:
                    no_user_review -= 1
                    no_of_removed_reviews += 1
                    if no_user_review < 10:
                        no_of_removed_reviews += no_user_review
                        for review in valid_reviews[uid]:
                            pid = review[1]
                            if pid not in original_dataset_products: continue
                            original_dataset_products[pid] -= 1
                            if original_dataset_products[pid] == 0:
                                del original_dataset_products[pid]
                        del valid_reviews[uid] #Remove user
                        break
            if len(pid_removed_valid_reviews) < 10: continue
            valid_reviews[uid] = pid_removed_valid_reviews
        valid_users = valid_reviews.keys()
        valid_products = original_dataset_products.keys()
        no_of_reviews_left = no_of_reviews-no_of_removed_reviews
        print("After preprocessing are left {} users, {} products, {} valid reviews and sparsity of {}.".format(len(valid_users), len(valid_products),
              no_of_reviews_left,no_of_reviews_left/(len(valid_users)*len(valid_products))))
        train_size = 0.7
        print("Preparing train/test split of size {}/{}".format(train_size*100, 100-train_size*100))
        train = []
        user_train_pids = {}
        test = []
        test_sizes = []
        for uid, reviews in valid_reviews.items():  # python dict are sorted, 1...nuser
            n_elements_test = int(len(reviews) * train_size)
            train.append(reviews[:n_elements_test])
            user_train_pids[uid] = [review[1] for review in reviews[:n_elements_test]]
            test.append(reviews[n_elements_test:])
            test_sizes.append(len(test[-1]))
        print("Average Test Size: ", np.array(test_sizes).mean())

        print("Writing train...")
        with open(DATASET_DIR[dataset_name] + "/train.txt", 'w+') as file:
            for user_reviews in train:
                for review in user_reviews:
                    s = ' '.join(review)
                    file.writelines(s)
                    file.write("\n")
        file.close()

        print("Writing test...")
        with open(DATASET_DIR[dataset_name] + "/test.txt", 'w+') as file:
            for user_reviews in test:
                for review in user_reviews:
                    s = ' '.join(review)
                    file.writelines(s)
                    file.write("\n")
        file.close()
        print("Zipping train and test...")
        zip_file(DATASET_DIR[dataset_name] + "/train.txt")
        zip_file(DATASET_DIR[dataset_name] + "/test.txt")
        print("Loading reviews.. DONE")


        print("Generating User and Product mappings between original dataset and kg new ids...")
        self.generate_user_mappings(valid_users)
        print("Generating User and Product mappings between original dataset and kg new ids... DONE")
        self.generate_entities_mappings(valid_products)

    def generate_user_mappings(self, users):
        dataset_name = args.dataset
        # User entity mapping
        ensure_dir(DATASET_DIR[dataset_name] + "/mappings/")
        fileo = open(DATASET_DIR[dataset_name] + "/mappings/user_mappings.txt", "w+")
        writer = csv.writer(fileo, delimiter="\t")
        header = ["kg_id", "amazon_id"]
        writer.writerow(header)
        for kgid, amazon_uid in enumerate(users):
            writer.writerow([kgid, amazon_uid])
        fileo.close()

        ensure_dir(DATASET_DIR[dataset_name] + "/entities/")
        fileo = open(DATASET_DIR[dataset_name] + "/entities/user.txt", "w+")
        writer = csv.writer(fileo)
        for kgid in range(len(users)):
            writer.writerow([kgid])
        fileo.close()
        zip_file(DATASET_DIR[dataset_name] + "/entities/user.txt")

#Adjust the PGPR Amazon 2014 dataset, recovering timestamps and ordering train-test by timestamp
class Amazon2014DatasetMapper(object):
    def __init__(self, args):
        self.args = args
        self.performe_adjustment()

    def performe_adjustment(self):
        dataset_name = args.dataset
        words = {}  # uid, pid, words
        uid_pids = {}
        count = 0
        os.rename(DATASET_DIR[dataset_name] + "/users.txt.gz", DATASET_DIR[dataset_name] + "/user.txt.gz")
        os.rename(DATASET_DIR[dataset_name] + "/train.txt.gz", DATASET_DIR[dataset_name] + "/otrain.txt.gz")
        os.rename(DATASET_DIR[dataset_name] + "/test.txt.gz", DATASET_DIR[dataset_name] + "/otest.txt.gz")
        for split in ["otrain", "otest"]:
            filei = gzip.open(DATASET_DIR[dataset_name] + "/" + split + ".txt.gz", 'rt')
            reader = csv.reader(filei, delimiter="\t")
            for row in reader:
                uid = row[0]
                pid = row[1]
                if uid not in uid_pids:
                    uid_pids[uid] = []
                uid_pids[uid].append(pid)
                if uid not in words:
                    words[uid] = {}
                words[uid][pid] = row[2]
                count += 1
            filei.close()

        raw_review_file = {CD: "CDs_and_Vinyl", BEAUTY: "Beauty", CLOTH: "Clothing_Shoes_and_Jewelry",
                           CELL: "Cell_Phones_and_Accessories"}
        f = open(DATASET_DIR[dataset_name] + '/' + raw_review_file[dataset_name] + '_5.json')
        data = []
        for line in f:
            data.append(json.loads(line))
        f.close()

        mapping_users = {}
        filei = gzip.open(DATASET_DIR[dataset_name] + "/user.txt.gz", 'rt')
        reader = csv.reader(filei)
        for idx, row in enumerate(reader):
            uid = row[0]
            mapping_users[uid] = str(idx)
        filei.close()

        mapping_product = {}
        filei = gzip.open(DATASET_DIR[dataset_name] + "/product.txt.gz", 'rt')
        reader = csv.reader(filei)
        for idx, row in enumerate(reader):
            pid = row[0]
            mapping_product[pid] = str(idx)

        rating_time = {}  # uid, pid, [rating, time]
        for x in data:
            uid_old = x['reviewerID']
            if uid_old not in mapping_users: continue
            uid = mapping_users[uid_old]
            old_pid = x['asin']
            rating = x['overall']
            if old_pid not in mapping_product:
                continue
            pid = mapping_product[old_pid]
            timestamp = x['unixReviewTime']
            if uid not in rating_time:
                rating_time[uid] = {}
            rating_time[uid][pid] = [rating, timestamp]

        train, test = [], []
        train_size = 0.8
        for uid in uid_pids.keys():
            uid_pids[uid].sort(key=lambda pid: rating_time[uid][pid][1])
            last_idx_train = int(len(uid_pids[uid]) * train_size)
            for i in range(last_idx_train):
                pid = uid_pids[uid][i]
                train.append([uid, pid, rating_time[uid][pid][0], rating_time[uid][pid][1], words[uid][pid]])
            for i in range(last_idx_train, len(uid_pids[uid])):
                pid = uid_pids[uid][i]
                test.append([uid, pid, rating_time[uid][pid][0], rating_time[uid][pid][1], words[uid][pid]])

        for split in ["train", "test"]:
            fileo = open(DATASET_DIR[dataset_name] + "/" + split + ".txt", 'w+')
            writer = csv.writer(fileo, delimiter="\t")
            curr_set = train if split == "train" else test
            for review in curr_set:
                writer.writerow(review)
            fileo.close()
            zip_file(DATASET_DIR[dataset_name] + "/" + split + ".txt")

if __name__ == '__main__':
    boolean = lambda x: (str(x).lower() == 'true')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=ML1M, help='One of {ML1M, LASTFM, CELL, CD, CLOTH, BEAUTY}')
    parser.add_argument('--amazon_year', type=str, default="2014", help='One of {2014, 2018}')
    parser.add_argument('--use_words', type=bool, default=True, help='Only for amazon dataset, uses the word extracted from the reviews as entities')
    args = parser.parse_args()

    if args.dataset == ML1M:
        ML1MDatasetMapper(args)
    elif args.dataset == LASTFM:
        LastFmDatasetMapper(args)
    elif args.dataset in AMAZON_DATASETS and args.amazon_year == "2018":
        Amazon2018DatasetMapper(args)
    elif args.dataset in AMAZON_DATASETS and args.amazon_year == "2014":
        Amazon2014DatasetMapper(args)
    else:
        print("Invalid dataset string, chose one between [ml1m, lastfm]")

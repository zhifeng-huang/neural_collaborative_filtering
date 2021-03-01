
import scipy.sparse as sp
import numpy as np
import json
import os

'''
card_type
card_subtype
card_level
'''

def has_video(detail):
    if detail['card_subtype'] == 'video':
        return 1
    else:
        return 0

def has_image(detail):
    if detail['card_subtype'] == 'image':
        return 1
    else:
        return 0

def has_link(detail):
    if detail['card_subtype'] == 'link':
        return 1
    else:
        return 0

def has_file(detail):
    if detail['card_subtype'] == 'file':
        return 1
    else: 
        return 0

def has_audio(detail):
    if detail['card_subtype'] == 'audio':
        return 1
    else: 
        return 0

def is_self_paced(detail):
    if detail['card_subtype'] == 'self_paced':
        return 1
    else:
        return 0

def is_weekly(detail):
    if detail['card_subtype'] == 'weekly':
        return 1
    else:
        return 0

def is_progressive_unlocking(detail):
    if detail['card_subtype'] == 'progressive_unlocking':
        return 1
    else:
        return 0

def is_project(detail):
    if detail['card_type'] == 'Project':
        return 1
    else:
        return 0

def is_poll(detail):
    if detail['card_type'] == 'Poll':
        return 1
    else:
        return 0

def is_scorm(detail):
    if detail['card_type'] == 'Scorm':
        return 1
    else:
        return 0

def is_course(detail):
    if detail['card_type'] == 'Course':
        return 1
    else:
        return 0

def is_video_stream(detail):
    if detail['card_type'] == 'Video Stream' and detail['card_subtype'] != 'role_play':
        return 1
    else:
        return 0

def is_role_play(detail):
    if detail['card_type'] == 'Video Stream' and detail['card_subtype'] == 'role_play':
        return 1
    else:
        return 0

def is_train(detail):
    if detail['card_type'] == 'Training':
        return 1
    else:
        return 0
        
def is_pathway(detail):
    if detail['card_type'] == 'Pathway':
        return 1
    else:
        return 0

def is_journey(detail):
    if detail['card_type'] == 'Journey':
        return 1
    else:
        return 0

def is_quiz(detail):
    if detail['card_type'] == 'Quiz':
        return 1
    else:
        return 0

def is_beginner(detail):
    if detail['card_level'] == 'Beginner':
        return 1
    else:
        return 0

def is_intermediate(detail):
    if detail['card_level'] == 'Intermediate':
        return 1
    else:
        return 0

def is_advanced(detail):
    if detail['card_level'] == 'Advanced':
        return 1
    else:
        return 0

def is_card_promoted(detail):
    if detail['is_card_promoted'] == '1.0' or detail['is_card_promoted'] == 'True':
        return 1
    else:
        return 0

def is_public(detail):
    if detail['card_is_public'] == '1.0' or detail['card_is_public'] == 'True':
        return 1
    else:
        return 0

def is_hidden(detail):
    if detail['card_hidden'] == '1.0' or detail['card_hidden'] == 'True':
        return 1
    else:
        return 0

def is_pathway_subcard(detail):
    if '/pathways/' in detail['referer']:
        return 1
    else:
        return 0

def is_journey_subcard(detail):
    if '/journey/' in detail['referer']:
        return 1
    else:
        return 0

def add_row(R, num_contents, card_detail_map, func):
    row = [0] * num_contents
    for card_id, detail in card_detail_map.items():
        card_i = detail['sid']
        row[card_i] = func(detail['detail'])
    R.append(row)

list_funcs = [has_video, has_image, has_link, has_file, has_audio, is_self_paced, is_weekly, is_progressive_unlocking, is_project,
                 is_poll, is_scorm, is_course, is_video_stream, is_role_play, is_train, is_pathway, is_journey, is_quiz, is_beginner, is_intermediate,
                 is_advanced, is_card_promoted, is_public, is_hidden, is_pathway_subcard, is_journey_subcard]
list_funcs_name = ['has_video', 'has_image', 'has_link', 'has_file', 'has_audio', 'is_self_paced', 'is_weekly', 'is_progressive_unlocking', 'is_project',
                 'is_poll', 'is_scorm', 'is_course', 'is_video_stream', 'is_role_play', 'is_train', 'is_pathway', 'is_journey', 'is_quiz', 'is_beginner', 'is_intermediate',
                 'is_advanced', 'is_card_promoted', 'is_public', 'is_hidden', 'is_pathway_subcard', 'is_journey_subcard']

class ED_Dataset(object):

    def __init__(self, path, skill_json=None, skill_map_json=None, use_attribute=False, use_rate=False):
        '''
        Constructor
        '''

        self.train_file = os.path.join(path, 'final_train.json')
        self.test_file = os.path.join(path, 'final_test.json')
        self.neg_file = os.path.join(path, 'negative.json')
        self.user_file = os.path.join(path, 'all_users.json')
        self.card_file = os.path.join(path, 'all_cards.json')
        self.rating_file = os.path.join(path, 'ratings.json')

        with open(self.user_file, 'r') as f:
            self.all_users = json.load(f)
        with open(self.card_file, 'r') as f:
            self.all_cards = json.load(f)

        with open(self.rating_file, 'r') as f:
            self.ratings = json.load(f)

        self.all_user_id_list = list(self.all_users['user_to_id'].keys())
        self.all_card_id_list = list(self.all_cards['card_to_id'].keys())

        self.num_users = len(self.all_user_id_list)
        self.num_items = len(self.all_card_id_list)

        self.user_to_id = self.all_users['user_to_id']
        self.card_to_id = self.all_cards['card_to_id']
        self.id_to_card = self.all_cards['id_to_card']

        self.ecl_map = {}
        for card_id, data in self.card_to_id.items():
            ecl_id = data['ecl_id']
            self.ecl_map[ecl_id] = card_id

        print(f'number of items: {self.num_items}')
        print(f'number of users: {self.num_users}')

        self.card_type_map = {}
        self.card_subtype_map = {}
        self.card_level_map = {}
        self.train_matrix = self.load_train_data_as_matrix(self.train_file, use_rate)

        assert self.num_users > 0
        assert self.num_items > 0
        assert (self.num_users, self.num_items) == self.train_matrix.shape
        
        self.test_ratings = self.load_test_data_as_list(self.test_file)
        self.test_negatives = self.load_negative_file(self.neg_file)
        assert len(self.test_ratings) == len(self.test_negatives)

        non_zeros = np.count_nonzero(self.train_matrix.toarray())
        sp = (1- non_zeros/(self.num_items*self.num_users))*100

        print(f'number of nonzeros in training matrix: {non_zeros}')
        print('sparsity of training matrix: %.2f%%' % (sp))

        self.skills = None

        self.skills_in_attribute = []

        if skill_json is not None:
            with open(skill_json, 'r') as f:
                data = json.load(f)
                self.skills = data['skills']

        self.skill_map = None 
        if skill_map_json is not None:
            with open(skill_map_json, 'r') as f:
                data = json.load(f)
                self.skill_map = data

        self.attribute_matrix = None
        if use_attribute:
            self.skill_to_id = None 

            if self.skills is not None and self.skill_map is not None:
                self.skill_to_id = {}

                for i in range(len(self.skills)):
                    self.skill_to_id[self.skills[i]] = i 

            self.attribute_matrix = self.create_attribute_matrix(self.skills, self.skill_map, self.skill_to_id)

            print(f'size of attribute matrix: {self.attribute_matrix.shape}')
            non_zeros = np.count_nonzero(self.attribute_matrix)
            sp = (1- non_zeros/(self.attribute_matrix.shape[0]*self.attribute_matrix.shape[1]))*100

            print(f'number of nonzeros in attribute matrix: {non_zeros}')
            print('sparsity of attribute matrix: %.2f%%' % (sp))
            # assert self.attribute_matrix.shape[0] == len(list_funcs) + len(self.skills)
            # assert self.attribute_matrix.shape[1] == self.num_items

    def get_ecl_id_from_card_id(self, card_ids):
        result = []
        for card_id in card_ids:
            detail = self.card_to_id[card_id]
            result.append(detail['ecl_id'])
        return result

    def get_attributes(self, detail):
        result = []
        for idx, fun in zip(range(len(list_funcs)), list_funcs):
            result.append((list_funcs_name[idx], fun(detail)))
        return result

    def create_attribute_matrix(self, skills, skill_map, skill_to_id):
        R = []

        # np.random.seed(0)
        # card_type_label = np.random.normal(0,0.5,len(self.card_type_map))
        # card_type_index_map = {t:i for t,i in zip(list(self.card_type_map.keys()), range(len(self.card_type_map)))}

        # np.random.seed(10)
        # card_subtype_label = np.random.normal(0,0.5,len(self.card_subtype_map))
        # card_subtype_index_map = {t:i for t, i in zip(list(self.card_subtype_map.keys()), range(len(self.card_subtype_map)))}

        # np.random.seed(100)
        # card_level_label = np.random.normal(0,0.5,len(self.card_level_map))
        # card_level_index_map = {t:i for t, i in zip(list(self.card_level_map.keys()), range(len(self.card_level_map)))}

        if skills is not None and skill_map is not None and skill_to_id is not None:
            for skill in self.skills:
                row = [0] * self.num_items

                card_list = self.skill_map[skill]
                for item in card_list:
                    ecl_id = item[0]
                    score = item[1]

                    if ecl_id in self.ecl_map:
                        card_id = self.ecl_map[ecl_id]
                        card_i = self.card_to_id[card_id]['sid']

                        row[card_i] = 1
                c = np.count_nonzero(row)
                if c > 10:        
                    R.append(row)
                    self.skills_in_attribute.append(skill)
                # else:
                    # print(f'{skill} has no card')

            sum = np.array(R).sum(axis=0)
            nonzeros = np.count_nonzero(sum)
            print(f'number of cards without skills: {self.num_items - nonzeros}')
            # print(self.skills_in_attribute)

        # row = [0] * self.num_items
        # for card_id, detail in self.card_to_id.items():
        #     card_i = detail['sid']
        #     d = detail['detail']
        #     i = card_type_index_map[d['card_type']]
        #     row[card_i] = card_type_label[i]
        # R.append(row)

        # row = [0] * self.num_items
        # for card_id, detail in self.card_to_id.items():
        #     card_i = detail['sid']
        #     d = detail['detail']
        #     i = card_subtype_index_map[d['card_subtype']]
        #     row[card_i] = card_subtype_label[i]
        # R.append(row)

        # row = [0] * self.num_items
        # for card_id, detail in self.card_to_id.items():
        #     card_i = detail['sid']
        #     d = detail['detail']
        #     i = card_level_index_map[d['card_level']]
        #     row[card_i] = card_level_label[i]
        # R.append(row)

        num_contents = self.num_items

        for fun in list_funcs:
            add_row(R, num_contents, self.card_to_id, fun)
        
        return np.array(R)

    def load_user_events_dis(self, user_id):
        with open(self.train_file, 'r') as f:
            train_data = json.load(f)
        with open(self.test_file, 'r') as f:
            test_data = json.load(f)

        result = {}

        events = train_data[user_id]
        for event in events:
            e = event['event']
            if e in result:
                result[e] += 1
            else:
                result[e] = 1
        return result

    def load_user_to_card(self):
        with open(self.train_file, 'r') as f:
            train_data = json.load(f)
        with open(self.test_file, 'r') as f:
            test_data = json.load(f)

        result = {}

        for user_id, events in train_data.items():
            cards = [e['card_id'] for e in events]
            result[user_id] = cards
        
        for user_id, e in test_data.items():
            elist = result[user_id]
            elist.append(e['card_id'])
        
        return result

    def get_card_ids(self, indices):
        result = []
        for i in range(len(indices)):
            card_id = self.id_to_card[str(indices[i])]['card_id']
            result.append(card_id)
        return result

    def get_user_index(self, user_id):
        return self.user_to_id[user_id]

    def get_cards_detail(self, indices, filter=False):
        result = []
        for i in range(len(indices)):
            card_id = self.id_to_card[str(indices[i])]['card_id']
            detail = self.card_to_id[card_id]['detail']
            result.append(detail)
        return result

    def load_negative_file(self, filename):
        negative_list = []
        with open(filename, 'r') as f:
            data = json.load(f)

        for user_id, negs in data.items():
            user = self.user_to_id[user_id]
            negs_ids = []
            for card_id in negs:
                item = self.card_to_id[card_id]['sid']
                negs_ids.append(item)
            negative_list.append(negs_ids)
        return negative_list

    def load_test_data_as_list(self, filename):
        test_data = []
        with open(filename, 'r') as f:
            data = json.load(f)
        for user_id, event in data.items():
            user = self.user_to_id[user_id]

            card_id = event['card_id']
            item = self.card_to_id[card_id]['sid']
            test_data.append([user, item])
        return test_data

    def load_train_data_as_matrix(self, filename, use_rate):
        with open(filename, 'r') as f:
            data = json.load(f)
        mat = sp.dok_matrix((self.num_users, self.num_items), dtype=np.float32)
        for user_id, events in data.items():
            user = self.user_to_id[user_id]
            for event in events:
                card_type = event['card_type']
                card_subtype = event['card_subtype']
                card_level = event['card_level']

                self.card_type_map[card_type] = True
                self.card_subtype_map[card_subtype] = True
                self.card_level_map[card_level] = True

                card_id = event['card_id']
                item = self.card_to_id[card_id]['sid']
                if use_rate:
                    mat[user, item] = self.ratings[event['event']]
                else:
                    mat[user, item] = 1 # we assume the rating is 1 or 0
        return mat




        

        

# This file contains all of the classes for each individual page of the survey

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import glob
import xml.etree.ElementTree as ET
import pandas as pd
plt.rcParams['figure.figsize'] = (10, 10)
from utils import *
from define import *
# import define

class Page0Processor:
    def __init__(self, xml_path):
        df = xml2csv(xml_path)

        self.all_checkbox = df.copy()

        self.all_checkbox_numpy =np.array(self.all_checkbox[['xmin', 'ymin', 'xmax', 'ymax']])
        question_of_row = [row.split('_') for row in self.all_checkbox['class']]
        self.question_of_row = [row[0] if len(row) == 2 else '_'.join(row[:3]) if len(row) == 4 else '_'.join(row[:2]) for row in question_of_row]
        
    def process(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        extractor = CheckboxExtractor()
        checkboxes = extractor.detect_checkbox(gray)    
        tick_list = list(map(lambda x: extractor.is_ticked(gray[x[1]:x[3], x[0]:x[2]]), checkboxes))
        checkboxes = np.hstack([checkboxes, np.array(tick_list).reshape(-1,1)]) # [x1,y1,x2,y2,is_ticked]

        row_ids = []
        for i in range(len(checkboxes)):
            checkbox_coor = checkboxes[i:i+1, :]
            _iou = np_vec_no_jit_iou(checkbox_coor[:,:4], self.all_checkbox_numpy)
            row_ids += [np.argmax(_iou)]
        
        result_df = pd.DataFrame()

        _id_offset = 0

        for _question_id in [1,2]:
            question_name = f'{_question_id}'

            _checkbox_ids = list(filter(lambda x: ( str(_question_id) == self.question_of_row[row_ids[x]] ) and (checkboxes[x][4] == 1), range(len(checkboxes))))

            answers = []
            if len(_checkbox_ids) > 0:
                for _checkbox_id in _checkbox_ids:
                    answers += [ANSWER_TEXT[str(_question_id)][row_ids[_checkbox_id] - _id_offset]]
            result_df[question_name] = [', '.join(answers)]
            _id_offset += len(ANSWER_TEXT[str(_question_id)])



        return result_df


class Page1Processor:
    def __init__(self, xml_path):
        df = xml2csv(xml_path)

        self.all_checkbox = df.copy()

        self.all_checkbox_numpy =np.array(self.all_checkbox[['xmin', 'ymin', 'xmax', 'ymax']])
        question_of_row = [row.split('_') for row in self.all_checkbox['class']]
        self.question_of_row = [row[0] if len(row) == 2 else '_'.join(row[:3]) if len(row) == 4 else '_'.join(row[:2]) for row in question_of_row]

        
    def process(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        extractor = CheckboxExtractor()
        checkboxes = extractor.detect_checkbox(gray)    
        tick_list = list(map(lambda x: extractor.is_ticked(gray[x[1]:x[3], x[0]:x[2]]), checkboxes))
        checkboxes = np.hstack([checkboxes, np.array(tick_list).reshape(-1,1)]) # [x1,y1,x2,y2,is_ticked]

        row_ids = []
        for i in range(len(checkboxes)):
            checkbox_coor = checkboxes[i:i+1, :]
            _iou = np_vec_no_jit_iou(checkbox_coor[:,:4], self.all_checkbox_numpy)
            row_ids += [np.argmax(_iou)]
        
        result_df = pd.DataFrame()

        _id_offset = 0

        for _question_id in [8,9,10,11]:
            question_name = f'{_question_id}'

            _checkbox_ids = list(filter(lambda x: ( str(_question_id) == self.question_of_row[row_ids[x]] ) and (checkboxes[x][4] == 1), range(len(checkboxes))))

            answers = []
            if len(_checkbox_ids) > 0:
                for _checkbox_id in _checkbox_ids:
                    answers += [ANSWER_TEXT[str(_question_id)][row_ids[_checkbox_id] - _id_offset]]
            result_df[question_name] = [', '.join(answers)]
            _id_offset += len(ANSWER_TEXT[str(_question_id)])


        return result_df

class Page2Processor:
    def __init__(self, xml_path):
        df = xml2csv(xml_path)

        self.all_checkbox = df.copy()

        self.all_checkbox_numpy =np.array(self.all_checkbox[['xmin', 'ymin', 'xmax', 'ymax']])
        question_of_row = [row.split('_') for row in self.all_checkbox['class']]
        self.question_of_row = [row[0] if len(row) == 2 else '_'.join(row[:3]) if len(row) == 4 else '_'.join(row[:2]) for row in question_of_row]

        
    def process(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        extractor = CheckboxExtractor()
        checkboxes = extractor.detect_checkbox(gray)    
        tick_list = list(map(lambda x: extractor.is_ticked(gray[x[1]:x[3], x[0]:x[2]]), checkboxes))
        checkboxes = np.hstack([checkboxes, np.array(tick_list).reshape(-1,1)]) # [x1,y1,x2,y2,is_ticked]

        row_ids = []
        for i in range(len(checkboxes)):
            checkbox_coor = checkboxes[i:i+1, :]
            _iou = np_vec_no_jit_iou(checkbox_coor[:,:4], self.all_checkbox_numpy)
            row_ids += [np.argmax(_iou)]
        
        result_df = pd.DataFrame()

        _id_offset = 0

        for _question_id in [18,19,20]:
            question_name = f'{_question_id}'

            _checkbox_ids = list(filter(lambda x: ( str(_question_id) == self.question_of_row[row_ids[x]] ) and (checkboxes[x][4] == 1), range(len(checkboxes))))

            answers = []
            if len(_checkbox_ids) > 0:
                for _checkbox_id in _checkbox_ids:
                    answers += [ANSWER_TEXT[str(_question_id)][row_ids[_checkbox_id] - _id_offset]]
            result_df[question_name] = [', '.join(answers)]
            _id_offset += len(ANSWER_TEXT[str(_question_id)])


        return result_df

class Page3Processor:
    def __init__(self, xml_path):
        df = xml2csv(xml_path)

        self.all_checkbox = df.copy()

        self.all_checkbox_numpy =np.array(self.all_checkbox[['xmin', 'ymin', 'xmax', 'ymax']])
        question_of_row = [row.split('_') for row in self.all_checkbox['class']]
        self.question_of_row = [row[0] if len(row) == 2 else '_'.join(row[:3]) if len(row) == 4 else '_'.join(row[:2]) for row in question_of_row]

        
    def process(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        extractor = CheckboxExtractor()
        checkboxes = extractor.detect_checkbox(gray)    
        tick_list = list(map(lambda x: extractor.is_ticked(gray[x[1]:x[3], x[0]:x[2]]), checkboxes))
        checkboxes = np.hstack([checkboxes, np.array(tick_list).reshape(-1,1)]) # [x1,y1,x2,y2,is_ticked]

        row_ids = []
        for i in range(len(checkboxes)):
            checkbox_coor = checkboxes[i:i+1, :]
            _iou = np_vec_no_jit_iou(checkbox_coor[:,:4], self.all_checkbox_numpy)
            row_ids += [np.argmax(_iou)]

        result_df = pd.DataFrame()

        _id_offset = 0

        for _question_id in range(21, 29):
            question_name = f'{_question_id}'

            _checkbox_ids = list(filter(lambda x: ( str(_question_id) == self.question_of_row[row_ids[x]] ) and (checkboxes[x][4] == 1), range(len(checkboxes))))

            answers = []
            if len(_checkbox_ids) > 0:
                for _checkbox_id in _checkbox_ids:
                    answers += [ANSWER_TEXT[str(_question_id)][row_ids[_checkbox_id] - _id_offset]]
            result_df[question_name] = [', '.join(answers)]
            _id_offset += len(ANSWER_TEXT[str(_question_id)])
            if _question_id == 22:
                answers = []
                question_name = f'{_question_id}_1'
                _checkbox_ids = list(filter(lambda x: ( self.question_of_row[row_ids[x]] == question_name ) and (checkboxes[x][4] == 1), range(len(checkboxes))))
                for _checkbox_id in _checkbox_ids:
                    answers += [ANSWER_TEXT[question_name][row_ids[_checkbox_id] - _id_offset]]
                result_df[question_name.replace('_', '.')] = [', '.join(answers)]
                _id_offset += len(ANSWER_TEXT[question_name])
        return result_df

class Page4Processor:
    def __init__(self, xml_path):
        df = xml2csv(xml_path)

        self.all_checkbox = df.copy()

        self.all_checkbox_numpy =np.array(self.all_checkbox[['xmin', 'ymin', 'xmax', 'ymax']])
        question_of_row = [row.split('_') for row in self.all_checkbox['class']]
        self.question_of_row = [row[0] if len(row) == 2 else '_'.join(row[:3]) if len(row) == 4 else '_'.join(row[:2]) for row in question_of_row]

        
    def process(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        extractor = CheckboxExtractor()
        checkboxes = extractor.detect_checkbox(gray)    
        tick_list = list(map(lambda x: extractor.is_ticked(gray[x[1]:x[3], x[0]:x[2]]), checkboxes))
        checkboxes = np.hstack([checkboxes, np.array(tick_list).reshape(-1,1)]) # [x1,y1,x2,y2,is_ticked]

        row_ids = []
        for i in range(len(checkboxes)):
            checkbox_coor = checkboxes[i:i+1, :]
            _iou = np_vec_no_jit_iou(checkbox_coor[:,:4], self.all_checkbox_numpy)
            row_ids += [np.argmax(_iou)]
        
        result_df = pd.DataFrame()

        _id_offset = 0

        for _question_id in [29,30]:
            question_name = f'{_question_id}'

            _checkbox_ids = list(filter(lambda x: ( str(_question_id) == self.question_of_row[row_ids[x]] ) and (checkboxes[x][4] == 1), range(len(checkboxes))))

            answers = []
            if len(_checkbox_ids) > 0:
                for _checkbox_id in _checkbox_ids:
                    answers += [ANSWER_TEXT[str(_question_id)][row_ids[_checkbox_id] - _id_offset]]
            result_df[question_name] = [', '.join(answers)]
            _id_offset += len(ANSWER_TEXT[str(_question_id)])
            if _question_id == 29:
                answers = []
                for sub_id in [2,4,5,6,7]:
                    question_name = f'{_question_id}_1_{sub_id}'
                    _checkbox_ids = list(filter(lambda x: ( self.question_of_row[row_ids[x]] == question_name ) and (checkboxes[x][4] == 1), range(len(checkboxes))))
                    for _checkbox_id in _checkbox_ids:
                        answers += [ANSWER_TEXT[question_name][row_ids[_checkbox_id] - _id_offset]]
                    result_df[question_name.replace('_', '.')] = [', '.join(answers)]
                    _id_offset += len(ANSWER_TEXT[question_name])
        return result_df

class Page5Processor:
    def __init__(self, xml_path):
        df = xml2csv(xml_path)

        self.all_checkbox = df.copy()

        self.all_checkbox_numpy =np.array(self.all_checkbox[['xmin', 'ymin', 'xmax', 'ymax']])
        question_of_row = [row.split('_') for row in self.all_checkbox['class']]
        self.question_of_row = [row[0] if len(row) == 2 else '_'.join(row[:3]) if len(row) == 4 else '_'.join(row[:2]) for row in question_of_row]

        
    def process(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        extractor = CheckboxExtractor()
        checkboxes = extractor.detect_checkbox(gray)    
        tick_list = list(map(lambda x: extractor.is_ticked(gray[x[1]:x[3], x[0]:x[2]]), checkboxes))
        checkboxes = np.hstack([checkboxes, np.array(tick_list).reshape(-1,1)]) # [x1,y1,x2,y2,is_ticked]

        row_ids = []
        for i in range(len(checkboxes)):
            checkbox_coor = checkboxes[i:i+1, :]
            _iou = np_vec_no_jit_iou(checkbox_coor[:,:4], self.all_checkbox_numpy)
            row_ids += [np.argmax(_iou)]
        

        result_df = pd.DataFrame()

        _id_offset = 0

        for _question_id in [31,33,34,35,36]:
            question_name = f'{_question_id}'

            _checkbox_ids = list(filter(lambda x: ( str(_question_id) == self.question_of_row[row_ids[x]] ) and (checkboxes[x][4] == 1), range(len(checkboxes))))

            answers = []
            if len(_checkbox_ids) > 0:
                for _checkbox_id in _checkbox_ids:
                    answers += [ANSWER_TEXT[str(_question_id)][row_ids[_checkbox_id] - _id_offset]]
            result_df[question_name] = [', '.join(answers)]
            _id_offset += len(ANSWER_TEXT[str(_question_id)])
            if _question_id in [31, 33]:
                answers = []
                question_name = f'{_question_id}_1'
                _checkbox_ids = list(filter(lambda x: ( self.question_of_row[row_ids[x]] == question_name ) and (checkboxes[x][4] == 1), range(len(checkboxes))))
                for _checkbox_id in _checkbox_ids:
                    answers += [ANSWER_TEXT[question_name][row_ids[_checkbox_id] - _id_offset]]
                result_df[question_name.replace('_', '.')] = [', '.join(answers)]
                _id_offset += len(ANSWER_TEXT[question_name])
        return result_df

class Page6Processor:
    def __init__(self, xml_path):
        df = xml2csv(xml_path)

        self.all_checkbox = df.copy()

        self.all_checkbox_numpy =np.array(self.all_checkbox[['xmin', 'ymin', 'xmax', 'ymax']])
        question_of_row = [row.split('_') for row in self.all_checkbox['class']]
        self.question_of_row = [row[0] if len(row) == 2 else '_'.join(row[:3]) if len(row) == 4 else '_'.join(row[:2]) for row in question_of_row]

        
    def process(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        extractor = CheckboxExtractor()
        checkboxes = extractor.detect_checkbox(gray)    
        tick_list = list(map(lambda x: extractor.is_ticked(gray[x[1]:x[3], x[0]:x[2]]), checkboxes))
        checkboxes = np.hstack([checkboxes, np.array(tick_list).reshape(-1,1)]) # [x1,y1,x2,y2,is_ticked]

        row_ids = []
        for i in range(len(checkboxes)):
            checkbox_coor = checkboxes[i:i+1, :]
            _iou = np_vec_no_jit_iou(checkbox_coor[:,:4], self.all_checkbox_numpy)
            row_ids += [np.argmax(_iou)]
        

        result_df = pd.DataFrame()

        _id_offset = 0

        result_df = pd.DataFrame()

        _id_offset = 0

        for _question_id in range(37, 44):
            question_name = f'{_question_id}'

            _checkbox_ids = list(filter(lambda x: ( str(_question_id) == self.question_of_row[row_ids[x]] ) and (checkboxes[x][4] == 1), range(len(checkboxes))))

            answers = []
            if len(_checkbox_ids) > 0:
                for _checkbox_id in _checkbox_ids:
                    answers += [ANSWER_TEXT[str(_question_id)][row_ids[_checkbox_id] - _id_offset]]
            result_df[question_name] = [', '.join(answers)]
            _id_offset += len(ANSWER_TEXT[str(_question_id)])
            if _question_id == 40:
                answers = []
                question_name = f'40_1'
                _checkbox_ids = list(filter(lambda x: ( self.question_of_row[row_ids[x]] == question_name ) and (checkboxes[x][4] == 1), range(len(checkboxes))))
                for _checkbox_id in _checkbox_ids:
                    answers += [ANSWER_TEXT[question_name][row_ids[_checkbox_id] - _id_offset]]
                result_df[question_name.replace('_', '.')] = [', '.join(answers)]
                _id_offset += len(ANSWER_TEXT[question_name])

        return result_df

class Page7Processor:
    def __init__(self, xml_path):
        df = xml2csv(xml_path)

        # self.all_checkbox = df.iloc[range(2, 12)]
        self.all_checkbox = df.copy()

        self.all_checkbox_numpy =np.array(self.all_checkbox[['xmin', 'ymin', 'xmax', 'ymax']])
        question_of_row = [row.split('_') for row in self.all_checkbox['class']]
        self.question_of_row = [row[0] if len(row) == 2 else '_'.join(row[:3]) if len(row) == 4 else '_'.join(row[:2]) for row in question_of_row]

        
    def process(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        extractor = CheckboxExtractor()
        checkboxes = extractor.detect_checkbox(gray)    
        tick_list = list(map(lambda x: extractor.is_ticked(gray[x[1]:x[3], x[0]:x[2]]), checkboxes))
        checkboxes = np.hstack([checkboxes, np.array(tick_list).reshape(-1,1)]) # [x1,y1,x2,y2,is_ticked]

        row_ids = []
        for i in range(len(checkboxes)):
            checkbox_coor = checkboxes[i:i+1, :]
            _iou = np_vec_no_jit_iou(checkbox_coor[:,:4], self.all_checkbox_numpy)
            row_ids += [np.argmax(_iou)]
        

        result_df = pd.DataFrame()

        _id_offset = 0

        for _question_id in [47,48]:
            question_name = f'{_question_id}'

            _checkbox_ids = list(filter(lambda x: ( str(_question_id) == self.question_of_row[row_ids[x]] ) and (checkboxes[x][4] == 1), range(len(checkboxes))))

            answers = []
            if len(_checkbox_ids) > 0:
                for _checkbox_id in _checkbox_ids:
                    answers += [ANSWER_TEXT[str(_question_id)][row_ids[_checkbox_id] - _id_offset]]
            result_df[question_name] = [', '.join(answers)]
            _id_offset += len(ANSWER_TEXT[str(_question_id)])
            if _question_id == 48:
                for sub_id in range(1,4):
                    answers = []
                    question_name = f'48_1_{sub_id}'
                    _checkbox_ids = list(filter(lambda x: ( self.question_of_row[row_ids[x]] == question_name ) and (checkboxes[x][4] == 1), range(len(checkboxes))))
                    for _checkbox_id in _checkbox_ids:
                        answers += [ANSWER_TEXT[question_name][row_ids[_checkbox_id] - _id_offset]]
                    result_df[question_name.replace('_', '.')] = [', '.join(answers)]
                    _id_offset += len(ANSWER_TEXT[question_name])
        return result_df

class Page8Processor:
    def __init__(self, xml_path):
        df = xml2csv(xml_path)

        # self.all_checkbox = df.iloc[list(range(5, 9)) + list(range(10, 22))]
        self.all_checkbox = df.copy()
        self.all_checkbox_numpy =np.array(self.all_checkbox[['xmin', 'ymin', 'xmax', 'ymax']])
        question_of_row = [row.split('_') for row in self.all_checkbox['class']]
        self.question_of_row = [row[0] if len(row) == 2 else '_'.join(row[:3]) if len(row) == 4 else '_'.join(row[:2]) for row in question_of_row]

        
    def process(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        extractor = CheckboxExtractor()
        checkboxes = extractor.detect_checkbox(gray)    
        tick_list = list(map(lambda x: extractor.is_ticked(gray[x[1]:x[3], x[0]:x[2]]), checkboxes))
        checkboxes = np.hstack([checkboxes, np.array(tick_list).reshape(-1,1)]) # [x1,y1,x2,y2,is_ticked]

        row_ids = []
        for i in range(len(checkboxes)):
            checkbox_coor = checkboxes[i:i+1, :]
            _iou = np_vec_no_jit_iou(checkbox_coor[:,:4], self.all_checkbox_numpy)
            row_ids += [np.argmax(_iou)]
        

        result_df = pd.DataFrame()

        _id_offset = 0

        for _question_id in list(range(49,52)) + [53,54]:
            question_name = f'{_question_id}'

            _checkbox_ids = list(filter(lambda x: ( str(_question_id) == self.question_of_row[row_ids[x]] ) and (checkboxes[x][4] == 1), range(len(checkboxes))))

            answers = []
            if len(_checkbox_ids) > 0:
                for _checkbox_id in _checkbox_ids:
                    answers += [ANSWER_TEXT[str(_question_id)][row_ids[_checkbox_id] - _id_offset]]
            result_df[question_name] = [', '.join(answers)]
            _id_offset += len(ANSWER_TEXT[str(_question_id)])
            if _question_id == 49:
                answers = []
                question_name = '49.4.2'
                _checkbox_ids = list(filter(lambda x: ( self.question_of_row[row_ids[x]] == '49_4_2' ) and (checkboxes[x][4] == 1), range(len(checkboxes))))
                for _checkbox_id in _checkbox_ids:
                    answers += [ANSWER_TEXT['49_4_2'][row_ids[_checkbox_id] - _id_offset]]
                result_df[question_name] = [', '.join(answers)]
                _id_offset += len(ANSWER_TEXT['49_4_2'])
        return result_df

class Page9Processor:
    def __init__(self, xml_path):
        df = xml2csv(xml_path)

        self.all_checkbox = df.copy()

        self.all_checkbox_numpy =np.array(self.all_checkbox[['xmin', 'ymin', 'xmax', 'ymax']])
        question_of_row = [row.split('_') for row in self.all_checkbox['class']]
        self.question_of_row = [row[0] if len(row) == 2 else '_'.join(row[:3]) if len(row) == 4 else '_'.join(row[:2]) for row in question_of_row]
        
    def process(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        extractor = CheckboxExtractor()
        checkboxes = extractor.detect_checkbox(gray)    
        tick_list = list(map(lambda x: extractor.is_ticked(gray[x[1]:x[3], x[0]:x[2]]), checkboxes))
        checkboxes = np.hstack([checkboxes, np.array(tick_list).reshape(-1,1)]) # [x1,y1,x2,y2,is_ticked]

        row_ids = []
        for i in range(len(checkboxes)):
            checkbox_coor = checkboxes[i:i+1, :]
            _iou = np_vec_no_jit_iou(checkbox_coor[:,:4], self.all_checkbox_numpy)
            row_ids += [np.argmax(_iou)]
        
        result_df = pd.DataFrame()

        _id_offset = 0

        for _question_id in [55]:
            question_name = f'{_question_id}'

            _checkbox_ids = list(filter(lambda x: ( str(_question_id) == self.question_of_row[row_ids[x]] ) and (checkboxes[x][4] == 1), range(len(checkboxes))))

            answers = []
            if len(_checkbox_ids) > 0:
                for _checkbox_id in _checkbox_ids:
                    answers += [ANSWER_TEXT[str(_question_id)][row_ids[_checkbox_id] - _id_offset]]
            result_df[question_name] = [', '.join(answers)]
            _id_offset += len(ANSWER_TEXT[str(_question_id)])
        return result_df

    
class Page10Processor:
    def __init__(self, xml_path):
        df = xml2csv(xml_path)

        self.all_columns = df.iloc[[7,8,9]]
        self.all_columns_np = np.array(self.all_columns[['xmin', 'ymin', 'xmax', 'ymax']])

        self.all_rows = df.iloc[range(10, 33)]
        self.question_of_rows = [row.split('_')[0] for row in self.all_rows['class']]
        self.all_rows_np = np.array(self.all_rows[['xmin', 'ymin', 'xmax', 'ymax']])
        
    def process(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        extractor = CheckboxExtractor()
        checkboxes = extractor.detect_checkbox(gray)    
        tick_list = list(map(lambda x: extractor.is_ticked(gray[x[1]:x[3], x[0]:x[2]]), checkboxes))
        checkboxes = np.hstack([checkboxes, np.array(tick_list).reshape(-1,1)]) # [x1,y1,x2,y2,is_ticked]

        column_ids  = []
        row_ids = []
        for i in range(len(checkboxes)):
            x1,y1,x2,y2,_is_tick = checkboxes[i]
            _column_id = np.where((self.all_columns_np[:,0] < x1) & (self.all_columns_np[:,2] > x2))[0][0]
            column_ids += [_column_id]

            checkbox_coor = checkboxes[i:i+1, :]
            _iou = np_vec_no_jit_iou(checkbox_coor[:,:4], self.all_rows_np)
            row_ids += [np.argmax(_iou)]

        _id_offset = 0
        result_df = pd.DataFrame()
        for _question_id in range(57, 64):
            for _column_id in range(3):
                question_name = f'{_question_id}.{_column_id+1}'

                _checkbox_ids = list(filter(lambda x: ( int(self.question_of_rows[row_ids[x]]) == _question_id ) and (column_ids[x] == _column_id) and (checkboxes[x][4] == 1), range(len(checkboxes)) ))
                answers = []
                if len(_checkbox_ids) > 0:
                    for _checkbox_id in _checkbox_ids:
                        answers += [ANSWER_TEXT[_question_id][row_ids[_checkbox_id] - _id_offset]]
                result_df[question_name] = [', '.join(answers)]

            _id_offset += len(ANSWER_TEXT[_question_id])

        return result_df

    

    


    


    

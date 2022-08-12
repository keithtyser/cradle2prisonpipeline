# This file contains all of the helper functions used for the checkbox detection

import numpy as np
import pandas as pd
import cv2
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
def get_threshold(areas):
    areas = np.array(sorted(areas))
    threshold = 50
    groups = [[areas[0]]]
    diffs = areas[1:] - areas[:-1]
    for i, dif in enumerate(diffs):
        if dif < threshold:
            groups[-1] += [areas[i+1]]
        else:
            groups += [[areas[i+1]]]
    lens = list(map(lambda x: len(x), groups))
    ret = np.mean(groups[np.argmax(lens)])
    return ret

class CheckboxExtractor:
  def __init__(self):
    pass
  
  def find_same_block(self, checkbox, groups):
    for _id in range(len(groups)-1, -1, -1):
      if np.abs(checkbox[0] - np.array(groups[_id][1:])[:,0].mean()) <= 30:
        return _id
    return None

  def has_numbers(self, inputString):
    return any(char.isdigit() for char in inputString)

  def is_question(self, text, group_type=1):
    sub_condic = False
    if group_type == 2:
      sub_condic  = '?' in text
    return self.has_numbers(text[:7]) or '.' in text[:7] or sub_condic

  def check_above_is_question(self, checkbox, lines, line_have_checkbox):
    checkbox_line_num = checkbox[6]
    for i in range(1,4):
      search_line_num = int(checkbox_line_num) - i
      if search_line_num >= 0 and search_line_num not in line_have_checkbox:
        if self.is_question(lines[1][ search_line_num ]) and ( lines[0][search_line_num][0] - 30 < checkbox[0] ) :
          return True
      else: break

    return False

  def find_question_for_group(self, group, lines, line_have_checkbox):
    group_type = group[0]
    top_checkbox = group[1]
    checkbox_line_num = top_checkbox[6]
    question_components = []
    for i in range(1,4):
      search_line_num = int(checkbox_line_num) - i
      if search_line_num >= 0 and search_line_num not in line_have_checkbox:
        question_components += [ lines[1][ search_line_num ] ]
        if self.is_question(lines[1][ search_line_num ], group_type=group_type):
          break
      else: break
    
    return question_components[::-1] 

  def group_checkbox(self, checkboxes, lines):
    
    checkboxes = np.array(sorted(checkboxes, key=lambda x: x[6])) # sort by line
    line_have_checkbox = list(checkboxes[:,6])
    finish_groups = []
    unfinish_groups = []

    for checkbox in checkboxes:

      # print('-----------------------------------------------------------------------')
      # print(f'Checkbox at line: {checkbox[6]}')
      # print(len(unfinish_groups))

      if len(unfinish_groups) == 2:
        unfinish_groups = sorted(unfinish_groups, key=lambda x: x[0])
      if checkbox[6] - 1 not in line_have_checkbox: # Above is text

        group_id = self.find_same_block(checkbox, unfinish_groups)
        if group_id is None: # No unfinish element has same align
          # Create a new unfinish group
          group = [ len(unfinish_groups) + 1, checkbox]
          unfinish_groups += [ group ]

        else:
          if self.check_above_is_question(checkbox, lines, line_have_checkbox):
            # Move the old checkbox to finish_froup and create new element for unfinish_groups
            finish_groups += [unfinish_groups.pop(group_id)]

            # print(f'Moved 1: \n{finish_groups[-1]}')

            if len(unfinish_groups) == 1 and unfinish_groups[0][0] == 2:
              finish_groups += [unfinish_groups.pop(0)]

              # print(f'Moved 2: \n{finish_groups[-1]}')

            group = [ len(unfinish_groups) + 1, checkbox]
            unfinish_groups += [ group ]
            
          else:
            unfinish_groups[ group_id ] += [checkbox]

      else: # Above is not text
        try:
          if np.abs(checkbox[0] - np.array(unfinish_groups[-1][1:])[:,0].mean()) <= 30: # nearest unfinish group has same align
            unfinish_groups[ -1 ] += [checkbox]
          else:
            try:
              finish_groups += [unfinish_groups.pop(-1)]
              # print(f'Moved 3: \n{finish_groups[-1]}')
              unfinish_groups[ -1 ] += [checkbox]
            except:
              # print('Some error might occur here')
              pass
        except: 
          pass
          # print('Bug log')
          # print(len(unfinish_groups))
          # print(checkbox)
    finish_groups += unfinish_groups
    return finish_groups, unfinish_groups


  def is_correct_area(self, contour, expected_area=625, tolerance=200):
   area = cv2.contourArea(contour)
   return abs(area - expected_area) <= tolerance

  def are_bounding_dimensions_correct(self, contour, expected_area=625, tolerance=200, squareness_tolerance=15):
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    return abs(area - expected_area) <= tolerance and abs(w - h) <= squareness_tolerance
    
  def is_contour_square(self, contour, contour_tolerance=0.0015, square_side=25, area_tolerance=200):
    expected_area = square_side * square_side
    area = cv2.contourArea(contour)
    template = np.array([[[0, 0]], [[0, 1]], [[1, 1]], [[1, 0]]], dtype=np.int32)
    return cv2.matchShapes(template, contour, 1, 0.0) <= contour_tolerance and abs(area - expected_area) <= area_tolerance

  def detect_checkbox(self, gray_img, threshold=None):
    thresh_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    bin_thresh_img = 255 - thresh_img
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,20))
    vertical = cv2.morphologyEx(bin_thresh_img, cv2.MORPH_OPEN, vertical_kernel, iterations=1)

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20,1))
    horizontal = cv2.morphologyEx(bin_thresh_img, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)

    rets = []
    temp_new = vertical | horizontal

    temp_kernel = np.ones((2,2), np.uint8)
    temp_new=cv2.dilate(temp_new,temp_kernel,iterations=1)

    _, labels, stats,_ = cv2.connectedComponentsWithStats(~temp_new, connectivity=8, ltype=cv2.CV_32S)
    # plt.imshow(labels)
    # threshold = np.mean(stats[2:][-3:])
    areas = list(filter(lambda x: 400 <= x <= 1000, stats[2:][:,4]))
    if threshold is None:
      threshold = get_threshold(areas)
    # print(f'Threshold: {threshold}')
    # ticks = []
    # print(type(stats))
    for x,y,w,h,area in stats[2:]:
      # print(area)
      # if area < 400: continue
      if np.abs(area - threshold) > 50: continue
      rets += [[x,y,w,h]]

    rets = np.array(rets)
    # print(rets.shape)
    # print('hello world')
    rets[:, 2] = rets[:, 0] + rets[:, 2]
    rets[:, 3] = rets[:, 1] + rets[:, 3]
    return rets

  def detect_text(self, gray_img):
    thresh_img = cv2.threshold(gray_img, 100, 255, cv2.THRESH_BINARY)[1]
    details = pytesseract.image_to_data(thresh_img, output_type=Output.DICT, config='--oem 3 --psm 6', lang='eng')
    total_boxes = len(details['text'])
    rets = []
    for sequence_number in range(total_boxes):
      if int(details['conf'][sequence_number]) >10:
        (x, y, w, h) = (details['left'][sequence_number], details['top'][sequence_number], details['width'][sequence_number],  details['height'][sequence_number])
        # ret = cv2.rectangle(ret, (x, y), (x + w, y + h), (0, 255, 0), 2)
        rets += [[x,y,w,h]]
    
    rets = np.array(rets)
    rets[:, 2] = rets[:, 0] + rets[:, 2]
    rets[:, 3] = rets[:, 1] + rets[:, 3]
    return rets

  def detect_text_line(self, gray_img):
    H,W = gray_img.shape
    thresh_img = cv2.threshold(gray_img, 100, 255, cv2.THRESH_BINARY)[1]
    details = pytesseract.image_to_data(thresh_img, output_type=Output.DICT, config='--oem 3 --psm 6', lang='eng')
    line_nums = np.array(details['line_num'])
    l_list = np.unique(line_nums)
    coords = []
    texts = []
    for l in l_list:
      ids = np.where(line_nums == l)
      x1 = np.array(details['left'])[ids]
      y1 = np.array(details['top'])[ids]
      w = np.array(details['width'])[ids]
      h = np.array(details['height'])[ids]
      text = ' '.join(list(np.array(details['text'])[ids]))
      remove_ids = np.where( (w == W) & (h == H) )

      x1 = np.delete(x1, remove_ids)
      y1 = np.delete(y1, remove_ids)
      w = np.delete(w, remove_ids)
      h = np.delete(h, remove_ids)
      
      if text.replace(' ', '') == '':
        continue
      x2 = x1 + w
      y2 = y1 + h
      x1 = np.min(x1)
      x2 = np.max(x2)
      y1 = np.min(y1)
      y2 = np.max(y2)
      coords += [[x1,y1,x2,y2]]
      texts += [text]
    
    return coords, texts
    
    

  def is_ticked(self, checkbox):
    h,w = checkbox.shape[:2]
    temp = cv2.threshold(checkbox, 150, 255, cv2.THRESH_BINARY_INV)[1]/255.0
    roi = temp[int(0.2*h):int(0.8*h), int(0.2*w):int(0.8*w)]
    return np.sum(roi) > 0.6*h*w * 0.01


def np_vec_no_jit_iou(checkbox_coor, allbox):
    def run(bboxes1, bboxes2):
        x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
        x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
        xA = np.maximum(x11, np.transpose(x21))
        yA = np.maximum(y11, np.transpose(y21))
        xB = np.minimum(x12, np.transpose(x22))
        yB = np.minimum(y12, np.transpose(y22))
        interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
        boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
        # boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
        iou = interArea / boxAArea
        return iou
    iou = run(checkbox_coor, allbox).ravel()
    return iou


def xml2csv(xml_path):
    """Convert XML to CSV

    Args:
        xml_path (str): Location of annotated XML file
    Returns:
        pd.DataFrame: converted csv file

    """
    print("xml to csv {}".format(xml_path))
    xml_list = []
    xml_df=pd.DataFrame()
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                    int(root.find('size')[0].text),
                    int(root.find('size')[1].text),
                    member[0].text,
                    int(member[4][0].text),
                    int(member[4][1].text),
                    int(member[4][2].text),
                    int(member[4][3].text)
                    )
            xml_list.append(value)
            column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
            xml_df = pd.DataFrame(xml_list, columns=column_name)
    except Exception as e:
        print('xml conversion failed:{}'.format(e))
        return pd.DataFrame(columns=['filename,width,height','class','xmin','ymin','xmax','ymax'])
    return xml_df
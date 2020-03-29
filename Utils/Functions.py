import cv2


def draw_box(image, coordinates, name='None', age='None', gender='None', bb_type=1):
    """
    Draw Bounding box on image
    :param image: np array of image
    :param coordinates: (x1,y1,x2,y2) or (x1,y1,w,h)
    :param name: (string) name to add
    :param age: (string) age to add
    :param gender: (string) gender to add
    :param bb_type: 1 for (x1,y1,x2,y2) | 2 for(x1,y1,w,h)
    :return: np.array with profile added
    """
    color = (0, 255, 0)
    bb_thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    font_thickness = 1
    left_bb = (coordinates[0], coordinates[1])
    right_bb = (coordinates[0] + coordinates[2], coordinates[1] + coordinates[3]) \
        if bb_type == 2 else (coordinates[2], coordinates[3])
    name = 'Name: {}'.format(name)
    age = 'Age: {}'.format(age)
    gender = 'Gender: {}'.format(gender)
    if coordinates[1] - 35 < 10:
        name_loc = (coordinates[0], coordinates[1] + 10)
        age_loc = (coordinates[0], coordinates[1] + 25)
        gender_loc = (coordinates[0], coordinates[1] + 40)
    else:
        name_loc = (coordinates[0], coordinates[1] - 35)
        age_loc = (coordinates[0], coordinates[1] - 20)
        gender_loc = (coordinates[0], coordinates[1] - 5)
    image = cv2.rectangle(image, left_bb, right_bb, color, bb_thickness)
    image = cv2.putText(image, name, name_loc, font, font_scale, color, font_thickness, cv2.LINE_AA)
    image = cv2.putText(image, gender, gender_loc, font, font_scale, color, font_thickness, cv2.LINE_AA)
    image = cv2.putText(image, age, age_loc, font, font_scale, color, font_thickness, cv2.LINE_AA)
    return image

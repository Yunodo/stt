#Function to encode letters into classes in array of strings

def encode_labels(text, d):
    """

    Inputs:
        text - python array of string
        d    - dictionary

    Outputs:
        big_arr    - python array of encoded strings represented as arrays, e.g. [[1,15,12,4], [2,3,1]]
        max_length - length of the longest encoded string

    """
    big_arr = []
    max_length = 0
    for string in text:
        small_arr = []
        for letter in string:
            if letter != '.' and letter != ',':
                small_arr.append(d.get(letter,27)) #encoding unknown characters as ''
        length = len(small_arr)
        if length > max_length:
            max_length = length
        big_arr.append((small_arr,length))
  return big_arr, max_length

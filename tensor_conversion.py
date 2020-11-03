def text_into_tensor(texts, max_length):

    """

    Inputs:
        texts - numpy arrays of encoded
        max_length - maximum length of the longest labels


    Outputs:
        tens    - tensors of shape [B, C], where B - batch size, C - number of classes
        lengths - lengths of labels before padding to the longest sequence

    """
    tens = torch.tensor(texts[0][0], dtype = torch.int16)
    tens = torch.nn.functional.pad(tens, (0, max_length - texts[0][1]), mode='constant', value=0)
    tens = torch.unsqueeze(tens, 0)
    lengths = torch.tensor(texts[0][1], dtype = torch.int16)
    lengths = torch.unsqueeze(lengths, 0)
    for i in range(1, len(texts)):
            ten = torch.tensor(texts[i][0], dtype = torch.int16)
            ten = torch.nn.functional.pad(ten, (0, max_length - texts[i][1]), mode='constant', value=0)
            ten = torch.unsqueeze(ten, 0)
            tens = torch.cat((tens,ten), 0)
            length = torch.tensor(texts[i][1], dtype = torch.int16)
            length = torch.unsqueeze(length, 0)
            lengths = torch.cat((lengths,length),0)
  return tens, lengths

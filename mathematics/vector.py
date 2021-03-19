def get_difference_item(array, current_data, index):
    """Computes differences between current data and the previous data from array"""
    if index <= 0:
        return [current_data.x, current_data.y, current_data.z]
    dx = current_data.x - array[index - 1][0]
    dy = current_data.y - array[index - 1][1]
    dz = current_data.z - array[index - 1][2]

    return [dx, dy, dz]

''' GRID '''
# returns a list of indices and tuples, describing the fields in the square-neighbourhood of x and y, in a distance measure where a disc of radius 1 is a square of 3x3 fields
def get_neighbourhood(x, y, radius):
    return 0

# returns a list of indices and tuples, describing the fields that can be reached within n_steps or less, excluding the own position (x,y)
# if n_steps == 1, the order is DOWN UP LEFT RIGHT
def get_step_neighbourhood(x, y, n_steps):
    neighb = []
    counter = 0
    for i in range(n_steps+1):
        for j in range(n_steps - i+1):
            if i == 0 and j == 0:
                continue
            # if one is zero, do not count +-zero twice:
            elif i == 0:
                for sign_j in [-1,1]:
                    neighb.append((counter, (x, y+sign_j*j)))
                    counter += 1
            elif j == 0:
                for sign_i in [-1,1]:
                    neighb.append((counter, (x+sign_i*i, y)))
                    counter += 1
            # case none is zero:
            else:
                for sign_i in [-1,1]:
                    for sign_j in [-1,1]:
                        neighb.append((counter, (x+sign_i*i, y+sign_j*j)))
                        counter += 1
    return neighb

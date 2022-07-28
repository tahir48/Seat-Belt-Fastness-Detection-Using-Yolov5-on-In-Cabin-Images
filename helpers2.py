def get_centroids(image,boxes_np,index):
    centroids = []
    for ind in index:
        x,y,w,h =  boxes_np[ind]
        centroidx = x + w/2
        centroidy = y + h/2
        centroids.append([int(centroidx),int(centroidy)])
    return centroids


def slope_detector(centroids):
    pass

def colinearity_detector(centroids):
    pass

def collinearity_value(X, Y, Z):
    "Calculates area of rectangle using determinant"
    value = X[0] * (Y[1] - Z[1]) + Y[0] * (Z[1] - X[1]) + Z[0] * (X[1] - Y[1])
    return value

def slope_value(X, Y):
    value = (Y[1]-X[1])/(Y[0]-X[0])
    return value
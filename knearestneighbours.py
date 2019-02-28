import numpy as np


class KNearestNeighbours(object):
    '''
    A class that represents the K-NearestNeighbours algorithm.
    '''

    def __init__(self, number_of_neighbours):
        '''
        Arguments:
            number_of_neighbours: the number of neighbours
                                  to consider when classifying
                                  a new point.
        '''
        self.number_of_neighbours = number_of_neighbours

        self.training_features = None
        self.training_labels = None

    def fit(self, X, y):
        '''
        Stores the training feature and labels for the use
        of classifying a new point.

        Arguments:  
            X: the training features
            y: the training labels
        '''
        self.training_features = X
        self.training_labels = y
        self.number_of_classes = len(np.unique(self.training_labels))

    def predict(self, X):
        '''
        Predicts the class of an array of new points. Calculates
        the euclidean distance between all points and the new point(s). 
        A label is decided for a new point by the majority class of its 
        neighbours

        Arguments:
            X: the unseen points to be predicted.

        Returns:
            predictions: the label predictions for the unseen points.
        '''

        # Store the predictions
        predictions = []

        # For each point in the unseen data.
        for example in X:

            # Store the neighbours.
            neighbours = []

            # Store the class proportions.
            class_proportions = np.zeros((self.number_of_classes, 1))

            # For each point in the training set.
            for i, j in zip(self.training_features, self.training_labels):

                # Calculate the Euclidean distance between training point
                # and unseen point
                distance = np.linalg.norm(i - example)
                neighbours.append((distance, j))

            # Sort the neighbours by distance
            sorted_neighbours = sorted(neighbours, key=lambda x: x[0])

            # Selected the top K neighbours.
            nearest_neighbours = sorted_neighbours[:self.number_of_neighbours]

            # For each class, count the number of neighbours
            # that have this class.
            for class_label in range(self.number_of_classes):
                number_class = len(
                    list(filter(lambda x: x[1] == class_label, nearest_neighbours)))

                class_proportions[class_label] = number_class

            # Choose the majority class as the prediction.
            prediction = np.argmax(class_proportions)

            # Append to predictions
            predictions.append(prediction)

        return predictions

    def accuracy(self, true_labels, predicted_labels):
        '''
        Calculates the accuracy of the KNN model. 

        Arguments:
            true_labels: the true labels of the data.
            predicted_labels: the predicted labels of the data.

        Returns:
            acc: the accuracy of the model.
        '''
        acc = np.mean(true_labels == predicted_labels)
        return acc

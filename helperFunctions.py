import random
from surprise import dataset,AlgoBase
from collections import defaultdict

# Also, a dummy Dataset class
class MyDataset(dataset.DatasetAutoFolds):

    def __init__(self, df, reader):

        self.raw_ratings = [(uid, iid, r, None) for (uid, iid, r) in
                            zip(df['userID'], df['itemID'], df['rating'])]
        self.reader=reader

class TopPop(AlgoBase):
    def __init__(self):

        AlgoBase.__init__(self)

    def fit(self,trainset):
        AlgoBase.fit(self,trainset)

        # Dictionary to store the number of ratings per movie
        ratings_per_movie = defaultdict(int)

        # Iterate through all ratings in the trainset
        for _, _, rating in trainset.all_ratings():
            movie_id = trainset.to_raw_iid(_)
            ratings_per_movie[movie_id] += 1

        # Sort the movies based on the number of ratings
        sorted_movies = sorted(ratings_per_movie.items(), key=lambda x: x[1], reverse=True)
        
        self.list = sorted_movies

        return self


    def estimate(self, u, i):

        return self.list

class MovieAvg(AlgoBase):
    def __init__(self):

        AlgoBase.__init__(self)

    def fit(self,trainset):
        AlgoBase.fit(self,trainset)

        movie_ratings = {}  # Dictionary to store movie id and its ratings
        movie_counts = {}   # Dictionary to store count of ratings for each movie

        # Iterate over each rating in the train set
        for _, _, rating in trainset.all_ratings():
            movie_id = trainset.to_raw_iid(_)
            if movie_id not in movie_ratings:
                movie_ratings[movie_id] = rating
                movie_counts[movie_id] = 1
            else:
                movie_ratings[movie_id] += rating
                movie_counts[movie_id] += 1

        average_ratings = []  # List to store tuples of movie id and its average rating

        # Calculate average ratings for each movie
        for movie_id, total_rating in movie_ratings.items():
            average_rating = total_rating / movie_counts[movie_id]
            average_ratings.append((movie_id, average_rating))

        average_ratings.sort(key=lambda x: x[1], reverse=True)
        self.list = average_ratings

        return self


    def estimate(self, u, i):

        return self.list

def calculate_long_tail(data):
    item_counts = data['itemID'].value_counts(normalize=True)
    items = []
    sum = 0
    for i, v in item_counts.items():
        if(sum < .3):
            items.append(i)
            sum += v
    long_tail = data[~data['itemID'].isin(items)]
    return long_tail


def evaluate_model(trainset, testset, model, N, five_star_num):
    # Train the model on the training set
    # model.fit(trainset)
    # Extract 5-star ratings from the test set
    test_5_star_ratings = [(uid, iid, r_ui) for (uid, iid, r_ui) in testset if r_ui == int(five_star_num)]

    # Initialize variables for precision and recall calculation
    hit_count = 0
    total_count = 0
    test_item_count = len(test_5_star_ratings)
    for uid, iid, rating in test_5_star_ratings:
        # Randomly select 1000 additional items unrated by user u
        # unrated_items = random_unrated_items(trainset, testset, uid, num_items=1000)
        unrated_items = random_unrated_items(trainset, uid, num_items=1000)
        # Predict the ratings for the test item i and the additional 1000 items
        p = model.predict(uid, iid, rating)
        predictions = [model.predict(uid, item) for item in unrated_items]
        predictions.append(p) # to get the 1001th item in the list
        # Form a ranked list by ordering all the 1001 items according to their predicted ratings
        ranked_list = sorted(predictions, key=lambda x: x.est, reverse=True)


        # print(f"Target User:{uid} Target Item:{iid} Rating:{rating}")
        # new_list = [(i,ranked_list[i].uid,ranked_list[i].iid,ranked_list[i].est) for i in range(len(ranked_list))]
        
        
        # item_list = [item.iid for item in ranked_list]

        # index = item_list.index(iid)
        # print(f"p:{index}")
        # print(new_list)
        


        # Form a top-N recommendation list
        top_N_recommendations = [item.iid for item in ranked_list[:N]]
        # Check if the test item is recommended within top N
        if iid in top_N_recommendations:
            hit_count += 1
        
        total_count += 1

        # Calculate recall and precision
    print(f"Hit count:{hit_count} item count:{test_item_count}")
    recall = hit_count / test_item_count
    precision = recall / N
    return recall, precision

def evaluate_baseline(testset,model,N,five_star_num):
     # Extract 5-star ratings from the test set
    test_5_star_ratings = [(uid, iid, r_ui) for (uid, iid, r_ui) in testset if r_ui == int(five_star_num)]

    # Initialize variables for precision and recall calculation
    list = model.list
    hit_count = 0
    total_count = 0
    test_item_count = len(test_5_star_ratings)
    for uid, iid, rating in test_5_star_ratings:
        
        top_N_recommendations = [rating[1] for rating in list[:N]]
        # Check if the test item is recommended within top N
        if iid in top_N_recommendations:
            hit_count += 1
        total_count += 1
    
    # Calculate recall and precision
    print(f"Hit count:{hit_count} item count:{test_item_count}")
    recall = hit_count / test_item_count
    precision = recall / N
    return recall, precision

# def random_unrated_items(trainset, testset, user_id, num_items=1000):
def random_unrated_items(trainset, user_id, num_items=1000):
    # Get all items in the trainset
    all_items = set(trainset.all_items())
    # Get items already rated by the user
    rated_items = set([item for item, _ in trainset.ur[user_id]])
    # rated_items2 = set([item for user,item,_ in testset if user == user_id])
    # rated_items = rated_items | rated_items2
    # Select unrated items
    unrated_items = all_items - rated_items
    # Convert to list and shuffle
    unrated_items = list(unrated_items)
    random.shuffle(unrated_items)
    # Take a slice of num_items
    unrated_items = unrated_items[:num_items]

    return unrated_items

def reservoir_sampling(lst, k):
    # Initialize reservoir with the first k elements
    reservoir = lst[:k]
    
    # Iterate over the remaining elements in the list
    for i in range(k, len(lst)):
        # Randomly select an index j from 0 to i
        j = random.randint(0, i)
        # If j is less than k, replace the element at j in the reservoir with the current element
        if j < k:
            reservoir[j] = lst[i]
    
    return reservoir
import random
import pandas as pd
from surprise.model_selection.split import train_test_split
from surprise import SVD,KNNBasic,KNNWithMeans,Reader,SVDpp
from surprise.model_selection import cross_validate
from random import sample 
from collections import Counter
import matplotlib.pyplot as plt
import tarfile
import os
from helperFunctions import MovieAvg,TopPop,evaluate_baseline,evaluate_model,MyDataset,calculate_long_tail,reservoir_sampling


print("----Prepping Data------")

# # CiaoDVD
# colnames = ["userID", "itemID", "movieCat", "reviewId", "rating", "tstamp"]
# path1 = "content/ciaodvd_dataset/movie-ratings.txt"
# df = pd.read_csv(path1, names=colnames)
# df = df.drop(df.columns[[2,3,5]], axis=1)
# reader = Reader(line_format='user item rating', rating_scale=(1, 5))
# ciaodata = MyDataset(df, reader)
# print("ciao: " + str(len(ciaodata.raw_ratings)))
# long_tail = calculate_long_tail(df)
# ciaolongtaildata = MyDataset(long_tail, reader)
# print("ciao longtail: " + str(len(ciaolongtaildata.raw_ratings)))

# # FilmTrust
# colnames = ["userID", "itemID", "rating"]
# path2 = "content/filmtrust_dataset/ratings.txt"
# df2 = pd.read_csv(path2, sep=" ", header=None, names = colnames)
# reader = Reader(line_format='user item rating', rating_scale=(1, 4))
# fTdata = MyDataset(df2, reader)
# print("filmtrust: " + str(len(fTdata.raw_ratings)))
# long_tail = calculate_long_tail(df2)
# fTlongtaildata = MyDataset(long_tail, reader)
# print("filmtrust longtail: " + str(len(fTlongtaildata.raw_ratings)))

# MovieLens
colnames = ["userID", "itemID", "rating", "timestamp"]
path3 = "content/movielens-25m/ratings.csv"
df3 = pd.read_csv(path3, names=colnames, skiprows=1, dtype={'userID': int, 'itemID': int, 'rating': float, 'timestamp': str})
df3 = df3.drop(df3.columns[[3]], axis=1)
df3_sampled = df3.sample(n=2000000)

reader = Reader(line_format='user item rating', rating_scale=(1, 5))

# number_of_ratings = 1000 # number of files to base dataset on!!! (currently 10% @5M)
# selected_ratings = random.sample(range(len(df3)), number_of_ratings)
# df3_sampled = df3.iloc[selected_ratings]

MVdata = MyDataset(df3_sampled, reader)
print("movieLens: " + str(len(MVdata.raw_ratings)))
long_tail = calculate_long_tail(df3_sampled)
MVlongtaildata = MyDataset(long_tail, reader)
print("movieLens longtail: " + str(len(MVlongtaildata.raw_ratings)))



# Netflix
# COMMENT IN FOR NEW USERS
# path4 = "content/netflix/training_set.tar"
# extract_folder = "content/netflix/training_set"
# with tarfile.open(path4, 'r') as tf: # ONLY NEED TO DO ONCE, CAN COMMENT OUT ONCE EXTRACTION FOLDER BUILT
#     tf.extractall(path=extract_folder)

# file_list = os.listdir(extract_folder) # full file list
# number_of_files = 1700 # number of files to base dataset on!!! (currently 10% @~10M with 1750)
# index_list = list(range(len(file_list)))
# shuffled_indices = random.sample(index_list, k=number_of_files)
# selected_files = [file_list[i] for i in shuffled_indices[:number_of_files]] # get random selection of total files

# data = []
# for file_name in file_list:
#     print(file_name)
#     if not file_name.endswith('.txt'):
#         continue
#     file_path = os.path.join(extract_folder, file_name)
#     movie_id = file_name.split('_')[1].split('.')[0] # movie id in the file_name

#     with open(file_path, 'r') as file:
#         next(file) # skip first "movie id:"" line
#         for line in file:
#             user_id, rating, _ = line.strip().split(',')
#             data.append((user_id, movie_id, rating))


# # Size of the test sample (e.g., 5%)
# test_sample_size = int(len(data) * 0.05)

# # Randomly partition the large list into a smaller test sample using reservoir sampling
# test_sample = reservoir_sampling(data, test_sample_size)

# with open('somefile.txt', 'w') as the_file:
#     for line in test_sample:
#         user_id = line[0]
#         movie_id = line[1]
#         rating = line[2]
#         string = user_id + "," + movie_id + "," + rating + "\n"
#         the_file.write(string)

# data = test_sample

colnames = ["userID", "itemID", "rating"]
path4 = "netflixsample.txt"
df4 = pd.read_csv(path4, sep=",", header=None, names = colnames)
df4 = df4.sample(frac=.8)
reader = Reader(line_format='user item rating', rating_scale=(1, 5))
netflixdata = MyDataset(df4, reader)
print("netflix: " + str(len(netflixdata.raw_ratings)))
long_tail = calculate_long_tail(df4)
netflixlongtaildata = MyDataset(long_tail, reader)
print("netflix longtail: " + str(len(netflixlongtaildata.raw_ratings)))




print("----Creating train and test sets------")
# # Ciao
# Ctrain, Ctest = train_test_split(ciaodata, test_size=.014)
# CtrainLT, CtestLT = train_test_split(ciaolongtaildata, test_size=.014)

# # Filmtrust
# fTtrain, fTtest = train_test_split(fTdata, test_size=.10)
# fTtrainLT, fTtestLT = train_test_split(fTlongtaildata, test_size=.014)

# MovieLens
MVtrain, MVtest = train_test_split(MVdata, test_size=.014)
MVtrainLT, MVtestLT = train_test_split(MVlongtaildata, test_size=.014)

# Netflix
Netflixtrain, Netflixtest = train_test_split(netflixdata, test_size=.014)
NetflixtrainLT, NetflixtestLT = train_test_split(netflixlongtaildata, test_size=.014)



print("----Algorithms------")
# Algorithm List

# Highest Movie Average
movieAvg = MovieAvg()
# Top Popular Movies
topPop = TopPop()
# Pearson Neighborhood
sim_options_CorNgbr = {'user_based' : False,
                    'name' : 'pearson'}
CorNgbr = KNNWithMeans(sim_options=sim_options_CorNgbr)
# Non Normalized Cosine Neighborhood
sim_options_NNCosNgbr = {'user_based' : False,
                    'name' : 'cosine'}
NNCosNgbr = KNNBasic(sim_options=sim_options_NNCosNgbr)   
# SVD++   
SVDpp = SVDpp()
# SVD with 50 features
PureSVD50 = SVD(n_factors=50)
# SVD with 150 features
PureSVD150 = SVD(n_factors=150)
# Asymetric SVD
AsymSVD = SVD()


# # Ciao
# # Loop through all items and long tail
# cdatasets = [(Ctrain,Ctest,"All Items"),(CtrainLT,CtestLT,"Long Tail")]
# for dataset in cdatasets:
#     # Create trained models
#     cmovieavg = movieAvg.fit(dataset[0])
#     ctoppop = topPop.fit(dataset[0])
#     cnn = NNCosNgbr.fit(dataset[0])
#     cngbr = CorNgbr.fit(dataset[0])
#     csvd50 = PureSVD50.fit(dataset[0])
#     csvd150 = PureSVD150.fit(dataset[0])
#     csvdpp = SVDpp.fit(dataset[0])
#     casvd = AsymSVD.fit(dataset[0])
#     # List to hold all of the models and their names
#     models = [(cmovieavg,"MovieAvg"),
#             (ctoppop, "TopPop"),
#             (cnn,"NNCorNgbr"),
#             (cngbr,"CorNgbr"),
#             (csvd50,"SVD50"),
#             (csvd150,"SVD150"),
#             (csvdpp,"SVD++"),
#             (casvd,"AsymSVD")]

#     Ns = [i for i in range(1,21,1)]
#     recalls = []
#     precisions = []
#     # Loop through each model and run each model 20 times
#     print("------Calculating Recall and Precision for Caio-------")
#     i = 0
#     for model,name in models:
#         print(i)
#         i += 1
#         print(len(Ns))
#         print(len(recalls))
#         print(f"Model used is:{name}")
#         # List to hold recall and precision
#         recall = []
#         precision = []
#         for n in range(1,21,1):
#             print(f"N:{n}")
#             # Check is baseline model
#             if(model == cmovieavg or model == ctoppop):
#                 crecall, cprecision = evaluate_baseline(dataset[1], model, n, 5.0)
#             else:
#                 crecall, cprecision = evaluate_model(dataset[0], dataset[1], model, n, 5.0)
#             recall.append(crecall)
#             precision.append(cprecision)

#         recalls.append(recall)
#         precisions.append(precision)

#         r_strings = [str(item) for item in recall]
#         p_strings = [str(item) for item in precision]
#         with open('data.txt', 'a') as the_file:
#             string = name + "\n"
#             string = the_file.write(string)
#             string = ','.join(r_strings)
#             string += "\n"
#             the_file.write(string)
#             string = ','.join(p_strings)
#             string += "\n"
#             the_file.write(string)

    # Plot N by Recall
    # plt.plot(Ns, recalls[0], label = "MovieAvg")
    # plt.plot(Ns, recalls[1], label = "TopPop")
    # plt.plot(Ns, recalls[2], label = "NNCorNgbr")
    # plt.plot(Ns, recalls[3], label = "CorNgbr")
    # plt.plot(Ns, recalls[4], label = "SVD50")
    # plt.plot(Ns, recalls[5], label = "SVD150")
    # plt.plot(Ns, recalls[6], label = "SVD++")
    # plt.plot(Ns, recalls[7], label = "AsymSVD")
    # plt.xlabel('N')
    # plt.ylabel('Recall')
    # plt.title(dataset[2])
    # plt.legend()
    # plt.show()
    # # Plot Recall by Precision
    # plt.plot(recalls[0],precisions[0],label = "MovieAvg")
    # plt.plot(recalls[1],precisions[1],label = "TopPop")
    # plt.plot(recalls[2],precisions[2],label = "NNCorNgbr")
    # plt.plot(recalls[3],precisions[3],label = "CorNgbr")
    # plt.plot(recalls[4],precisions[4],label = "SVD50")
    # plt.plot(recalls[5],precisions[5],label = "SVD150")
    # plt.plot(recalls[6],precisions[6],label = "SVD++")
    # plt.plot(recalls[7],precisions[7],label = "AsymSVD")
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title(dataset[2])
    # plt.legend()
    # plt.show()


# # # FilmTrust
# fdatasets = [(fTtrain,fTtest,"All Items"),(fTtrainLT,fTtestLT,"Long Tail")]
# for dataset in fdatasets:
#     # Create trained models
#     fTmovieavg = movieAvg.fit(dataset[0])
#     fTtoppop = topPop.fit(dataset[0])
#     fTnn = NNCosNgbr.fit(dataset[0])
#     fTngbr = CorNgbr.fit(dataset[0])
#     fTsvd50 = PureSVD50.fit(dataset[0])
#     fTsvd150 = PureSVD150.fit(dataset[0])
#     fTsvdpp = SVDpp.fit(dataset[0])
#     fTasvd = AsymSVD.fit(dataset[0])
#     # List to hold all of the models and their names
#     models = [(fTmovieavg,"MovieAvg"),
#             (fTtoppop, "TopPop"),
#             (fTnn,"NNCorNgbr"),
#             (fTngbr,"CorNgbr"),
#             (fTsvd50,"SVD50"),
#             (fTsvd150,"SVD150"),
#             (fTsvdpp,"SVD++"),
#             (fTasvd,"AsymSVD")]


#     fTrecall, fTprecision = evaluate_model(dataset[0], dataset[1], fTngbr, 10, 4.0)


#     Ns = [i for i in range(1,21,1)]
#     recalls = []
#     precisions = []
#     # Loop through each model and run each model 20 times
#     print("------Calculating Recall and Precision for FilmTrust-------")
#     for model,name in models:
#         print(f"Model used is:{name}")
#         # List to hold recall and precision
#         recall = []
#         precision = []
#         for n in range(1,21,1):
#             print(f"N:{n}")
#             # Check is baseline model
#             if(model == fTmovieavg or model == fTtoppop):
#                 fTrecall, fTprecision = evaluate_baseline(dataset[1], model, n, 4.0)
#             else:
#                 fTrecall, fTprecision = evaluate_model(dataset[0], dataset[1], model, n, 4.0)
#             recall.append(fTrecall)
#             precision.append(fTprecision)
#         recalls.append(recall)
#         precisions.append(precision)
#         r_strings = [str(item) for item in recall]
#         p_strings = [str(item) for item in precision]
#         with open('data.txt', 'a') as the_file:
#             string = name + "\n"
#             string = the_file.write(string)
#             string = ','.join(r_strings)
#             string += "\n"
#             the_file.write(string)
#             string = ','.join(p_strings)
#             string += "\n"
#             the_file.write(string)


    # plt.plot(Ns, recalls[0], label = "MovieAvg")
    # plt.plot(Ns, recalls[1], label = "TopPop")
    # plt.plot(Ns, recalls[2], label = "NNCorNgbr")
    # plt.plot(Ns, recalls[3], label = "CorNgbr")
    # plt.plot(Ns, recalls[4], label = "SVD50")
    # plt.plot(Ns, recalls[5], label = "SVD150")
    # plt.plot(Ns, recalls[6], label = "SVD++")
    # plt.plot(Ns, recalls[7], label = "AsymSVD")
    # plt.xlabel('N')
    # plt.ylabel('Recall')
    # plt.title(dataset[2])
    # plt.legend()
    # plt.show()
    # #Plot Recall by Precision
    # plt.plot(recalls[0],precisions[0],label = "MovieAvg")
    # plt.plot(recalls[1],precisions[1],label = "TopPop")
    # plt.plot(recalls[2],precisions[2],label = "NNCorNgbr")
    # plt.plot(recalls[3],precisions[3],label = "CorNgbr")
    # plt.plot(recalls[4],precisions[4],label = "SVD50")
    # plt.plot(recalls[5],precisions[5],label = "SVD150")
    # plt.plot(recalls[6],precisions[6],label = "SVD++")
    # plt.plot(recalls[7],precisions[7],label = "AsymSVD")
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title(dataset[2])
    # plt.legend()
    # plt.show()


# # MovieLens
# mdatasets = [(MVtrain,MVtest,"All Items"),(MVtrainLT,MVtestLT,"Long Tail")]
# for dataset in mdatasets:
#     # Create trained models
#     mvmovieavg = movieAvg.fit(dataset[0])
#     mvtoppop = topPop.fit(dataset[0])
#     mvnn = NNCosNgbr.fit(dataset[0])
#     mvngbr = CorNgbr.fit(dataset[0])
#     mvsvd50 = PureSVD50.fit(dataset[0])
#     mvsvd150 = PureSVD150.fit(dataset[0])
#     mvsvdpp = SVDpp.fit(dataset[0])
#     mvasvd = AsymSVD.fit(dataset[0])
#     # List to hold all of the models and their names
#     models = [(mvmovieavg,"MovieAvg"),
#             (mvtoppop, "TopPop"),
#             (mvnn,"NNCorNgbr"),
#             (mvngbr,"CorNgbr"),
#             (mvsvd50,"SVD50"),
#             (mvsvd150,"SVD150"),
#             (mvsvdpp,"SVD++"),
#             (mvasvd,"AsymSVD")]

#     Ns = [i for i in range(1,21,1)]
#     recalls = []
#     precisions = []
#     # Loop through each model and run each model 20 times
#     print("------Calculating Recall and Precision for MovieLens-------")
#     for model,name in models:
#         print(f"Model used is:{name}")
#         # List to hold recall and precision
#         recall = []
#         precision = []
#         for n in range(1,21,1):
#             print(f"N:{n}")
#             # Check is baseline model
#             if(model == mvmovieavg or model == mvtoppop):
#                 mvrecall, mvprecision = evaluate_baseline(dataset[1], model, n, 5.0)
#             else:
#                 mvrecall, mvprecision = evaluate_model(dataset[0], dataset[1], model, n, 5.0)
#             recall.append(mvrecall)
#             precision.append(mvprecision)
#         recalls.append(recall)
#         precisions.append(precision)
#         r_strings = [str(item) for item in recall]
#         p_strings = [str(item) for item in precision]
#         with open('data.txt', 'a') as the_file:
#             string = name + "\n"
#             string = the_file.write(string)
#             string = ','.join(r_strings)
#             string += "\n"
#             the_file.write(string)
#             string = ','.join(p_strings)
#             string += "\n"
#             the_file.write(string)

    # Plot N by Recall
    # plt.plot(Ns, recalls[0], label = "MovieAvg")
    # plt.plot(Ns, recalls[1], label = "TopPop")
    # plt.plot(Ns, recalls[2], label = "NNCorNgbr")
    # plt.plot(Ns, recalls[3], label = "CorNgbr")
    # plt.plot(Ns, recalls[4], label = "SVD50")
    # plt.plot(Ns, recalls[5], label = "SVD150")
    # plt.plot(Ns, recalls[6], label = "SVD++")
    # plt.plot(Ns, recalls[7], label = "AsymSVD")
    # plt.xlabel('N')
    # plt.ylabel('Recall')
    # plt.title(dataset[2])
    # plt.legend()
    # plt.show()
    # # Plot Recall by Precision
    # plt.plot(recalls[0],precisions[0],label = "MovieAvg")
    # plt.plot(recalls[1],precisions[1],label = "TopPop")
    # plt.plot(recalls[2],precisions[2],label = "NNCorNgbr")
    # plt.plot(recalls[3],precisions[3],label = "CorNgbr")
    # plt.plot(recalls[4],precisions[4],label = "SVD50")
    # plt.plot(recalls[5],precisions[5],label = "SVD150")
    # plt.plot(recalls[6],precisions[6],label = "SVD++")
    # plt.plot(recalls[7],precisions[7],label = "AsymSVD")
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title(dataset[2])
    # plt.legend()
    # plt.show()

# Netflix
ndatasets = [(Netflixtrain,Netflixtest,"All Items"),(NetflixtrainLT,NetflixtestLT,"Long Tail")]
for dataset in ndatasets:
    # Create trained models
    netmovieavg = movieAvg.fit(dataset[0])
    nettoppop = topPop.fit(dataset[0])
    netnn = NNCosNgbr.fit(dataset[0])
    netngbr = CorNgbr.fit(dataset[0])
    netsvd50 = PureSVD50.fit(dataset[0])
    netsvd150 = PureSVD150.fit(dataset[0])
    netsvdpp = SVDpp.fit(dataset[0])
    netasvd = AsymSVD.fit(dataset[0])
    # List to hold all of the models and their names
    models = [(netmovieavg,"MovieAvg"),
            (nettoppop, "TopPop"),
            (netnn,"NNCorNgbr"),
            (netngbr,"CorNgbr"),
            (netsvd50,"SVD50"),
            (netsvd150,"SVD150"),
            (netsvdpp,"SVD++"),
            (netasvd,"AsymSVD")]

    Ns = [i for i in range(1,21,1)]
    recalls = []
    precisions = []
    # Loop through each model and run each model 20 times
    print("------Calculating Recall and Precision for Netflix-------")
    for model,name in models:
        print(f"Model used is:{name}")
        # List to hold recall and precision
        recall = []
        precision = []
        for n in range(1,21,1):
            print(f"N:{n}")
            # Check is baseline model
            if(model == netmovieavg or model == nettoppop):
                nrecall, nprecision = evaluate_baseline(dataset[1], model, n, 5.0)
            else:
                nrecall, nprecision = evaluate_model(dataset[0], dataset[1], model, n, 5.0)
            recall.append(nrecall)
            precision.append(nprecision)
        recalls.append(recall)
        precisions.append(precision)
        r_strings = [str(item) for item in recall]
        p_strings = [str(item) for item in precision]
        with open('data.txt', 'a') as the_file:
            string = name + "\n"
            string = the_file.write(string)
            string = ','.join(r_strings)
            string += "\n"
            the_file.write(string)
            string = ','.join(p_strings)
            string += "\n"
            the_file.write(string)

    # Plot N by Recall
    # plt.plot(Ns, recalls[0], label = "MovieAvg")
    # plt.plot(Ns, recalls[1], label = "TopPop")
    # plt.plot(Ns, recalls[2], label = "NNCorNgbr")
    # plt.plot(Ns, recalls[3], label = "CorNgbr")
    # plt.plot(Ns, recalls[4], label = "SVD50")
    # plt.plot(Ns, recalls[5], label = "SVD150")
    # plt.plot(Ns, recalls[6], label = "SVD++")
    # plt.plot(Ns, recalls[7], label = "AsymSVD")
    # plt.xlabel('N')
    # plt.ylabel('Recall')
    # plt.title(dataset[2])
    # plt.legend()
    # plt.show()
    # # Plot Recall by Precision
    # plt.plot(recalls[0],precisions[0],label = "MovieAvg")
    # plt.plot(recalls[1],precisions[1],label = "TopPop")
    # plt.plot(recalls[2],precisions[2],label = "NNCorNgbr")
    # plt.plot(recalls[3],precisions[3],label = "CorNgbr")
    # plt.plot(recalls[4],precisions[4],label = "SVD50")
    # plt.plot(recalls[5],precisions[5],label = "SVD150")
    # plt.plot(recalls[6],precisions[6],label = "SVD++")
    # plt.plot(recalls[7],precisions[7],label = "AsymSVD")
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title(dataset[2])
    # plt.legend()
    # plt.show()
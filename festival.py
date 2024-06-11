import csv
import math
import pandas as pd


def tfidf(file_path, terms, festival=True):
    names = []

    tf_idf_res = {term: [] for term in terms}  # will hold the final results - tf_idf and temporarily the tf results

    num_of_docs_in_corpus = 0
    terms_total_rows_count_dict = {term: 0 for term in terms}

    file = open(file_path, mode='r')
    csv_reader = csv.reader(file, delimiter=',')
    next(csv_reader)  # skipping first line of headers

    for row in csv_reader:
        if festival:
            desc = row[7].replace(',', '').replace('-', ' ').lower().split(' ')
        else:
            desc = row[1].replace(',', '').replace('-', ' ').lower().split(' ')

        tf_res_per_row_tmp = {term: 0 for term in terms}  # temp param to help count the tf value for each term in a row
        for term in terms:
            if term in desc:
                terms_total_rows_count_dict[term] += 1  # counting number of rows each term appears in
                tf_res_per_row_tmp[term] += 1  # counting number of appearances of each term in each row

        # counting tf for each row (for each festival)
        num_of_terms_in_doc = len(desc)
        for term in tf_res_per_row_tmp:
            tf_idf_res[term].append(tf_res_per_row_tmp[term] / num_of_terms_in_doc)

        names.append(row[0])  # saving name of the row
        num_of_docs_in_corpus += 1  # counting number of rows

    # calculating final result:
    for term in terms:
        idf_val = math.log(num_of_docs_in_corpus / terms_total_rows_count_dict[term], 10)
        tf_idf_res[term] = [x * idf_val for x in tf_idf_res[term]]

    festivals_tf_idf = pd.DataFrame(tf_idf_res, index=names)

    return festivals_tf_idf


if __name__ == '__main__':
    path = r'C:\Users\Admin\Desktop\Uni\Courses\haystack\music_festivals.csv'
    terms = ['annual', 'music', 'festival', 'soul', 'jazz', 'belgium', 'hungary', 'israel', 'rock', 'dance', 'desert',
             'electronic', 'arts']
    print(tfidf(path, terms).to_string())

# a (2)
# soul:
# num of docs contains term - 1
# num of docs in corpus - 11
# idf = log (11 / 1) = 1.041
# tf_per_festival = [0/10, 0/8, 0/8, 0/9, 0/13, 0/10, 0/10, 0/8, 0/11, 1/12, 0/12]
# tf_idf_res_soul = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0867, 0]


# festival
# num of docs contains term - 11
# num of docs in corpus - 11
# idf = log (11 / 11) = 0
# tf_per_festival = [1/10, 1/8, 1/8, 1/9, 1/13, 1/10, 1/10, 1/8, 1/11, 1/12, 1/12]
# tf_idf_res_soul = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# the tf-idf of soul and festival expected to be very different because of the way the idf is calculated. the fact that
# festival appears in every row, zeroes the idf in contrast that soul appears in only 1 row which maximizes the idf


# todo need to write it clearer (for each row)
# (b)
# first of all we will normalize the number of participants by dividing the given number by the number of the world
# population at that year or by the population in each country (if it's a local festival without foreign visitors)
# maybe we can divide the population by the size of the country in square meters
# 2nd we will normalize the price to the worth of it be today, for example 700 USD in 1970 would be 5656 USD today



import csv
import math
import pandas as pd


def tfidf(file_path, terms, festival=True):
    names = []

    # will hold the final results - tf_idf and temporarily the tf results
    tf_idf_res = {term: [] for term in terms}

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

        # counting tf for each row
        for term in tf_res_per_row_tmp:
            tf_idf_res[term].append(tf_res_per_row_tmp[term])

        names.append(row[0])  # saving name of the row
        num_of_docs_in_corpus += 1  # counting number of rows

    # calculating final result:
    for term in terms:
        idf_val = math.log(num_of_docs_in_corpus / terms_total_rows_count_dict[term], 10)
        tf_idf_res[term] = [x * idf_val for x in tf_idf_res[term]]

    festivals_tf_idf = pd.DataFrame(tf_idf_res, index=names)

    return festivals_tf_idf


def question_7_a_1():
    path = r'music_festivals.csv'
    terms = ['annual', 'music', 'festival', 'soul', 'jazz', 'belgium', 'hungary', 'israel', 'rock', 'dance', 'desert',
             'electronic', 'arts']
    print(tfidf(path, terms).to_string())


if __name__ == '__main__':
    question_7_a_1()


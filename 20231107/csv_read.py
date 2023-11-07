import csv

file = 'exchange_rates.csv'

result = []


with open(file, mode='r', encoding='utf-8') as file:
    # csv_reader = csv.reader(file, delimiter=',')
    # csv_write = csv.writer()

    # print(csv_reader)
    #
    # for row in csv_reader:
    #     result.append(row)

    csv_reader = csv.DictReader(file, delimiter=',')

    for i in csv_reader:
        print(i)


print(csv_reader)

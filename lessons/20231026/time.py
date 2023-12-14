import time

timestamp = time.time()  # Поточний час
local_time = time.localtime(timestamp)  # Перетворення часу на локальний часовий пояс

print(local_time, type(local_time))
print(local_time.tm_year, local_time.tm_mday, local_time.tm_mon) # отримати складову часу за допомогою дот нотації
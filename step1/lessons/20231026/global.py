count = 1  # global

def increment_value():
    # local
    global count  # global var
    count += 1


print(count)

increment_value()

print(count)